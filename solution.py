import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "test" / "test" / "images"
TEMPLATE = ROOT / "test" / "sample_submission.csv"
MODEL = ROOT / "models" / "PP-LCNet_x0_25_textline_ori.onnx"
OUTPUT = ROOT / "submission.csv"

IMAGE_SIZE = (160, 80)  # OpenCV задаёт размер как (ширина, высота).
BATCH_SIZE = 64
CLASS_180_INDEX = 1  # Порядок классов в модели: 0_degree, затем 180_degree.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_template_ids(path: Path) -> list[str]:
    """Читает идентификаторы в требуемом порядке из sample_submission.csv."""
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_id", "p_180"]:
            raise ValueError("Шаблон должен содержать колонки image_id,p_180")
        image_ids = [row["image_id"] for row in reader]

    if any(not image_id for image_id in image_ids):
        raise ValueError("В шаблоне есть пустой image_id")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("В шаблоне есть повторяющиеся image_id")
    return image_ids


def load_batch(paths: list[Path]) -> np.ndarray:
    """Загружает изображения и приводит их к входному формату модели."""
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")

        # Модель обучена на RGB: OpenCV читает изображение в порядке BGR.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        # Масштабируем пиксели в [0, 1], нормализуем каналы и ставим каналы первыми.
        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        images.append(image.transpose(2, 0, 1))  # HWC -> CHW

    return np.stack(images)


def write_submission(path: Path, rows: list[tuple[str, float]]) -> None:
    """Сохраняет предсказания в формате, который ожидает проверяющая система."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "p_180"])
        writer.writerows((image_id, f"{p_180:.8f}") for image_id, p_180 in rows)


def run_inference(
    model_path: Path = MODEL,
    output_path: Path = OUTPUT,
    image_dir: Path = IMAGE_DIR,
    template_path: Path = TEMPLATE,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[str, float]]:
    """Получает p_180 для изображений и записывает CSV с предсказаниями."""
    if limit is not None and limit < 1:
        raise ValueError("limit должен быть больше нуля")
    if batch_size < 1:
        raise ValueError("batch_size должен быть больше нуля")
    if not model_path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    image_paths = sorted(image_dir.glob("*.png"))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"В папке нет PNG-изображений: {image_dir}")

    image_ids = [path.stem for path in image_paths]
    template_ids = read_template_ids(template_path)
    expected_ids = template_ids if limit is None else template_ids[: len(image_ids)]
    if image_ids != expected_ids:
        raise ValueError("Имена изображений не совпадают с порядком в шаблоне")

    # Ограничиваем число потоков CPU, чтобы запуск был предсказуем по ресурсам.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=options)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rows = []
    started = perf_counter()
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = load_batch(batch_paths)
        scores = session.run([output_name], {input_name: batch})[0]

        # Защищаемся от неожиданного выхода, прежде чем трактовать его как вероятность.
        if scores.shape != (len(batch_paths), 2):
            raise ValueError(f"Неожиданная форма выхода модели: {scores.shape}")
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Модель вернула некорректные вероятности")
        if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Вероятности классов модели не дают сумму 1")

        rows.extend(
            (path.stem, float(score))
            for path, score in zip(
                batch_paths, scores[:, CLASS_180_INDEX], strict=True
            )
        )

        processed = start + len(batch_paths)
        if processed % (batch_size * 16) == 0 or processed == len(image_paths):
            print(f"Обработано {processed} из {len(image_paths)}")

    write_submission(output_path, rows)
    print(f"Сохранён файл {output_path}; строк: {len(rows)}")
    print(f"Время: {perf_counter() - started:.1f} с")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Предсказывает вероятность поворота текста на 180 градусов."
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL, help="Путь к ONNX-модели"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="Куда сохранить CSV"
    )
    parser.add_argument(
        "--limit", type=int, help="Обработать только первые N изображений"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Размер пакета"
    )
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        output_path=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
