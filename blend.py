import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SMALL_FILE = ROOT / "submission.csv"
LARGE_FILE = ROOT / "submission_x1.csv"
OUTPUT_FILE = ROOT / "submission_blend.csv"

# Весовые коэффициенты выбраны по результатам отправок на платформу.
SMALL_WEIGHT = 0.715
LARGE_WEIGHT = 0.285


def read_predictions(path: Path) -> list[tuple[str, float]]:
    """Читает пары (image_id, p_180) и проверяет значения вероятностей."""
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_id", "p_180"]:
            raise ValueError(f"Неверные колонки в {path.name}")
        rows = [(row["image_id"], float(row["p_180"])) for row in reader]

    if not rows:
        raise ValueError(f"В файле нет предсказаний: {path}")
    if len({image_id for image_id, _ in rows}) != len(rows):
        raise ValueError(f"В файле есть повторяющиеся image_id: {path.name}")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for _, p in rows):
        raise ValueError(f"Некорректная вероятность в {path.name}")
    return rows


def write_predictions(path: Path, rows: list[tuple[str, float]]) -> None:
    """Записывает смешанные вероятности в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "p_180"])
        writer.writerows((image_id, f"{p_180:.8f}") for image_id, p_180 in rows)


def blend_predictions(
    small_path: Path = SMALL_FILE,
    large_path: Path = LARGE_FILE,
    output_path: Path = OUTPUT_FILE,
    small_weight: float = SMALL_WEIGHT,
    large_weight: float = LARGE_WEIGHT,
) -> list[tuple[str, float]]:
    """Смешивает p_180 по формуле w_small*p_small + w_large*p_large."""
    if small_weight < 0 or large_weight < 0:
        raise ValueError("Веса моделей не могут быть отрицательными")
    if not math.isclose(small_weight + large_weight, 1.0, abs_tol=1e-9):
        raise ValueError("Сумма весов моделей должна быть равна 1")

    small = read_predictions(small_path)
    large = read_predictions(large_path)
    small_ids = [image_id for image_id, _ in small]
    large_ids = [image_id for image_id, _ in large]
    if small_ids != large_ids:
        raise ValueError("Списки изображений в файлах не совпадают")

    # Усредняем вероятности, а не превращаем каждую модель в ответ 0 или 1.
    rows = [
        (image_id, small_weight * p_small + large_weight * p_large)
        for (image_id, p_small), (_, p_large) in zip(small, large, strict=True)
    ]
    write_predictions(output_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Смешивает предсказания компактной и крупной моделей."
    )
    parser.add_argument(
        "--small", type=Path, default=SMALL_FILE, help="CSV компактной модели"
    )
    parser.add_argument(
        "--large", type=Path, default=LARGE_FILE, help="CSV крупной модели"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help="Куда сохранить смешанные предсказания"
    )
    args = parser.parse_args()

    rows = blend_predictions(args.small, args.large, args.output)
    print(f"Сохранён {args.output}: {len(rows)} предсказаний")


if __name__ == "__main__":
    main()
