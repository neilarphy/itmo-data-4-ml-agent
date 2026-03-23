"""
Сравнительный отчёт до/после чистки.

Использование:
  python reporter.py --before data/raw/combined.parquet \
                     --after data/cleaned/cleaned.parquet \
                     --problems data/detective/problems.json \
                     --strategy balanced \
                     --output data/detective/comparison.md
"""
import argparse
import json
from datetime import datetime
import pandas as pd


def make_change(before: float, after: float, inverse: bool = False) -> str:
    """Форматировать изменение: красный если хуже, зелёный если лучше (в md нет цвета, просто знак)."""
    if before == 0:
        return "—"
    delta = (after - before) / before * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"


def reporter(before_path: str, after_path: str, problems_path: str, strategy: str, output_path: str):
    df_before = pd.read_parquet(before_path)
    df_after = pd.read_parquet(after_path)

    with open(problems_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    issues = problems.get("issues", {})
    n_before = len(df_before)
    n_after = len(df_after)

    # Считать метрики после
    missing_after = int(df_after["text"].isnull().sum() + (df_after["text"].str.strip() == "").sum()) if "text" in df_after.columns else 0
    dup_after = int(df_after.duplicated(subset=["text"]).sum()) if "text" in df_after.columns else 0

    lengths_after = df_after["text"].str.len() if "text" in df_after.columns else pd.Series(dtype=float)
    if len(lengths_after) > 0 and lengths_after.std() > 0:
        z = (lengths_after - lengths_after.mean()) / lengths_after.std()
        outliers_after = int((z.abs() > 3).sum())
    else:
        outliers_after = 0

    # Метрики до
    miss_before = sum(v["count"] for v in issues.get("missing", {}).values())
    dup_before = issues.get("duplicates", {}).get("count", 0)
    outliers_before = issues.get("outliers", {}).get("z3_count", 0)

    imbalance_before = issues.get("class_imbalance", {}).get("imbalance_ratio")
    label_after = df_after["label"].value_counts() if "label" in df_after.columns else pd.Series()
    labeled_after = label_after[label_after.index != "unlabeled"] if "unlabeled" in label_after.index else label_after
    imbalance_after = round(labeled_after.max() / labeled_after.min(), 2) if len(labeled_after) >= 2 and labeled_after.min() > 0 else None

    # Обоснование стратегии
    justifications = {
        "aggressive": "Выбрана агрессивная стратегия — максимальная чистота датасета. Подходит когда данных достаточно и критична точность модели: каждый грязный пример может сдвинуть метрики.",
        "conservative": "Выбрана консервативная стратегия — сохранить как можно больше данных. Подходит когда примеров мало и важен объём: модель лучше обобщает на большей выборке, даже с артефактами.",
        "balanced": "Выбрана сбалансированная стратегия — оптимальный компромисс. Убирает явный мусор (дубли, пустые строки, экстремальные выбросы) без потери значимых пограничных примеров.",
    }

    lines = [
        "# Quality Guard — Сравнительный отчёт",
        f"\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Стратегия чистки: **{strategy}**",
        "\n---\n",
        "## Метрики до/после\n",
        "| Метрика | До | После | Изменение |",
        "|---------|-----|-------|-----------|",
        f"| Всего строк | {n_before:,} | {n_after:,} | {make_change(n_before, n_after)} |",
        f"| Пропуски / пустые тексты | {miss_before:,} | {missing_after:,} | {make_change(miss_before, missing_after)} |",
        f"| Дубликаты | {dup_before:,} | {dup_after:,} | {make_change(dup_before, dup_after)} |",
        f"| Выбросы (z > 3) | {outliers_before:,} | {outliers_after:,} | {make_change(outliers_before, outliers_after)} |",
        f"| Дисбаланс классов | {imbalance_before or '?'}x | {imbalance_after or '?'}x | — |",
        "\n## Распределение классов после чистки\n",
        "| Класс | Строк | % |",
        "|-------|-------|---|",
    ]

    for label, cnt in label_after.items():
        pct = cnt / n_after * 100
        lines.append(f"| `{label}` | {cnt:,} | {pct:.1f}% |")

    lines += [
        "\n## Обоснование стратегии\n",
        justifications.get(strategy, ""),
        "\n## Следующий шаг\n",
        "Запустить `/auto-tagger` для автоматической разметки очищенного датасета.",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Отчёт сохранён: {output_path}")
    print(f"\nСтроки: {n_before:,} → {n_after:,} ({(n_before-n_after)/n_before*100:.1f}% убрано)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reporter(args.before, args.after, args.problems, args.strategy, args.output)
