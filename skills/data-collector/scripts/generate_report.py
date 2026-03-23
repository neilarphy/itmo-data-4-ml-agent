"""
Генерация REPORT.md из stats.json после EDA.
Использование: python generate_report.py --stats data/eda/stats.json --output data/eda/REPORT.md
"""
import argparse
import json
from datetime import datetime


def generate_report(stats_path: str, output_path: str):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    lines = [
        "# EDA Report — Data Collector",
        f"\nСгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n---\n",
        "## Общая статистика\n",
        f"| Метрика | Значение |",
        f"|---------|---------|",
        f"| Всего строк | {stats.get('total_rows', '?'):,} |",
        f"| Средняя длина текста | {stats.get('text_len_mean', 0):.0f} символов |",
        f"| Медианная длина | {stats.get('text_len_median', 0):.0f} символов |",
        f"| 95-й перцентиль длины | {stats.get('text_len_p95', 0):.0f} символов |",
        "\n## Источники данных\n",
        "| Источник | Строк |",
        "|----------|-------|",
    ]

    for source, count in stats.get("sources", {}).items():
        lines.append(f"| `{source}` | {count:,} |")

    lines += [
        "\n## Распределение классов\n",
        "| Класс | Количество | Доля |",
        "|-------|-----------|------|",
    ]

    total = stats.get("total_rows", 1)
    for label, count in stats.get("labels", {}).items():
        pct = count / total * 100
        lines.append(f"| `{label}` | {count:,} | {pct:.1f}% |")

    # Дисбаланс
    label_counts = list(stats.get("labels", {}).values())
    if len(label_counts) >= 2:
        imbalance = max(label_counts) / min(label_counts) if min(label_counts) > 0 else float("inf")
        lines += [
            f"\n**Коэффициент дисбаланса:** {imbalance:.2f}x",
            f"\n{'⚠️ Высокий дисбаланс — рассмотри oversampling/undersampling' if imbalance > 3 else '✅ Дисбаланс в норме'}",
        ]

    lines += [
        "\n## Файлы\n",
        "- `data/raw/combined.parquet` — объединённый датасет",
        "- `data/eda/eda_overview.png` — визуализации",
        "- `data/eda/stats.json` — статистика (этот файл)",
        "\n## Следующий шаг\n",
        "Запустить `/quality-guard` для анализа и очистки данных.",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Отчёт сохранён: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_report(args.stats, args.output)
