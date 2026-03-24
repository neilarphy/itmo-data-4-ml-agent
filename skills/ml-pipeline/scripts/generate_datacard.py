"""
Автоматическая генерация Data Card и финального отчёта пайплайна.
Читает артефакты из data/ и models/, генерирует DATA_CARD.md.

Использование:
  python generate_datacard.py --task "Sentiment classification" \\
                               --classes "positive,negative,neutral" \\
                               --output DATA_CARD.md
"""
import argparse
import json
import os
from datetime import datetime


def _read_json(path: str) -> dict | list | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parquet_stats(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        stats = {"rows": len(df), "columns": list(df.columns)}
        if "label" in df.columns:
            stats["label_dist"] = df["label"].value_counts().to_dict()
        if "modality" in df.columns:
            stats["modality_dist"] = df["modality"].value_counts().to_dict()
        if "source" in df.columns:
            stats["sources"] = df["source"].nunique()
            stats["source_list"] = df["source"].value_counts().head(10).to_dict()
        if "confidence" in df.columns:
            stats["avg_confidence"] = round(df["confidence"].mean(), 3)
            stats["low_conf_pct"] = round(
                (df["confidence"] < 0.75).sum() / len(df) * 100, 1)
        return stats
    except Exception as e:
        return {"error": str(e)}


def _model_info(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import pickle
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return {
            "model_type": bundle.get("model_type", "logreg"),
            "modality": bundle.get("modality", "text"),
            "feature_mode": bundle.get("feature_mode", "tfidf"),
            "classes": bundle.get("classes", []),
            "accuracy": bundle.get("metrics", {}).get("accuracy"),
            "f1": bundle.get("metrics", {}).get("f1"),
            "n_labeled": bundle.get("n_labeled"),
            "strategy": bundle.get("strategy"),
        }
    except Exception:
        return {}


def _al_savings(histories_dir: str = "data/active") -> dict:
    """Посчитать экономию AL vs random из history JSON файлов."""
    results = {}
    for strategy in ["entropy", "margin", "random"]:
        path = os.path.join(histories_dir, f"history_{strategy}.json")
        data = _read_json(path)
        if not data:
            continue
        final = data[-1]
        results[strategy] = {
            "final_accuracy": final.get("accuracy"),
            "final_f1": final.get("f1"),
            "n_labeled": final.get("n_labeled"),
            "iterations": [{"iter": r["iteration"],
                            "n": r["n_labeled"],
                            "acc": r["accuracy"]} for r in data],
        }

    if "entropy" in results and "random" in results:
        # Найти итерацию где entropy достигает финальной accuracy random
        random_final_acc = results["random"]["final_accuracy"]
        for rec in results["entropy"]["iterations"]:
            if rec["acc"] >= random_final_acc:
                savings = results["random"]["n_labeled"] - rec["n"]
                savings_pct = round(savings / results["random"]["n_labeled"] * 100, 1)
                results["_savings"] = {
                    "examples_saved": max(0, savings),
                    "pct_saved": max(0, savings_pct),
                    "entropy_n_to_match": rec["n"],
                    "random_n_final": results["random"]["n_labeled"],
                }
                break

    return results


def generate(task: str, classes: list, output_path: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    raw = _parquet_stats("data/raw/combined.parquet")
    cleaned = _parquet_stats("data/cleaned/cleaned.parquet")
    labeled = _parquet_stats("data/labeled/labeled_final.parquet")
    model = _model_info("models/final_model.pkl")
    problems = _read_json("data/detective/problems.json") or {}
    quality = _read_json("data/labeled/quality.json") or {}
    al = _al_savings()

    review_count = 0
    if os.path.exists("review_queue.csv"):
        try:
            import pandas as pd
            rq = pd.read_csv("review_queue.csv")
            review_count = len(rq)
        except Exception:
            pass

    lines = [
        f"# Data Card",
        f"",
        f"> Автоматически сгенерировано ML Pipeline · {now}",
        f"",
        f"---",
        f"",
        f"## 1. Описание задачи и датасета",
        f"",
        f"| Параметр | Значение |",
        f"|----------|---------|",
        f"| Задача | {task} |",
        f"| Классы | {', '.join(classes)} |",
        f"| Модальность | {model.get('modality', '—')} |",
        f"| Финальный объём | {labeled.get('rows', '—')} строк |",
        f"| Источников | {labeled.get('sources', '—')} |",
        f"",
    ]

    # Источники
    if labeled.get("source_list"):
        lines += [f"### Источники данных", f"", f"| Источник | Строк |", f"|---------|------|"]
        for src, cnt in labeled["source_list"].items():
            lines.append(f"| {src} | {cnt} |")
        lines.append("")

    # Распределение меток
    if labeled.get("label_dist"):
        total = sum(labeled["label_dist"].values())
        lines += [f"### Распределение меток", f"", f"| Класс | Строк | % |", f"|-------|------|---|"]
        for lbl, cnt in labeled["label_dist"].items():
            lines.append(f"| {lbl} | {cnt} | {cnt/total*100:.1f}% |")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## 2. Что делал каждый агент",
        f"",
        f"### data-collector",
        f"- Собрал {raw.get('rows', '?')} строк из {raw.get('sources', '?')} источников",
        f"- Унифицировал схему: text / label / modality / source / collected_at",
        f"",
        f"### quality-guard",
    ]

    if problems:
        for p_type, p_info in problems.items() if isinstance(problems, dict) else []:
            lines.append(f"- {p_type}: {p_info}")
    if raw.get("rows") and cleaned.get("rows"):
        removed = raw["rows"] - cleaned["rows"]
        pct = round(removed / raw["rows"] * 100, 1)
        lines.append(f"- Удалено {removed} строк ({pct}%) → осталось {cleaned['rows']}")

    lines += [
        f"",
        f"### auto-tagger",
        f"- Разметил {labeled.get('rows', '?')} примеров через LLM",
        f"- Средняя уверенность: {labeled.get('avg_confidence', '—')}",
        f"- На ручную проверку отправлено: {review_count} примеров",
        f"",
        f"### smart-sampler (Active Learning)",
        f"- Стратегия: {model.get('strategy', '—')}",
        f"- Модель: {model.get('model_type', '—')} | Фичи: {model.get('feature_mode', '—')}",
        f"- Финальный обучающий набор: {model.get('n_labeled', '—')} примеров",
        f"",
        f"---",
        f"",
        f"## 3. Human-in-the-Loop",
        f"",
        f"| Checkpoint | Что проверялось | Действие |",
        f"|------------|----------------|---------|",
        f"| #1 (сбор) | Выбор источников | Подтверждены конкретные датасеты |",
        f"| #2 (чистка) | Стратегия cleaning | Выбрана стратегия |",
        f"| #3a (разметка) | Задача и классы | Подтверждены |",
        f"| #3b (review) | {review_count} примеров с низкой уверенностью | Исправлены метки вручную |",
        f"| #4 (AL) | Настройки AL | Подтверждены seed/batch/iterations |",
        f"",
    ]

    lines += [
        f"---",
        f"",
        f"## 4. Метрики",
        f"",
        f"### Качество данных",
        f"| Этап | Строк |",
        f"|------|------|",
        f"| После сбора (raw) | {raw.get('rows', '—')} |",
        f"| После чистки | {cleaned.get('rows', '—')} |",
        f"| Финальный (labeled) | {labeled.get('rows', '—')} |",
        f"",
        f"### Метрики модели",
        f"| Метрика | Значение |",
        f"|---------|---------|",
        f"| Accuracy | {model.get('accuracy', '—')} |",
        f"| F1 (weighted) | {model.get('f1', '—')} |",
        f"| Классификатор | {model.get('model_type', '—')} |",
        f"| Обучено на | {model.get('n_labeled', '—')} примерах |",
        f"",
    ]

    # AL savings
    if al.get("_savings"):
        sv = al["_savings"]
        lines += [
            f"### Active Learning — экономия",
            f"| Стратегия | Accuracy | F1 | Примеров |",
            f"|-----------|----------|----|---------| ",
        ]
        for s in ["entropy", "margin", "random"]:
            if s in al:
                lines.append(f"| {s} | {al[s]['final_accuracy']} | "
                              f"{al[s]['final_f1']} | {al[s]['n_labeled']} |")
        lines += [
            f"",
            f"**Экономия entropy vs random:** "
            f"{sv['examples_saved']} примеров ({sv['pct_saved']}%) "
            f"при том же качестве",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## 5. Ретроспектива",
        f"",
        f"_Заполнить вручную после защиты:_",
        f"",
        f"- **Что сработало:** ...",
        f"- **Что не сработало:** ...",
        f"- **Что бы сделал иначе:** ...",
        f"",
        f"---",
        f"",
        f"## Артефакты",
        f"",
        f"| Файл | Описание |",
        f"|------|---------|",
        f"| `data/raw/combined.parquet` | Сырые данные |",
        f"| `data/cleaned/cleaned.parquet` | Очищенные данные |",
        f"| `data/labeled/labeled_final.parquet` | Финальный датасет |",
        f"| `data/labeled/spec.md` | Спецификация разметки |",
        f"| `data/labeled/labelstudio_import.json` | Экспорт в LabelStudio |",
        f"| `data/active/learning_curve.png` | Кривые обучения AL |",
        f"| `data/active/REPORT.md` | Отчёт Active Learning |",
        f"| `models/final_model.pkl` | Финальная модель (pickle) |",
        f"| `review_queue.csv` | HITL — проверенные примеры |",
        f"",
    ]

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Data Card сгенерирован: {output_path}")
    print(f"  Строк: {labeled.get('rows', '?')} | "
          f"Accuracy: {model.get('accuracy', '?')} | "
          f"F1: {model.get('f1', '?')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Описание задачи ML")
    parser.add_argument("--classes", required=True, help="Классы через запятую")
    parser.add_argument("--output", default="DATA_CARD.md")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    generate(args.task, classes, args.output)
