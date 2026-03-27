"""
Экспорт датасета в формат LabelStudio JSON для импорта задач.

Использование:
  python export_labelstudio.py --input data/labeled/labeled_final.parquet \
                               --output data/labeled/labelstudio_import.json \
                               --task "Sentiment classification" \
                               --classes "positive,negative,neutral"
"""
import argparse
import json
import os
import uuid
from datetime import datetime
import pandas as pd


def export_labelstudio(input_path: str, output_path: str, task: str, classes: list[str]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_parquet(input_path)

    tasks = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        label = str(row.get("label", "unlabeled"))
        confidence = float(row.get("confidence", 1.0))
        source = str(row.get("source", ""))
        modality = str(row.get("modality", "text"))

        # Формируем data и to_name в зависимости от модальности
        if modality == "image":
            task_data = {"image": text, "source": source}
            to_name = "image"
        elif modality == "audio":
            task_data = {"audio": text, "source": source}
            to_name = "audio"
        else:  # text и tabular — оба отображаются как текст
            task_data = {"text": text, "source": source}
            to_name = "text"

        task_obj = {
            "id": str(uuid.uuid4()),
            "data": task_data,
        }

        # Добавить pre-annotation если есть метка (не unlabeled)
        if label != "unlabeled" and label in classes:
            task_obj["predictions"] = [
                {
                    "model_version": "auto-tagger-v1",
                    "score": confidence,
                    "result": [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "choices",
                            "value": {"choices": [label]},
                            "from_name": "label",
                            "to_name": to_name,
                        }
                    ],
                }
            ]

        tasks.append(task_obj)

    # LabelStudio project config
    output = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "task": task,
        "classes": classes,
        "total_tasks": len(tasks),
        "tasks": tasks,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    labeled_count = sum(1 for t in tasks if "predictions" in t)
    print(f"\nЭкспорт в LabelStudio:")
    print(f"  Всего задач:     {len(tasks):,}")
    print(f"  С pre-аннотацией: {labeled_count:,}")
    print(f"  Без аннотации:   {len(tasks) - labeled_count:,}  ← для разметки вручную")
    print(f"  Сохранено: {output_path}")
    print(f"\nИмпорт в LabelStudio:")
    print(f"  Project → Import → выбери {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--classes", required=True)
    args = parser.parse_args()
    classes = [c.strip() for c in args.classes.split(",")]
    export_labelstudio(args.input, args.output, args.task, classes)
