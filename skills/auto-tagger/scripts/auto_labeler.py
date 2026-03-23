"""
Батчевая авторазметка через LLM API (OpenAI-compatible).

Использование:
  python auto_labeler.py --input data/cleaned/cleaned.parquet \
                         --output data/labeled/labeled.parquet \
                         --classes "positive,negative,neutral" \
                         --task "Sentiment classification of product reviews" \
                         --confidence-threshold 0.75
"""
import argparse
import json
import os
import time
import pandas as pd
from datetime import datetime

def load_client():
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        raise ImportError("pip install openai python-dotenv")

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_BASE_URL", "https://bothub.chat/api/v2/openai/v1")
    model = os.getenv("API_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError("API_KEY не найден в .env")

    return OpenAI(api_key=api_key, base_url=base_url), model


def label_batch(client, model: str, texts: list[str], classes: list[str], task: str) -> list[dict]:
    """Разметить батч текстов. Вернуть список {label, confidence}."""
    classes_str = ", ".join(classes)
    numbered = "\n".join(f"{i+1}. {t[:300]}" for i, t in enumerate(texts))

    prompt = f"""Task: {task}
Classes: {classes_str}

Label each text. Respond ONLY with a JSON array, one object per text:
[{{"label": "<class>", "confidence": <0.0-1.0>}}, ...]

Texts:
{numbered}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=len(texts) * 40 + 100,
        )
        raw = resp.choices[0].message.content.strip()
        # Вытащить JSON даже если есть лишний текст
        start = raw.find("[")
        end = raw.rfind("]") + 1
        results = json.loads(raw[start:end])
        # Валидация
        for r in results:
            if r.get("label") not in classes:
                r["label"] = classes[0]
            r["confidence"] = max(0.0, min(1.0, float(r.get("confidence", 0.5))))
        return results
    except Exception as e:
        print(f"  WARNING: ошибка батча ({e}), ставлю defaults")
        return [{"label": classes[0], "confidence": 0.0} for _ in texts]


def auto_label(input_path: str, output_path: str, classes: list[str],
               task: str, threshold: float = 0.75, batch_size: int = 10):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_parquet(input_path)

    # Только неразмеченные
    if "label" in df.columns:
        to_label = df[df["label"] == "unlabeled"].copy()
        already = df[df["label"] != "unlabeled"].copy()
    else:
        to_label = df.copy()
        already = pd.DataFrame()

    print(f"Нужно разметить: {len(to_label)} строк")
    print(f"Уже размечено:   {len(already)} строк")
    print(f"Классы: {classes}")
    print(f"Модель: {os.getenv('API_MODEL', 'gpt-4o-mini')}")
    print(f"Порог уверенности: {threshold}\n")

    client, model = load_client()

    labels, confidences = [], []
    total = len(to_label)
    texts = to_label["text"].tolist()

    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        results = label_batch(client, model, batch, classes, task)

        for r in results:
            labels.append(r["label"])
            confidences.append(r["confidence"])

        done = min(i + batch_size, total)
        pct = done / total * 100
        avg_conf = sum(confidences) / len(confidences) * 100
        print(f"  [{done:>5}/{total}] {pct:.0f}%  avg confidence: {avg_conf:.1f}%")

        # Сохранять промежуточный результат каждые 100 строк
        if done % 100 == 0 or done == total:
            to_label_copy = to_label.iloc[:done].copy()
            to_label_copy["label"] = labels
            to_label_copy["confidence"] = confidences
            combined = pd.concat([already, to_label_copy], ignore_index=True)
            combined.to_parquet(output_path, index=False)

        time.sleep(0.3)  # rate limiting

    to_label["label"] = labels
    to_label["confidence"] = confidences

    # Финальный датасет
    result = pd.concat([already, to_label], ignore_index=True)
    result.to_parquet(output_path, index=False)

    # review_queue.csv — примеры с низкой уверенностью
    review = to_label[to_label["confidence"] < threshold][["text", "label", "source", "confidence"]].copy()
    review["corrected_label"] = review["label"]  # колонка для правки
    review_path = "review_queue.csv"
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    # Итог
    print(f"\n{'='*50}")
    print(f"  АВТОРАЗМЕТКА ЗАВЕРШЕНА")
    print(f"{'='*50}")
    print(f"  Размечено: {len(to_label):,} примеров")
    print(f"  Средняя уверенность: {sum(confidences)/len(confidences)*100:.1f}%")
    print(f"  На проверку (< {threshold}): {len(review):,} примеров → {review_path}")
    print(f"\n  Распределение меток:")
    for lbl, cnt in result["label"].value_counts().items():
        print(f"    {lbl}: {cnt:,} ({cnt/len(result)*100:.1f}%)")
    print(f"\n  Сохранено: {output_path}")
    print(f"  Очередь проверки: {review_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", required=True, help="Классы через запятую")
    parser.add_argument("--task", required=True, help="Описание задачи классификации")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    auto_label(args.input, args.output, classes, args.task,
               args.confidence_threshold, args.batch_size)
