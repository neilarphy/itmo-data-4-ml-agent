"""
Батчевая авторазметка через LLM API (OpenAI-compatible).
Поддерживает: text (батч), image (GPT-4o-mini vision, по одному),
              audio (faster-whisper → транскрипция → текст → LLM).

Использование:
  python auto_labeler.py --input data/cleaned/cleaned.parquet \\
                         --output data/labeled/labeled.parquet \\
                         --classes "positive,negative,neutral" \\
                         --task "Sentiment classification of product reviews" \\
                         --confidence-threshold 0.75
"""
import argparse
import base64
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


# ── TEXT ──────────────────────────────────────────────────────────────

def label_text_batch(client, model: str, texts: list[str],
                     classes: list[str], task: str) -> list[dict]:
    """Разметить батч текстов. Возвращает [{label, confidence}, ...]."""
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
        start, end = raw.find("["), raw.rfind("]") + 1
        results = json.loads(raw[start:end])
        for r in results:
            if r.get("label") not in classes:
                r["label"] = classes[0]
            r["confidence"] = max(0.0, min(1.0, float(r.get("confidence", 0.5))))
        return results
    except Exception as e:
        print(f"  WARNING: ошибка батча ({e}), ставлю defaults")
        return [{"label": classes[0], "confidence": 0.0} for _ in texts]


# ── IMAGE ─────────────────────────────────────────────────────────────

def _encode_image(path: str) -> tuple[str, str]:
    """Base64 + MIME type."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def label_image_single(client, model: str, path: str,
                        classes: list[str], task: str) -> dict:
    """Разметить одно изображение через vision API."""
    try:
        data, mime = _encode_image(path)
    except Exception as e:
        print(f"  WARNING: не могу открыть {path}: {e}")
        return {"label": classes[0], "confidence": 0.0}

    prompt = (f"Task: {task}\n"
              f"Classify this image into one of these classes: {', '.join(classes)}\n"
              f"Respond ONLY with JSON: {{\"label\": \"<class>\", \"confidence\": <0.0-1.0>}}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{data}",
                                   "detail": "low"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.1,
            max_tokens=60,
        )
        raw = resp.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        result = json.loads(raw[start:end])
        if result.get("label") not in classes:
            result["label"] = classes[0]
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        return result
    except Exception as e:
        print(f"  WARNING: ошибка vision API ({e})")
        return {"label": classes[0], "confidence": 0.0}


# ── AUDIO ─────────────────────────────────────────────────────────────

_whisper_model_cache = None


def _transcribe_audio(path: str) -> str | None:
    """Транскрибировать аудио через faster-whisper (tiny, CPU, бесплатно)."""
    global _whisper_model_cache
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None  # caller handles fallback

    if _whisper_model_cache is None:
        print("  Загрузка Whisper tiny (первый раз, ~75MB)...")
        _whisper_model_cache = WhisperModel("tiny", device="cpu", compute_type="int8")

    try:
        segments, _ = _whisper_model_cache.transcribe(path, beam_size=1)
        return " ".join(s.text for s in segments).strip()
    except Exception as e:
        print(f"  WARNING: транскрипция не удалась ({e})")
        return None


def label_audio_single(client, model: str, path: str,
                        classes: list[str], task: str) -> dict:
    """Транскрибировать аудио → разметить текст через LLM."""
    transcript = _transcribe_audio(path)

    if transcript:
        results = label_text_batch(client, model, [transcript], classes, task)
        return results[0]

    # Fallback: нет faster-whisper — низкий confidence, требует ручной проверки
    print(f"  INFO: faster-whisper не установлен. pip install faster-whisper")
    print(f"        Аудио {os.path.basename(path)} помечено для ручной проверки.")
    return {"label": classes[0], "confidence": 0.0}


# ── Основной pipeline ─────────────────────────────────────────────────

def auto_label(input_path: str, output_path: str, classes: list[str],
               task: str, threshold: float = 0.75, batch_size: int = 10):

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df = pd.read_parquet(input_path)

    if "label" in df.columns:
        to_label = df[df["label"] == "unlabeled"].copy().reset_index(drop=True)
        already = df[df["label"] != "unlabeled"].copy()
    else:
        to_label = df.copy().reset_index(drop=True)
        already = pd.DataFrame()

    # Определить доминирующую модальность
    modality = "text"
    if "modality" in to_label.columns and len(to_label) > 0:
        modality = to_label["modality"].mode()[0]

    print(f"Нужно разметить: {len(to_label)} строк")
    print(f"Уже размечено:   {len(already)} строк")
    print(f"Модальность:     {modality}")
    print(f"Классы: {classes}")
    print(f"Модель: {os.getenv('API_MODEL', 'gpt-4o-mini')}")
    print(f"Порог уверенности: {threshold}\n")

    if modality == "image":
        print("  Режим: IMAGE (GPT-4o-mini vision, по одному примеру)")
        print("  Убедись что API_MODEL поддерживает vision (gpt-4o-mini ✓)\n")
    elif modality == "audio":
        print("  Режим: AUDIO (faster-whisper → транскрипция → LLM)")
        print("  Установка: pip install faster-whisper\n")

    client, model = load_client()

    labels, confidences = [], []
    total = len(to_label)

    if modality == "text":
        # Батчевая разметка текстов
        texts = to_label["text"].tolist()
        for i in range(0, total, batch_size):
            batch = texts[i: i + batch_size]
            results = label_text_batch(client, model, batch, classes, task)
            for r in results:
                labels.append(r["label"])
                confidences.append(r["confidence"])
            _print_progress(labels, confidences, total)
            _maybe_save(to_label, already, labels, confidences, output_path, i, batch_size, total)
            time.sleep(0.3)

    elif modality == "image":
        # По одному — vision API
        paths = to_label["text"].tolist()
        for i, path in enumerate(paths):
            r = label_image_single(client, model, path, classes, task)
            labels.append(r["label"])
            confidences.append(r["confidence"])
            if (i + 1) % 10 == 0 or (i + 1) == total:
                _print_progress(labels, confidences, total)
                _maybe_save(to_label, already, labels, confidences, output_path, i, 1, total)
            time.sleep(0.5)  # vision API медленнее

    elif modality == "audio":
        # Транскрипция → текст
        paths = to_label["text"].tolist()
        for i, path in enumerate(paths):
            r = label_audio_single(client, model, path, classes, task)
            labels.append(r["label"])
            confidences.append(r["confidence"])
            if (i + 1) % 10 == 0 or (i + 1) == total:
                _print_progress(labels, confidences, total)
                _maybe_save(to_label, already, labels, confidences, output_path, i, 1, total)
            time.sleep(0.3)

    else:
        # Unknown modality → treat as text
        texts = to_label["text"].astype(str).tolist()
        for i in range(0, total, batch_size):
            batch = texts[i: i + batch_size]
            results = label_text_batch(client, model, batch, classes, task)
            for r in results:
                labels.append(r["label"])
                confidences.append(r["confidence"])
            _print_progress(labels, confidences, total)
            _maybe_save(to_label, already, labels, confidences, output_path, i, batch_size, total)
            time.sleep(0.3)

    to_label["label"] = labels
    to_label["confidence"] = confidences

    result = pd.concat([already, to_label], ignore_index=True)
    result.to_parquet(output_path, index=False)

    # review_queue.csv
    review_cols = ["text", "label", "source", "confidence"]
    review_cols = [c for c in review_cols if c in to_label.columns]
    review = to_label[to_label["confidence"] < threshold][review_cols].copy()
    review["corrected_label"] = review["label"]
    review.to_csv("review_queue.csv", index=False, encoding="utf-8-sig")

    _print_summary(to_label, result, labels, confidences, threshold)


def _print_progress(labels, confidences, total):
    done = len(labels)
    pct = done / total * 100
    avg_conf = sum(confidences) / len(confidences) * 100
    print(f"  [{done:>5}/{total}] {pct:.0f}%  avg confidence: {avg_conf:.1f}%")


def _maybe_save(to_label, already, labels, confidences, output_path, i, step, total):
    done = min(i + step, total - 1) + 1
    if done % 100 == 0 or done == total:
        partial = to_label.iloc[:done].copy()
        partial["label"] = labels[:done]
        partial["confidence"] = confidences[:done]
        combined = pd.concat([already, partial], ignore_index=True)
        combined.to_parquet(output_path, index=False)


def _print_summary(to_label, result, labels, confidences, threshold):
    review_count = sum(1 for c in confidences if c < threshold)
    print(f"\n{'='*50}")
    print(f"  АВТОРАЗМЕТКА ЗАВЕРШЕНА")
    print(f"{'='*50}")
    print(f"  Размечено: {len(to_label):,} примеров")
    print(f"  Средняя уверенность: {sum(confidences)/len(confidences)*100:.1f}%")
    print(f"  На проверку (< {threshold}): {review_count:,} примеров → review_queue.csv")
    print(f"\n  Распределение меток:")
    for lbl, cnt in result["label"].value_counts().items():
        print(f"    {lbl}: {cnt:,} ({cnt/len(result)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    auto_label(args.input, args.output, classes, args.task,
               args.confidence_threshold, args.batch_size)
