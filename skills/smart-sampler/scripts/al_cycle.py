"""
Active Learning цикл. Поддерживает text, image, audio.
Стратегии: entropy, margin, random.

Использование:
  python al_cycle.py --seed data/active/seed.parquet \
                     --pool data/active/pool.parquet \
                     --test data/active/test.parquet \
                     --output data/active/history_entropy.json \
                     --strategy entropy \
                     --n-iterations 5 \
                     --batch-size 20 \
                     [--modality text|image|audio]  # автоопределение если не указан
"""
import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Извлечение признаков по модальности ─────────────────────────────

def get_modality(df: pd.DataFrame, modality_hint: str = None) -> str:
    if modality_hint and modality_hint in ("text", "image", "audio"):
        return modality_hint
    if "modality" in df.columns:
        return df["modality"].mode()[0]
    return "text"


def extract_features_text(train_texts, test_texts=None):
    """TF-IDF для текста."""
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    if test_texts is not None:
        X_test = vec.transform(test_texts)
        return X_train, X_test, vec
    return X_train, vec


def extract_features_image(train_paths, test_paths=None):
    """Признаки изображений: цветовые гистограммы (не требует GPU)."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow")

    def img_features(path: str) -> np.ndarray:
        try:
            img = Image.open(path).convert("RGB").resize((64, 64))
            arr = np.array(img).astype(float) / 255.0
            # Гистограммы по каналам (R, G, B) — 48 признаков
            hists = [np.histogram(arr[:, :, c].ravel(), bins=16, range=(0, 1))[0] for c in range(3)]
            feat = np.concatenate(hists).astype(float)
            return feat / (feat.sum() + 1e-10)
        except Exception:
            return np.zeros(48)

    X_train = np.array([img_features(p) for p in train_paths])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if test_paths is not None:
        X_test = scaler.transform(np.array([img_features(p) for p in test_paths]))
        return X_train, X_test, scaler
    return X_train, scaler


def extract_features_audio(train_paths, test_paths=None):
    """MFCC признаки для аудио."""
    try:
        import librosa
    except ImportError:
        raise ImportError("pip install librosa")

    def audio_features(path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(path, sr=22050, duration=10.0)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            # Статистики по каждому MFCC: mean + std = 40 признаков
            return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        except Exception:
            return np.zeros(40)

    X_train = np.array([audio_features(p) for p in train_paths])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if test_paths is not None:
        X_test = scaler.transform(np.array([audio_features(p) for p in test_paths]))
        return X_train, X_test, scaler
    return X_train, scaler


def get_features(train_df: pd.DataFrame, test_df: pd.DataFrame, modality: str):
    """Общий диспетчер признаков."""
    col = "text"  # Колонка в унифицированной схеме (путь к файлу для image/audio)

    if modality == "text":
        return extract_features_text(train_df[col].tolist(), test_df[col].tolist())
    elif modality == "image":
        return extract_features_image(train_df[col].tolist(), test_df[col].tolist())
    elif modality == "audio":
        return extract_features_audio(train_df[col].tolist(), test_df[col].tolist())
    else:
        print(f"WARNING: неизвестная модальность '{modality}', используем text/TF-IDF")
        return extract_features_text(train_df[col].astype(str).tolist(), test_df[col].astype(str).tolist())


# ── Стратегии выбора ─────────────────────────────────────────────────

def entropy_query(clf, X_pool, n: int) -> list[int]:
    proba = clf.predict_proba(X_pool)
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    return list(np.argsort(entropy)[-n:][::-1])


def margin_query(clf, X_pool, n: int) -> list[int]:
    proba = clf.predict_proba(X_pool)
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return list(np.argsort(margin)[:n])


def random_query(n_pool: int, n: int) -> list[int]:
    indices = np.arange(n_pool)
    np.random.shuffle(indices)
    return list(indices[:n])


# ── Основной цикл ────────────────────────────────────────────────────

def run_cycle(seed_path: str, pool_path: str, test_path: str,
              output_path: str, strategy: str,
              n_iterations: int, batch_size: int, modality_hint: str = None,
              save_model: str = None):

    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    seed = pd.read_parquet(seed_path)
    pool = pd.read_parquet(pool_path).copy().reset_index(drop=True)
    test = pd.read_parquet(test_path)

    modality = get_modality(seed, modality_hint)
    print(f"\n{'='*55}")
    print(f"  ACTIVE LEARNING — {strategy.upper()} | модальность: {modality}")
    print(f"  Seed: {len(seed)} | Pool: {len(pool)} | Test: {len(test)}")
    print(f"  Итераций: {n_iterations} × batch {batch_size}")
    print(f"{'='*55}\n")

    labeled = seed.copy()
    history = []

    final_vectorizer = None
    for iteration in range(n_iterations + 1):
        # Признаки (переобучать на каждой итерации)
        X_train, X_test, final_vectorizer = get_features(labeled, test, modality)
        y_train = labeled["label"].tolist()
        y_test = test["label"].tolist()

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        record = {
            "iteration": iteration,
            "n_labeled": len(labeled),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "strategy": strategy,
            "modality": modality,
        }
        history.append(record)
        print(f"  Iter {iteration:>2}: labeled={len(labeled):>4} | acc={acc:.3f} | f1={f1:.3f}")

        if iteration == n_iterations or len(pool) == 0:
            break

        # Признаки пула
        if modality == "text":
            _, X_pool, _ = get_features(labeled, pool, modality)
        elif modality == "image":
            X_pool, _ = extract_features_image(pool["text"].tolist())
        elif modality == "audio":
            X_pool, _ = extract_features_audio(pool["text"].tolist())
        else:
            X_pool = X_train[:len(pool)]  # fallback

        n_query = min(batch_size, len(pool))

        if strategy == "entropy":
            indices = entropy_query(clf, X_pool, n_query)
        elif strategy == "margin":
            indices = margin_query(clf, X_pool, n_query)
        elif strategy == "random":
            indices = random_query(len(pool), n_query)
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")

        selected = pool.iloc[indices]
        labeled = pd.concat([labeled, selected], ignore_index=True)
        pool = pool.drop(pool.index[indices]).reset_index(drop=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    final = history[-1]
    print(f"\n  Финал: accuracy={final['accuracy']:.3f} | f1={final['f1']:.3f}")
    print(f"  Использовано примеров: {final['n_labeled']}")
    print(f"  Сохранено: {output_path}")

    if save_model:
        os.makedirs(os.path.dirname(save_model) if os.path.dirname(save_model) else ".", exist_ok=True)
        model_bundle = {
            "vectorizer": final_vectorizer,
            "classifier": clf,
            "modality": modality,
            "classes": clf.classes_.tolist(),
            "metrics": {"accuracy": final["accuracy"], "f1": final["f1"]},
            "n_labeled": final["n_labeled"],
            "strategy": strategy,
        }
        with open(save_model, "wb") as f:
            pickle.dump(model_bundle, f)
        print(f"  Модель сохранена: {save_model}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", required=True, choices=["entropy", "margin", "random"])
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--modality", default=None, choices=["text", "image", "audio"])
    parser.add_argument("--save-model", default=None, help="Путь для сохранения модели (.pkl)")
    args = parser.parse_args()

    run_cycle(args.seed, args.pool, args.test, args.output,
              args.strategy, args.n_iterations, args.batch_size, args.modality,
              args.save_model)
