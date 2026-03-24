"""
Active Learning цикл. Поддерживает text, image, audio, multimodal (смешанные строки).
Стратегии: entropy, margin, random.
Модели: logreg (default), svm, rf.
Фичи: tfidf/sentence (text), resnet/histogram (image), mfcc (audio).

Использование:
  python al_cycle.py --seed data/active/seed.parquet \\
                     --pool data/active/pool.parquet \\
                     --test data/active/test.parquet \\
                     --output data/active/history_entropy.json \\
                     --strategy entropy \\
                     [--model logreg|svm|rf] \\
                     [--features tfidf|sentence] \\
                     [--save-model models/final_model.pkl]
"""
import argparse
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ── Извлечение признаков: TEXT ────────────────────────────────────────

def extract_features_text_tfidf(train_texts, test_texts=None):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    if test_texts is not None:
        return X_train, vec.transform(test_texts), vec
    return X_train, vec


def extract_features_text_sentence(train_texts, test_texts=None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  INFO: sentence-transformers не установлен, используем TF-IDF")
        return extract_features_text_tfidf(train_texts, test_texts)

    print("  Загрузка sentence-transformers (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = model.encode(train_texts, show_progress_bar=False, batch_size=64)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if test_texts is not None:
        X_test = scaler.transform(model.encode(test_texts, show_progress_bar=False, batch_size=64))
        return X_train, X_test, ("sentence", model, scaler)
    return X_train, ("sentence", model, scaler)


# ── Извлечение признаков: IMAGE ───────────────────────────────────────

def _try_resnet_extractor():
    """Вернуть (extractor, transform) или None если torch недоступен."""
    try:
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as tv_transforms

        weights_attr = getattr(tv_models, "ResNet18_Weights", None)
        if weights_attr:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
            model = tv_models.resnet18(weights=weights)
        else:
            model = tv_models.resnet18(pretrained=True)

        extractor = torch.nn.Sequential(*list(model.children())[:-1])
        extractor.eval()

        transform = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
        return extractor, transform
    except Exception as e:
        print(f"  INFO: torch/torchvision недоступен ({e}), используем цветовые гистограммы")
        return None


def _img_features_resnet(path, extractor, transform):
    try:
        import torch
        from PIL import Image
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = extractor(tensor).squeeze().numpy()  # 512-dim
        return feat
    except Exception:
        return np.zeros(512)


def _img_features_histogram(path):
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((64, 64))
        arr = np.array(img).astype(float) / 255.0
        hists = [np.histogram(arr[:, :, c].ravel(), bins=16, range=(0, 1))[0] for c in range(3)]
        feat = np.concatenate(hists).astype(float)
        return feat / (feat.sum() + 1e-10)
    except Exception:
        return np.zeros(48)


def extract_features_image(train_paths, test_paths=None):
    resnet = _try_resnet_extractor()
    n_feats = 512 if resnet else 48
    method = "ResNet18 (512-dim)" if resnet else "color histogram (48-dim)"
    print(f"  Image features: {method}")

    def get_feat(path):
        if resnet:
            return _img_features_resnet(path, resnet[0], resnet[1])
        return _img_features_histogram(path)

    X_train = np.array([get_feat(p) for p in train_paths])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if test_paths is not None:
        X_test = scaler.transform(np.array([get_feat(p) for p in test_paths]))
        return X_train, X_test, ("image_scaler", scaler, resnet)
    return X_train, ("image_scaler", scaler, resnet)


# ── Извлечение признаков: AUDIO ───────────────────────────────────────

def extract_features_audio(train_paths, test_paths=None):
    try:
        import librosa
    except ImportError:
        raise ImportError("pip install librosa")

    def audio_features(path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(path, sr=22050, duration=10.0)
            # MFCC mean + std (40)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfcc)
            # Спектральные признаки (6)
            zcr = librosa.feature.zero_crossing_rate(y)
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)
            sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            return np.concatenate([
                mfcc.mean(axis=1), mfcc.std(axis=1),
                mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
                [zcr.mean(), zcr.std()],
                [sc.mean(), sc.std()],
                [sb.mean(), sb.std()],
                [rms.mean(), rms.std()],
            ])  # ~88 признаков
        except Exception:
            return np.zeros(88)

    X_train = np.array([audio_features(p) for p in train_paths])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if test_paths is not None:
        X_test = scaler.transform(np.array([audio_features(p) for p in test_paths]))
        return X_train, X_test, ("audio_scaler", scaler)
    return X_train, ("audio_scaler", scaler)


# ── Мультимодальный диспетчер ─────────────────────────────────────────

def extract_features_tabular(train_texts, test_texts=None):
    """JSON строки → numpy array числовых фичей."""
    import json as _json

    def parse(s):
        try:
            d = _json.loads(s)
            return np.array(list(d.values()), dtype=float)
        except Exception:
            return np.zeros(16)

    X_train = np.array([parse(s) for s in train_texts])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if test_texts is not None:
        X_test = scaler.transform(np.array([parse(s) for s in test_texts]))
        return X_train, X_test, ("tabular_scaler", scaler)
    return X_train, ("tabular_scaler", scaler)


# ── Мультимодальный диспетчер ─────────────────────────────────────────

def detect_modality(df: pd.DataFrame, hint: str = None) -> str:
    if hint and hint in ("text", "image", "audio", "tabular"):
        return hint
    if "modality" in df.columns:
        counts = df["modality"].value_counts()
        dominant = counts.index[0]
        if len(counts) > 1:
            print(f"  INFO: смешанные модальности {counts.to_dict()}, используем доминирующую: {dominant}")
        return dominant
    return "text"


def get_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 modality: str, feature_mode: str = "tfidf"):
    col = "text"
    if modality == "text":
        if feature_mode == "sentence":
            return extract_features_text_sentence(train_df[col].tolist(), test_df[col].tolist())
        return extract_features_text_tfidf(train_df[col].tolist(), test_df[col].tolist())
    elif modality == "image":
        return extract_features_image(train_df[col].tolist(), test_df[col].tolist())
    elif modality == "audio":
        return extract_features_audio(train_df[col].tolist(), test_df[col].tolist())
    elif modality == "tabular":
        return extract_features_tabular(train_df[col].tolist(), test_df[col].tolist())
    else:
        print(f"  WARNING: неизвестная модальность '{modality}', используем TF-IDF")
        return extract_features_text_tfidf(train_df[col].astype(str).tolist(),
                                           test_df[col].astype(str).tolist())


def get_features_pool(pool_df: pd.DataFrame, modality: str,
                      vectorizer, feature_mode: str = "tfidf"):
    """Трансформировать пул используя уже обученный vectorizer."""
    col = "text"
    if modality == "text":
        if feature_mode == "sentence" and isinstance(vectorizer, tuple) and vectorizer[0] == "sentence":
            _, model, scaler = vectorizer
            X = model.encode(pool_df[col].tolist(), show_progress_bar=False, batch_size=64)
            return scaler.transform(X)
        return vectorizer.transform(pool_df[col].tolist())
    elif modality == "image":
        _, scaler, resnet = vectorizer

        def get_feat(path):
            if resnet:
                return _img_features_resnet(path, resnet[0], resnet[1])
            return _img_features_histogram(path)

        X = np.array([get_feat(p) for p in pool_df[col].tolist()])
        return scaler.transform(X)
    elif modality == "tabular":
        import json as _json
        _, scaler = vectorizer

        def parse(s):
            try:
                d = _json.loads(s)
                return np.array(list(d.values()), dtype=float)
            except Exception:
                return np.zeros(16)

        X = np.array([parse(s) for s in pool_df[col].tolist()])
        return scaler.transform(X)
    elif modality == "audio":
        try:
            import librosa
        except ImportError:
            return np.zeros((len(pool_df), 88))
        _, scaler = vectorizer

        def audio_features(path):
            try:
                y, sr = librosa.load(path, sr=22050, duration=10.0)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_delta = librosa.feature.delta(mfcc)
                zcr = librosa.feature.zero_crossing_rate(y)
                sc = librosa.feature.spectral_centroid(y=y, sr=sr)
                sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)
                return np.concatenate([
                    mfcc.mean(axis=1), mfcc.std(axis=1),
                    mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
                    [zcr.mean(), zcr.std()], [sc.mean(), sc.std()],
                    [sb.mean(), sb.std()], [rms.mean(), rms.std()],
                ])
            except Exception:
                return np.zeros(88)

        X = np.array([audio_features(p) for p in pool_df[col].tolist()])
        return scaler.transform(X)
    return vectorizer.transform(pool_df[col].astype(str).tolist())


# ── Модели ────────────────────────────────────────────────────────────

def get_classifier(model_type: str):
    if model_type == "svm":
        return SVC(probability=True, kernel="rbf", C=1.0, random_state=42,
                   max_iter=2000)
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:  # logreg (default)
        return LogisticRegression(max_iter=1000, C=1.0, random_state=42)


# ── Стратегии выбора ──────────────────────────────────────────────────

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
    idx = np.arange(n_pool)
    np.random.shuffle(idx)
    return list(idx[:n])


# ── Основной цикл ─────────────────────────────────────────────────────

def run_cycle(seed_path: str, pool_path: str, test_path: str,
              output_path: str, strategy: str,
              n_iterations: int, batch_size: int,
              modality_hint: str = None, feature_mode: str = "tfidf",
              model_type: str = "logreg", save_model: str = None):

    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    seed = pd.read_parquet(seed_path)
    pool = pd.read_parquet(pool_path).copy().reset_index(drop=True)
    test = pd.read_parquet(test_path)

    modality = detect_modality(seed, modality_hint)
    print(f"\n{'='*58}")
    print(f"  ACTIVE LEARNING — {strategy.upper()}")
    print(f"  Модальность: {modality} | Модель: {model_type} | Фичи: {feature_mode}")
    print(f"  Seed: {len(seed)} | Pool: {len(pool)} | Test: {len(test)}")
    print(f"  Итераций: {n_iterations} × batch {batch_size}")
    print(f"{'='*58}\n")

    labeled = seed.copy()
    history = []
    final_vectorizer = None
    final_clf = None

    for iteration in range(n_iterations + 1):
        X_train, X_test, final_vectorizer = get_features(labeled, test, modality, feature_mode)
        y_train = labeled["label"].tolist()
        y_test = test["label"].tolist()

        clf = get_classifier(model_type)
        clf.fit(X_train, y_train)
        final_clf = clf

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
            "model": model_type,
        }
        history.append(record)
        print(f"  Iter {iteration:>2}: labeled={len(labeled):>4} | acc={acc:.3f} | f1={f1:.3f}")

        if iteration == n_iterations or len(pool) == 0:
            break

        X_pool = get_features_pool(pool, modality, final_vectorizer, feature_mode)
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

    if save_model and final_clf is not None:
        os.makedirs(os.path.dirname(save_model) if os.path.dirname(save_model) else ".", exist_ok=True)
        model_bundle = {
            "vectorizer": final_vectorizer,
            "classifier": final_clf,
            "modality": modality,
            "feature_mode": feature_mode,
            "model_type": model_type,
            "classes": final_clf.classes_.tolist(),
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
    parser.add_argument("--modality", default=None, choices=["text", "image", "audio", "tabular"])
    parser.add_argument("--features", default="tfidf", choices=["tfidf", "sentence"],
                        help="Метод извлечения фичей для текста")
    parser.add_argument("--model", default="logreg", choices=["logreg", "svm", "rf"],
                        help="Классификатор (logreg — лучшая калибровка для AL)")
    parser.add_argument("--save-model", default=None, help="Путь для сохранения модели (.pkl)")
    args = parser.parse_args()

    run_cycle(args.seed, args.pool, args.test, args.output,
              args.strategy, args.n_iterations, args.batch_size,
              args.modality, args.features, args.model, args.save_model)
