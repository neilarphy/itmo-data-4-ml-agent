"""
Inference на финальной модели из AL-цикла.

Использование:
  python predict.py --model models/final_model.pkl --input new_data.csv
  python predict.py --model models/final_model.pkl --input new_data.parquet --output predictions.csv
  python predict.py --model models/final_model.pkl --text "Some text to classify"
"""
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd


def load_model(model_path: str) -> dict:
    if not os.path.exists(model_path):
        print(f"ERROR: модель не найдена: {model_path}")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def extract_features(texts_or_paths: list, bundle: dict) -> np.ndarray:
    modality = bundle["modality"]
    feature_mode = bundle.get("feature_mode", "tfidf")
    vectorizer = bundle["vectorizer"]

    if modality == "text":
        if feature_mode == "sentence" and isinstance(vectorizer, tuple) and vectorizer[0] == "sentence":
            _, model, scaler = vectorizer
            X = model.encode(texts_or_paths, show_progress_bar=False, batch_size=64)
            return scaler.transform(X)
        # TF-IDF
        return vectorizer.transform(texts_or_paths)

    elif modality == "image":
        _, scaler, resnet = vectorizer

        def _get_feat(path):
            try:
                from PIL import Image
                img = Image.open(path).convert("RGB")
                if resnet:
                    import torch
                    import torchvision.transforms as tv_transforms
                    transform = tv_transforms.Compose([
                        tv_transforms.Resize(256),
                        tv_transforms.CenterCrop(224),
                        tv_transforms.ToTensor(),
                        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                    ])
                    extractor, _ = resnet
                    tensor = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        return extractor(tensor).squeeze().numpy()
                else:
                    img = img.resize((64, 64))
                    arr = np.array(img).astype(float) / 255.0
                    hists = [np.histogram(arr[:, :, c].ravel(), bins=16, range=(0, 1))[0]
                             for c in range(3)]
                    feat = np.concatenate(hists).astype(float)
                    return feat / (feat.sum() + 1e-10)
            except Exception:
                return np.zeros(512 if resnet else 48)

        X = np.array([_get_feat(p) for p in texts_or_paths])
        return scaler.transform(X)

    elif modality == "audio":
        _, scaler = vectorizer
        try:
            import librosa
        except ImportError:
            print("ERROR: pip install librosa")
            sys.exit(1)

        def _audio_feat(path):
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

        X = np.array([_audio_feat(p) for p in texts_or_paths])
        return scaler.transform(X)

    # fallback: TF-IDF
    return vectorizer.transform([str(x) for x in texts_or_paths])


def predict(model_path: str, df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model(model_path)
    clf = bundle["classifier"]
    modality = bundle["modality"]
    classes = bundle["classes"]

    print(f"Модель: {bundle.get('model_type', 'logreg')}")
    print(f"Модальность: {modality}")
    print(f"Классы: {classes}")
    print(f"Метрики при обучении: acc={bundle['metrics']['accuracy']:.3f}, "
          f"f1={bundle['metrics']['f1']:.3f}")
    print(f"Примеров: {len(df)}\n")

    texts_or_paths = df["text"].tolist()
    X = extract_features(texts_or_paths, bundle)

    predictions = clf.predict(X)
    proba = clf.predict_proba(X)
    confidence = proba.max(axis=1)

    result = df.copy()
    result["predicted_label"] = predictions
    result["confidence"] = np.round(confidence, 4)

    # Добавить вероятности по классам
    for i, cls in enumerate(classes):
        result[f"prob_{cls}"] = np.round(proba[:, i], 4)

    return result


def main():
    parser = argparse.ArgumentParser(description="Inference на финальной AL-модели")
    parser.add_argument("--model", required=True, help="Путь к .pkl файлу модели")
    parser.add_argument("--input", help="CSV или parquet с колонкой 'text'")
    parser.add_argument("--text", help="Один текст/путь для классификации")
    parser.add_argument("--output", default=None, help="Куда сохранить CSV с предсказаниями")
    args = parser.parse_args()

    if args.text:
        df = pd.DataFrame([{"text": args.text}])
    elif args.input:
        if args.input.endswith(".parquet"):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
    else:
        print("ERROR: укажи --input или --text")
        sys.exit(1)

    result = predict(args.model, df)

    print("Результаты:")
    print(result[["text", "predicted_label", "confidence"]].to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\nСохранено: {args.output}")

    # Сводка
    print(f"\nРаспределение предсказаний:")
    for lbl, cnt in result["predicted_label"].value_counts().items():
        print(f"  {lbl}: {cnt} ({cnt/len(result)*100:.1f}%)")


if __name__ == "__main__":
    main()
