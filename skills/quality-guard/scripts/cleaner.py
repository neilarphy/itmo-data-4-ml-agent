"""
Чистка данных по выбранной стратегии.

Стратегии:
  aggressive   — удалить пропуски, дубли, IQR-выбросы
  conservative — заполнить пропуски, оставить дубли и выбросы
  balanced     — удалить дубли, заполнить пропуски, удалить z>3 выбросы

Использование:
  python cleaner.py --input data/raw/combined.parquet \
                    --output data/cleaned/cleaned.parquet \
                    --strategy balanced
"""
import argparse
import os
import pandas as pd
import numpy as np


def clean(input_path: str, output_path: str, strategy: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_parquet(input_path).copy()
    before = len(df)
    log = []

    if strategy == "aggressive":
        # Пропуски → удалить строки
        n = len(df)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]
        log.append(f"Удалено строк с пропусками/пустым текстом: {n - len(df):,}")

        # Дубликаты → удалить
        n = len(df)
        df = df.drop_duplicates(subset=["text"])
        log.append(f"Удалено дубликатов: {n - len(df):,}")

        # Выбросы по длине текста (IQR)
        if "text" in df.columns:
            lengths = df["text"].str.len()
            Q1, Q3 = lengths.quantile(0.25), lengths.quantile(0.75)
            IQR = Q3 - Q1
            mask = (lengths >= Q1 - 1.5 * IQR) & (lengths <= Q3 + 1.5 * IQR)
            n = len(df)
            df = df[mask]
            log.append(f"Удалено IQR-выбросов: {n - len(df):,}")

    elif strategy == "conservative":
        # Пропуски → заполнить
        n_missing = df["text"].isnull().sum()
        df["text"] = df["text"].fillna("").astype(str)
        log.append(f"Заполнено пропусков в text: {n_missing:,}")

        for col in ["label", "source", "collected_at"]:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        # Дубликаты → оставить
        log.append("Дубликаты: оставлены (conservative)")

        # Выбросы → оставить
        log.append("Выбросы: оставлены (conservative)")

    elif strategy == "balanced":
        # Пропуски → заполнить, пустые удалить
        df["text"] = df["text"].fillna("").astype(str)
        n = len(df)
        df = df[df["text"].str.strip() != ""]
        log.append(f"Удалено пустых текстов: {n - len(df):,}")

        for col in ["label", "source", "collected_at"]:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        # Дубликаты → удалить
        n = len(df)
        df = df.drop_duplicates(subset=["text"])
        log.append(f"Удалено дубликатов: {n - len(df):,}")

        # Выбросы → удалить экстремальные (z-score > 3)
        if "text" in df.columns:
            lengths = df["text"].str.len()
            mean, std = lengths.mean(), lengths.std()
            if std > 0:
                z = (lengths - mean) / std
                n = len(df)
                df = df[z.abs() <= 3]
                log.append(f"Удалено z>3 выбросов: {n - len(df):,}")
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}. Доступны: aggressive, conservative, balanced")

    df = df.reset_index(drop=True)
    df.to_parquet(output_path, index=False)

    after = len(df)
    print(f"\n{'='*50}")
    print(f"  ЧИСТКА — стратегия: {strategy.upper()}")
    print(f"{'='*50}")
    for line in log:
        print(f"  • {line}")
    print(f"\n  Было:   {before:>7,} строк")
    print(f"  Стало:  {after:>7,} строк")
    print(f"  Убрано: {before - after:>7,} ({(before - after) / before * 100:.1f}%)")
    print(f"\n  Сохранено: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", required=True, choices=["aggressive", "conservative", "balanced"])
    args = parser.parse_args()
    clean(args.input, args.output, args.strategy)
