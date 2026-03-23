"""
Метрики качества разметки: распределение меток, средняя уверенность, % на проверку.

Использование:
  python quality_check.py --input data/labeled/labeled_final.parquet \
                          --output data/labeled/quality.json
"""
import argparse
import json
import os
import pandas as pd
import numpy as np


def check_quality(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_parquet(input_path)

    metrics = {
        "total": len(df),
        "labeled": int((df["label"] != "unlabeled").sum()) if "label" in df.columns else len(df),
        "unlabeled": int((df["label"] == "unlabeled").sum()) if "label" in df.columns else 0,
    }

    if "confidence" in df.columns:
        conf = df["confidence"].dropna()
        metrics["confidence"] = {
            "mean": round(float(conf.mean()), 3),
            "median": round(float(conf.median()), 3),
            "std": round(float(conf.std()), 3),
            "pct_high": round(float((conf >= 0.75).mean() * 100), 1),
            "pct_low":  round(float((conf < 0.75).mean() * 100), 1),
        }

    if "label" in df.columns:
        label_counts = df["label"].value_counts().to_dict()
        total_labeled = metrics["labeled"]
        metrics["label_distribution"] = {
            lbl: {"count": int(cnt), "pct": round(cnt / total_labeled * 100, 1)}
            for lbl, cnt in label_counts.items()
            if lbl != "unlabeled"
        }

        counts = [v["count"] for v in metrics["label_distribution"].values()]
        if len(counts) >= 2 and min(counts) > 0:
            metrics["imbalance_ratio"] = round(max(counts) / min(counts), 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*45}")
    print(f"  МЕТРИКИ КАЧЕСТВА РАЗМЕТКИ")
    print(f"{'='*45}")
    print(f"  Всего строк:       {metrics['total']:,}")
    print(f"  Размечено:         {metrics['labeled']:,}")
    if "confidence" in metrics:
        c = metrics["confidence"]
        print(f"  Средняя уверенность: {c['mean']*100:.1f}%")
        print(f"  Высокая (≥75%):      {c['pct_high']:.1f}%")
        print(f"  Низкая (<75%):       {c['pct_low']:.1f}%  ← на ручную проверку")
    if "label_distribution" in metrics:
        print(f"\n  Распределение меток:")
        for lbl, v in metrics["label_distribution"].items():
            bar = "█" * int(v["pct"] / 2)
            print(f"    {lbl:<15} {v['count']:>6,}  {v['pct']:>5.1f}%  {bar}")
    if "imbalance_ratio" in metrics:
        r = metrics["imbalance_ratio"]
        flag = "⚠️" if r > 3 else "✓"
        print(f"\n  Дисбаланс: {r}x {flag}")
    print(f"\n  Сохранено: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_quality(args.input, args.output)
