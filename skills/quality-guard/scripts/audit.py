"""
Аудит качества данных: пропуски, дубликаты, выбросы, дисбаланс классов.
Создаёт problems.json + 4 визуализации.

Использование:
  python audit.py --input data/raw/combined.parquet --output data/detective
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def classify_severity(pct: float) -> str:
    if pct >= 10:   return "Высокая"
    if pct >= 2:    return "Средняя"
    if pct > 0:     return "Низкая"
    return "Нет"


def audit(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(input_path)
    total = len(df)
    problems = {"total_rows": total, "issues": {}}

    # ── 1. Пропущенные значения ──────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)
    problems["issues"]["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    if problems["issues"]["missing"]:
        cols = list(problems["issues"]["missing"].keys())
        pcts = [problems["issues"]["missing"][c]["pct"] for c in cols]
        ax.barh(cols, pcts, color="#e74c3c")
        ax.set_xlabel("% пропущенных значений")
        ax.set_title("Пропущенные значения по колонкам")
        for i, v in enumerate(pcts):
            ax.text(v + 0.1, i, f"{v:.1f}%", va="center")
    else:
        ax.text(0.5, 0.5, "Пропусков нет ✓", ha="center", va="center", fontsize=14, color="green")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/missing_values.png", dpi=130)
    plt.close()

    # ── 2. Дубликаты ─────────────────────────────────────────────────
    dup_count = int(df.duplicated(subset=["text"]).sum())
    dup_pct = round(dup_count / total * 100, 2)
    problems["issues"]["duplicates"] = {"count": dup_count, "pct": dup_pct}

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [total - dup_count, dup_count],
        labels=["Уникальные", "Дубликаты"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(f"Дубликаты: {dup_count:,} ({dup_pct:.1f}%)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/duplicates.png", dpi=130)
    plt.close()

    # ── 3. Выбросы по длине текста (IQR + z-score) ───────────────────
    outliers_info = {}
    if "text" in df.columns:
        lengths = df["text"].dropna().str.len()
        Q1, Q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = int(((lengths < Q1 - 1.5 * IQR) | (lengths > Q3 + 1.5 * IQR)).sum())

        z_scores = (lengths - lengths.mean()) / lengths.std()
        z_outliers = int((z_scores.abs() > 3).sum())

        outliers_info = {
            "iqr_count": iqr_outliers,
            "iqr_pct": round(iqr_outliers / total * 100, 2),
            "z3_count": z_outliers,
            "z3_pct": round(z_outliers / total * 100, 2),
            "q1": float(Q1), "q3": float(Q3), "iqr": float(IQR),
            "mean": float(lengths.mean()), "std": float(lengths.std()),
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(lengths, bins=60, color="#3498db", alpha=0.7, edgecolor="white")
        axes[0].axvline(Q1 - 1.5 * IQR, color="red", linestyle="--", label=f"IQR границы")
        axes[0].axvline(Q3 + 1.5 * IQR, color="red", linestyle="--")
        axes[0].set_title(f"Длина текста (IQR выбросов: {iqr_outliers:,})")
        axes[0].set_xlabel("Символов")
        axes[0].legend()

        axes[1].scatter(range(len(z_scores)), z_scores.values, alpha=0.3, s=5, color="#3498db")
        axes[1].axhline(3, color="red", linestyle="--", label="z=3")
        axes[1].axhline(-3, color="red", linestyle="--")
        axes[1].set_title(f"Z-score (выбросов: {z_outliers:,})")
        axes[1].set_xlabel("Индекс")
        axes[1].set_ylabel("Z-score")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/outliers.png", dpi=130)
        plt.close()
    else:
        # Для нетекстовых данных просто заглушка
        outliers_info = {"note": "Не применимо для данной модальности"}
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Анализ выбросов\nдля текстов N/A", ha="center", va="center")
        ax.axis("off")
        plt.savefig(f"{output_dir}/outliers.png", dpi=130)
        plt.close()

    problems["issues"]["outliers"] = outliers_info

    # ── 4. Дисбаланс классов ─────────────────────────────────────────
    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        labeled = df[df["label"] != "unlabeled"]["label"].value_counts() if "unlabeled" in df["label"].values else label_counts

        imbalance_ratio = float(labeled.max() / labeled.min()) if len(labeled) >= 2 and labeled.min() > 0 else None
        problems["issues"]["class_imbalance"] = {
            "distribution": label_counts.to_dict(),
            "imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio else None,
            "labeled_count": int((df["label"] != "unlabeled").sum()),
            "unlabeled_count": int((df["label"] == "unlabeled").sum()),
        }

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(label_counts)))
        label_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_title(f"Распределение классов (дисбаланс: {imbalance_ratio:.1f}x)" if imbalance_ratio else "Распределение классов")
        ax.set_xlabel("Класс")
        ax.set_ylabel("Количество")
        ax.tick_params(axis="x", rotation=30)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_balance.png", dpi=130)
        plt.close()

    # ── Сохранить итог ───────────────────────────────────────────────
    with open(f"{output_dir}/problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)

    # ── Вывод в консоль ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  АУДИТ КАЧЕСТВА ДАННЫХ ({total:,} строк)")
    print(f"{'='*55}")

    miss_total = sum(v["count"] for v in problems["issues"].get("missing", {}).values())
    print(f"\n  Пропуски:         {miss_total:>6,}  ({miss_total/total*100:.1f}%)  → {classify_severity(miss_total/total*100)}")
    print(f"  Дубликаты:        {dup_count:>6,}  ({dup_pct:.1f}%)  → {classify_severity(dup_pct)}")
    if "iqr_count" in outliers_info:
        print(f"  Выбросы (IQR):    {outliers_info['iqr_count']:>6,}  ({outliers_info['iqr_pct']:.1f}%)  → {classify_severity(outliers_info['iqr_pct'])}")
        print(f"  Выбросы (z>3):    {outliers_info['z3_count']:>6,}  ({outliers_info['z3_pct']:.1f}%)")
    if "class_imbalance" in problems["issues"]:
        r = problems["issues"]["class_imbalance"]["imbalance_ratio"]
        print(f"  Дисбаланс классов: {r:.1f}x  → {'Высокая' if r and r > 3 else 'Средняя' if r and r > 1.5 else 'Нет'}")

    print(f"\n  Визуализации: {output_dir}/")
    print(f"  JSON-отчёт:   {output_dir}/problems.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.input, args.output)
