"""
Интерактивный терминальный ревьювер для неуверенных аннотаций.

Использование:
  python review_terminal.py
  python review_terminal.py --input review_queue.csv --classes "c1,c2,c3"
"""
import argparse
import os
import sys
import textwrap

import pandas as pd


def clear_line():
    print("\033[2K\r", end="")


def print_header(idx: int, total: int, confidence: float, modality: str):
    bar_done = int((idx / total) * 30)
    bar = "█" * bar_done + "░" * (30 - bar_done)
    print(f"\n[{idx}/{total}]  [{bar}]  conf: {confidence:.2f}  modality: {modality}")
    print("─" * 60)


def print_item(text: str, modality: str):
    if modality in ("image", "audio"):
        print(f"  {modality.upper()} PATH: {text}")
    else:
        lines = textwrap.wrap(text[:500], width=56)
        for line in lines[:8]:
            print(f"  {line}")
        if len(text) > 500:
            print(f"  ... [{len(text)} chars total]")
    print("─" * 60)


def print_choices(classes: list[str], current_label: str):
    for i, cls in enumerate(classes, 1):
        marker = " ←" if cls == current_label else ""
        print(f"  {i}. {cls}{marker}")
    print(f"  s. пропустить")
    print(f"  q. выйти и сохранить")


def ask(classes: list[str]) -> str | None:
    valid = {str(i): cls for i, cls in enumerate(classes, 1)}
    valid["s"] = "skip"
    valid["q"] = "quit"
    while True:
        try:
            choice = input("\n→ ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if choice in valid:
            return valid[choice]
        print(f"  Введи цифру 1–{len(classes)}, s или q")


def run(input_path: str, classes: list[str] | None):
    if not os.path.exists(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    if "corrected_label" not in df.columns:
        df["corrected_label"] = df.get("label", "")

    # Определяем классы: из аргумента или из уникальных меток в данных
    if classes is None:
        raw_labels = df["label"].dropna().unique().tolist()
        classes = sorted(set(l for l in raw_labels if l != "unlabeled"))
        if not classes:
            print("Не удалось определить классы. Передай --classes 'c1,c2,...'")
            sys.exit(1)
        print(f"Классы из данных: {classes}")

    # Только строки без исправления (где corrected_label == label или пустая)
    pending_mask = (
        df["corrected_label"].isna()
        | (df["corrected_label"] == df.get("label", ""))
        | (df["corrected_label"] == "")
    )
    pending_idx = df[pending_mask].index.tolist()
    total = len(pending_idx)

    if total == 0:
        print("Нет строк для проверки — все уже размечены.")
        return

    print(f"\nНа проверку: {total} примеров  |  Классы: {classes}")
    print("=" * 60)

    changed = 0
    for n, row_idx in enumerate(pending_idx, 1):
        row = df.loc[row_idx]
        modality = str(row.get("modality", "text"))
        text = str(row.get("text", ""))
        current_label = str(row.get("label", ""))
        confidence = float(row.get("confidence", 0.0))

        print_header(n, total, confidence, modality)
        print_item(text, modality)
        print(f"\n  Авто-метка: {current_label}  (conf: {confidence:.2f})\n")
        print_choices(classes, current_label)

        result = ask(classes)

        if result == "quit":
            print("\nСохраняю и выхожу...")
            break
        elif result == "skip":
            continue
        else:
            if result != current_label:
                changed += 1
            df.at[row_idx, "corrected_label"] = result
            df.to_csv(input_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  Проверено:  {n}/{total}")
    print(f"  Исправлено: {changed}")
    print(f"  Сохранено:  {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="review_queue.csv")
    parser.add_argument("--classes", default=None,
                        help="Классы через запятую. По умолчанию берёт из данных.")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    run(args.input, classes)
