"""
Визуализация learning curves для сравнения AL-стратегий.

Использование:
  python visualize.py \
    --histories data/active/history_entropy.json data/active/history_margin.json data/active/history_random.json \
    --labels entropy margin random \
    --output data/active/learning_curve.png
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


COLORS = {"entropy": "#e74c3c", "margin": "#3498db", "random": "#95a5a6"}
STYLES = {"entropy": "-o", "margin": "-s", "random": "--^"}


def load_history(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def visualize(history_paths: list[str], labels: list[str], output_path: str):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Active Learning — Сравнение стратегий", fontsize=13, fontweight="bold")

    savings = {}

    for path, label in zip(history_paths, labels):
        history = load_history(path)
        n_labeled = [h["n_labeled"] for h in history]
        accuracy = [h["accuracy"] for h in history]
        f1 = [h["f1"] for h in history]

        color = COLORS.get(label, "#2ecc71")
        style = STYLES.get(label, "-o")

        axes[0].plot(n_labeled, accuracy, style, color=color, label=label, linewidth=2, markersize=6)
        axes[1].plot(n_labeled, f1, style, color=color, label=label, linewidth=2, markersize=6)

        savings[label] = {"final_acc": accuracy[-1], "final_f1": f1[-1], "n_labeled": n_labeled[-1]}

    # Оформление
    for ax, metric in zip(axes, ["Accuracy", "F1 (weighted)"]):
        ax.set_xlabel("Количество размеченных примеров", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"{metric} vs. N labeled", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Вывод в консоль
    print(f"\n{'='*55}")
    print(f"  СРАВНЕНИЕ СТРАТЕГИЙ ACTIVE LEARNING")
    print(f"{'='*55}")
    print(f"  {'Стратегия':<12} {'Accuracy':>10} {'F1':>8} {'Примеров':>10}")
    print(f"  {'-'*44}")
    for label, s in savings.items():
        print(f"  {label:<12} {s['final_acc']:>10.3f} {s['final_f1']:>8.3f} {s['n_labeled']:>10}")

    # Считать экономию относительно random
    if "random" in savings and len(savings) > 1:
        random_acc = savings["random"]["final_acc"]
        random_n = savings["random"]["n_labeled"]
        print(f"\n  Экономия примеров vs. random (при том же accuracy):")
        for label, s in savings.items():
            if label == "random":
                continue
            # Найти минимальное N при котором стратегия достигает random_acc
            history = load_history(history_paths[labels.index(label)])
            n_to_match = next(
                (h["n_labeled"] for h in history if h["accuracy"] >= random_acc),
                s["n_labeled"]
            )
            saved = random_n - n_to_match
            pct = saved / random_n * 100 if random_n > 0 else 0
            print(f"  {label}: -{saved} примеров (-{pct:.0f}%) для достижения {random_acc:.3f} acc")

    print(f"\n  График: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    visualize(args.histories, args.labels, args.output)
