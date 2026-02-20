import csv
from typing import Dict, List


def save_history_csv(path: str, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def load_dicts_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_dicts_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: str, history: List[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["training_loss"] for h in history]
    val_loss = [h["validation_loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_search_seed_loss_plot(path: str, losses_by_arch: Dict[str, List[float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    arch_names = []
    seed_losses = []
    for arch_name, losses in losses_by_arch.items():
        if losses:
            arch_names.append(arch_name)
            seed_losses.append(losses)

    if not seed_losses:
        return

    fig_w = max(7.0, len(arch_names) * 0.75)
    plt.figure(figsize=(fig_w, 5.2))

    positions = list(range(1, len(seed_losses) + 1))
    boxplot = plt.boxplot(
        seed_losses,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "#d00000",
            "markeredgecolor": "#d00000",
            "markersize": 4,
        },
    )

    for patch in boxplot["boxes"]:
        patch.set(facecolor="#8ecae6", alpha=0.55, edgecolor="#1d3557")

    for whisker in boxplot["whiskers"]:
        whisker.set(color="#1d3557")
    for cap in boxplot["caps"]:
        cap.set(color="#1d3557")
    for median in boxplot["medians"]:
        median.set(color="#1d3557", linewidth=1.25)

    for idx, losses in enumerate(seed_losses, start=1):
        if len(losses) == 1:
            xs = [idx]
        else:
            xs = [
                idx - 0.16 + 0.32 * (i / (len(losses) - 1))
                for i in range(len(losses))
            ]
        plt.scatter(xs, losses, s=18, color="#1d3557", alpha=0.85, zorder=3)

    plt.xticks(positions, arch_names, rotation=40, ha="right")
    plt.ylabel("Test Loss")
    plt.xlabel("Architecture (own optimal hyperparameters)")
    plt.title("Search Phase Test Loss Across Retrain Seeds")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "var": float("nan"), "min": float("nan"), "max": float("nan")}

    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "mean": mean,
        "var": var,
        "min": min(values),
        "max": max(values),
    }


def running_best(values: List[float]) -> List[float]:
    best = []
    current = float("inf")
    for v in values:
        current = min(current, v)
        best.append(current)
    return best
