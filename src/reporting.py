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
