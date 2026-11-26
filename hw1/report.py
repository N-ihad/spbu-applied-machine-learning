"""Evaluate locked checkpoints and draw figures from the saved measurements."""

import argparse
import json
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.transforms import PILToTensor

from data import DATA, HERE, ImageSubset, images, labels
from engine import evaluate
from models import NAMES, Classifier
from train import device_for

TITLES = {"resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-Small"}
ARMS = {"supervised": "Только разметка", "pseudo": "С псевдометками"}
COLORS = {"supervised": "#176879", "pseudo": "#b74b24"}


def paired_effect(
    truth: np.ndarray, base: np.ndarray, pseudo: np.ndarray
) -> dict:
    base_correct, pseudo_correct = base == truth, pseudo == truth
    gained = int(np.count_nonzero(pseudo_correct & ~base_correct))
    lost = int(np.count_nonzero(base_correct & ~pseudo_correct))
    n = len(truth)
    # A paired image bootstrap has just three possible per-image differences.
    counts = np.array([gained, lost, n - gained - lost])
    draws = np.random.default_rng(42).multinomial(n, counts / n, size=10_000)
    delta = (draws[:, 0] - draws[:, 1]) / n
    return {
        "gain_images": gained,
        "loss_images": lost,
        "accuracy_difference": (gained - lost) / n,
        "paired_bootstrap_95": np.quantile(delta, [0.025, 0.975]).tolist(),
    }


def measure(run: Path, root: Path, device: torch.device) -> dict:
    records = {
        name: json.loads((run / name / "training.json").read_text())
        for name in NAMES
    }
    if records[NAMES[0]]["settings"] != records[NAMES[1]]["settings"]:
        raise ValueError(
            "The architectures must use the same experimental settings."
        )
    truth = labels(root, "test")
    test = ImageSubset(
        images(root, "test"), np.arange(len(truth)), truth, PILToTensor()
    )
    batches = DataLoader(test, batch_size=128)
    classes = (
        (root / STL10.base_folder / "class_names.txt").read_text().splitlines()
    )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for name, record in records.items():
        model = Classifier(name, pretrained=False).to(device)
        predictions = {}
        for arm in ARMS:
            checkpoint = torch.load(
                run / name / arm / "weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(checkpoint)
            metrics, predictions[arm] = evaluate(model, batches, device)
            record["variants"][arm]["test"] = metrics
            print(
                f"{name} {arm} | test accuracy {metrics['accuracy']:.2%}",
                flush=True,
            )
        record["paired"] = paired_effect(
            truth, predictions["supervised"], predictions["pseudo"]
        )
    return {"classes": classes, "models": records}


def save_figure(fig: Figure, path: Path) -> None:
    with StringIO() as svg:
        fig.savefig(svg, format="svg", metadata={"Date": None})
        lines = (line.rstrip() for line in svg.getvalue().splitlines())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.close(fig)


def learning_curves(records: dict, output: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(10.2, 6.6), sharex=True, sharey="row"
    )
    for col, (name, record) in enumerate(records.items()):
        axes[0, col].set_title(TITLES[name])
        shared = record["shared"]
        for row, (field, title, scale) in enumerate(
            [
                ("val_accuracy", "Accuracy, %", 100),
                ("val_loss", "Кросс-энтропия", 1),
            ]
        ):
            ax = axes[row, col]
            ax.plot(
                [r["epoch"] for r in shared],
                [r[field] * scale for r in shared],
                "o-",
                color="#414b57",
                label="Общее обучение",
            )
            for arm, label in ARMS.items():
                rows = [shared[-1], *record["variants"][arm]["history"]]
                ax.plot(
                    [r["epoch"] for r in rows],
                    [r[field] * scale for r in rows],
                    "o-",
                    color=COLORS[arm],
                    label=label,
                )
            ax.grid(axis="y", alpha=0.22)
            ax.set_ylabel(title)
            ax.spines[["top", "right"]].set_visible(False)
        axes[1, col].set_xlabel("Эпоха")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Качество на валидационной выборке")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save_figure(fig, output / "learning_curves.svg")


def confusion_matrices(results: dict, output: Path) -> None:
    classes = results["classes"]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10), layout="constrained")
    for row, (name, record) in enumerate(results["models"].items()):
        for col, (arm, label) in enumerate(ARMS.items()):
            ax = axes[row, col]
            matrix = np.array(record["variants"][arm]["test"]["confusion"])
            maximum = int(matrix.sum(1).max())
            mesh = ax.pcolormesh(matrix, cmap="Blues", vmin=0, vmax=maximum)
            ticks = np.arange(len(classes)) + 0.5
            ax.set_xticks(ticks, classes, rotation=40, ha="right")
            ax.set_yticks(ticks, classes)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_title(f"{TITLES[name]} · {label}", fontsize=12)
            ax.set_xlabel("Предсказанный класс")
            ax.set_ylabel("Истинный класс")
            for i, j in zip(*np.where(matrix >= 8)):
                value = matrix[i, j]
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > maximum * 0.54 else "#182432",
                )
    fig.colorbar(mesh, ax=axes, shrink=0.64, label="Число изображений")
    save_figure(fig, output / "confusion_matrices.svg")


def main() -> None:
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-run", type=Path, help="Evaluate the four saved checkpoints."
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "metrics.json"
    if args.from_run:
        results = measure(args.from_run, args.data, device_for(args.device))
        path.write_text(json.dumps(results, indent=2) + "\n")
    else:
        results = json.loads(path.read_text())
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "stl10-hw1"})
    learning_curves(results["models"], args.output)
    confusion_matrices(results, args.output)


if __name__ == "__main__":
    main()
