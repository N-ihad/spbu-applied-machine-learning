"""Train two matched supervised and pseudo-labeled variants on STL-10."""

import argparse
import copy
import json
import os
import platform
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import PILToTensor

from data import (
    CLASSES,
    COUNTS,
    DATA,
    HERE,
    ImageSubset,
    augmentation,
    epoch_order,
    images,
    labels,
    split_train,
)
from engine import Selected, confident_predictions, evaluate, train_epoch
from models import BLOCK_NAMES, NAMES, Classifier


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    batch_size: int = 64
    epochs_per_stage: int = 3
    train_per_class: int = 400
    val_per_class: int = 100
    pool_size: int = COUNTS["unlabeled"]
    threshold: float = 0.90
    cap_per_class: int = 300
    fixed_bn: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


class TrainingData:
    def __init__(self, root: Path, cfg: Settings) -> None:
        self.root = root
        self.labeled = images(root, "train")
        self.targets = labels(root, "train")
        train, val = split_train(self.targets, cfg.seed)
        self.train = train.reshape(CLASSES, -1)[
            :, : cfg.train_per_class
        ].ravel()
        self.val = val.reshape(CLASSES, -1)[:, : cfg.val_per_class].ravel()
        rng = np.random.default_rng(cfg.seed + 1)
        self.pool = np.arange(COUNTS["unlabeled"])
        if cfg.pool_size < len(self.pool):
            self.pool = rng.choice(
                self.pool, size=cfg.pool_size, replace=False
            )
        dataset = ImageSubset(
            self.labeled, self.val, self.targets[self.val], PILToTensor()
        )
        self.validation = DataLoader(dataset, batch_size=cfg.batch_size * 2)

    @cached_property
    def unlabeled(self) -> np.ndarray:
        return images(self.root, "unlabeled")


class Trainer:
    def __init__(
        self,
        name: str,
        data: TrainingData,
        cfg: Settings,
        device: torch.device,
        pretrained: bool = True,
    ) -> None:
        set_seed(cfg.seed)
        self.model = Classifier(name, pretrained=pretrained).to(device)
        self.optimizer = self.model.optimizer()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=0, threshold=0.01
        )
        self.data = data
        self.cfg = cfg
        self.device = device

    def epoch(self, epoch: int, selected: Selected | None = None) -> dict:
        cfg, data = self.cfg, self.data
        stage = (epoch - 1) // cfg.epochs_per_stage
        self.model.set_stage(stage, self.cfg.fixed_bn)
        seed = cfg.seed + 100 * epoch
        set_seed(seed)
        transform = augmentation(stage)
        dataset = ImageSubset(
            data.labeled, data.train, data.targets[data.train], transform
        )
        n_labeled, n_selected = len(dataset), 0
        if selected is not None:
            pseudo = ImageSubset(
                data.unlabeled, selected.keys, selected.targets, transform
            )
            n_selected = len(pseudo)
            dataset = ConcatDataset([dataset, pseudo])
        order = epoch_order(n_labeled, n_selected, cfg.batch_size, seed)
        batches = DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=order.tolist()
        )
        rates = {
            k: g["lr"]
            for k, g in zip(BLOCK_NAMES, self.optimizer.param_groups)
        }
        loss, accuracy = train_epoch(
            self.model, batches, self.optimizer, self.device
        )
        val, _ = evaluate(self.model, data.validation, self.device)
        self.scheduler.step(val["loss"])
        pseudo_draws = int((order >= n_labeled).sum())
        print(
            f"  epoch {epoch:2}  stage {stage}  train loss {loss:.3f}  "
            f"val loss {val['loss']:.3f}  val accuracy {val['accuracy']:.1%}",
            flush=True,
        )
        return {
            "epoch": epoch,
            "stage": stage,
            "steps": len(batches),
            "labeled_draws": len(order) - pseudo_draws,
            "pseudo_draws": pseudo_draws,
            "trainable_params": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "lr": rates,
            "train_loss": loss,
            "train_accuracy": accuracy,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
        }

    def pseudo_labels(self) -> tuple[Selected, dict]:
        data, cfg = self.data, self.cfg
        dataset = ImageSubset(
            data.unlabeled, data.pool, data.pool, PILToTensor()
        )
        batches = DataLoader(dataset, batch_size=cfg.batch_size * 2)
        candidates = confident_predictions(
            self.model, batches, self.device, cfg.threshold
        )
        selected = candidates.capped(cfg.cap_per_class)
        if not len(selected.keys):
            raise RuntimeError(
                "No pseudo-labels passed the teacher's confidence threshold."
            )
        diagnostic = confident_predictions(
            self.model, data.validation, self.device, cfg.threshold
        )
        precision = None
        if len(diagnostic.keys):
            precision = float((diagnostic.keys == diagnostic.targets).mean())
        return selected, {
            "pool_size": len(data.pool),
            "above_threshold_and_agree": len(candidates.keys),
            "used": len(selected.keys),
            "per_class": np.bincount(
                selected.targets, minlength=CLASSES
            ).tolist(),
            "min_confidence": float(selected.scores.min()),
            "mean_confidence": float(selected.scores.mean()),
            "validation_accepted": len(diagnostic.keys),
            "validation_total": len(data.val),
            "validation_precision": precision,
        }

    def teacher(self, name: str, run: Path) -> Path:
        path = run / f"{name}-teacher.pt"
        if path.exists():
            return path
        print(f"{name} | shared teacher", flush=True)
        shared = [
            self.epoch(i) for i in range(1, 2 * self.cfg.epochs_per_stage + 1)
        ]
        snapshot = {
            "settings": asdict(self.cfg),
            "shared": shared,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        with tempfile.NamedTemporaryFile(dir=run, suffix=".pt") as tmp:
            torch.save(snapshot, tmp)
            tmp.flush()
            # Keep an interrupted write from publishing a partial checkpoint.
            os.link(tmp.name, path)
        return path

    def fit_arm(self, arm: str, output: Path, snapshot: dict) -> dict:
        # Restore identical weights, Adam moments and LR for each comparison.
        self.model.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        self.scheduler.load_state_dict(copy.deepcopy(snapshot["scheduler"]))
        selected, stats = None, None
        if arm == "pseudo":
            selected, stats = self.pseudo_labels()
            np.savez(output / "pseudo_labels.npz", **vars(selected))
            print(
                f"  pseudo-labels: {stats['used']} / {stats['pool_size']}",
                flush=True,
            )
        history = []
        best_key = (-1.0, -float("inf"))
        n = self.cfg.epochs_per_stage
        for epoch in range(2 * n + 1, 4 * n + 1):
            row = self.epoch(epoch, selected)
            history.append(row)
            key = (row["val_accuracy"], -row["val_loss"])
            if key > best_key:
                best_key = key
                best_epoch = epoch
                torch.save(self.model.state_dict(), output / "weights.pt")
        return {
            "best_epoch": best_epoch,
            "history": history,
            "pseudo_labels": stats,
        }


def train_model(
    name: str,
    run: Path,
    data: TrainingData,
    cfg: Settings,
    device: torch.device,
    through: str,
) -> None:
    destination = run / name
    destination.mkdir(exist_ok=True)
    manifest = destination / "training.json"
    if manifest.exists():
        settings = json.loads(manifest.read_text())["settings"]
        if {"fixed_bn": False, **settings} != asdict(cfg):
            raise ValueError(f"{manifest} uses different settings.")
        print(f"Already trained: {destination}", flush=True)
        return
    cached_teacher = (run / f"{name}-teacher.pt").exists()
    trainer = Trainer(name, data, cfg, device, pretrained=not cached_teacher)
    path = trainer.teacher(name, run)
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    saved_settings = {"fixed_bn": False, **snapshot["settings"]}
    if saved_settings != asdict(cfg):
        raise ValueError(
            f"{path} uses different settings. Choose a fresh --run."
        )
    if through == "teacher":
        print(f"Shared checkpoint ready: {path}", flush=True)
        return
    variants = {}
    for arm in ("supervised", "pseudo"):
        target = destination / arm
        if not target.is_dir():
            print(f"{name} | {arm}", flush=True)
            # Only complete runs become visible to retries and evaluation.
            with tempfile.TemporaryDirectory(
                dir=destination, prefix=f".{arm}-"
            ) as tmp:
                output = Path(tmp)
                log = trainer.fit_arm(arm, output, snapshot)
                (output / "history.json").write_text(
                    json.dumps(log, indent=2) + "\n"
                )
                os.rename(output, target)
        variants[arm] = json.loads((target / "history.json").read_text())
        if arm == through:
            break
    if through == "supervised":
        return
    stats = variants["pseudo"].pop("pseudo_labels")
    variants["supervised"].pop("pseudo_labels")
    result = {
        "model": name,
        "settings": asdict(cfg),
        "shared": snapshot["shared"],
        "variants": variants,
        "pseudo_labels": stats,
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": torchvision.__version__,
            "device": str(device),
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", dir=destination, delete=False
    ) as tmp:
        tmp.write(json.dumps(result, indent=2) + "\n")
    os.replace(tmp.name, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "full")
    parser.add_argument("--model", choices=("both", *NAMES), default="both")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Small real-data smoke run."
    )
    parser.add_argument(
        "--bn-stats",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="Keep or update BatchNorm running statistics in unfrozen blocks.",
    )
    parser.add_argument(
        "--through",
        choices=("teacher", "supervised", "pseudo"),
        default="pseudo",
        help="Stop after this stage; a repeat run reuses completed stages.",
    )
    args = parser.parse_args()
    cfg = Settings(fixed_bn=args.bn_stats == "fixed")
    if args.quick:
        cfg = replace(
            cfg,
            batch_size=32,
            epochs_per_stage=1,
            train_per_class=80,
            val_per_class=20,
            pool_size=1_000,
            threshold=0.7,
            cap_per_class=60,
        )
    run = args.run
    if args.quick and run == HERE / "runs" / "full":
        run = HERE / "runs" / "quick"
    run.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(HERE / ".cache" / "torch" / "hub"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = device_for(args.device)
    data = TrainingData(args.data, cfg)
    names = NAMES if args.model == "both" else (args.model,)
    for name in names:
        train_model(name, run, data, cfg, device, args.through)


if __name__ == "__main__":
    main()
