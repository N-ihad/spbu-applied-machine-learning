"""Optimization, evaluation and two-view pseudo-label selection."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import CLASSES


def train_epoch(
    model: nn.Module,
    batches: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    totals = torch.zeros(2, device=device)
    samples = 0
    for pixels, labels in batches:
        targets = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixels.to(device))
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        correct = (logits.detach().argmax(1) == targets).sum()
        totals += torch.stack((loss.detach() * len(targets), correct))
        samples += len(targets)
    loss, accuracy = (totals / samples).tolist()
    return loss, accuracy


@torch.inference_mode()
def evaluate(
    model: nn.Module, batches: DataLoader, device: torch.device
) -> tuple[dict, np.ndarray]:
    model.eval()
    confusion = torch.zeros(CLASSES, CLASSES, dtype=torch.long)
    loss_sum = torch.zeros((), device=device)
    predictions = []
    for pixels, labels in batches:
        logits = model(pixels.to(device))
        loss_sum += F.cross_entropy(logits, labels.to(device), reduction="sum")
        batch_predictions = logits.argmax(1).cpu()
        predictions.append(batch_predictions)
        pairs = labels * CLASSES + batch_predictions
        confusion += torch.bincount(pairs, minlength=CLASSES**2).view(
            CLASSES, CLASSES
        )
    diagonal = confusion.diag()
    total = int(confusion.sum())
    class_f1 = (
        2 * diagonal / (confusion.sum(0) + confusion.sum(1)).clamp_min(1)
    )
    metrics = {
        "n": total,
        "loss": loss_sum.item() / total,
        "accuracy": diagonal.sum().item() / total,
        "macro_f1": class_f1.mean().item(),
        "confusion": confusion.tolist(),
    }
    return metrics, torch.cat(predictions).numpy()


@dataclass
class Selected:
    keys: np.ndarray
    targets: np.ndarray
    scores: np.ndarray

    def capped(self, per_class: int) -> "Selected":
        keep = []
        for label in range(CLASSES):
            indices = np.flatnonzero(self.targets == label)
            order = np.argsort(-self.scores[indices], kind="stable")
            keep.extend(indices[order[:per_class]])
        indices = np.asarray(keep, dtype=np.int64)
        return Selected(
            self.keys[indices], self.targets[indices], self.scores[indices]
        )


@torch.inference_mode()
def confident_predictions(
    model: nn.Module,
    batches: DataLoader,
    device: torch.device,
    threshold: float,
) -> Selected:
    model.eval()
    keys, targets, scores = [], [], []
    for pixels, batch_keys in batches:
        original = pixels.to(device)
        views = torch.cat((original, original.flip(-1)))
        probabilities = model(views).softmax(1)
        confidence, prediction = probabilities.max(dim=1)
        left, right = prediction.chunk(2)
        score = torch.minimum(*confidence.chunk(2))
        mask = ((left == right) & (score >= threshold)).cpu()
        keys.append(batch_keys[mask])
        targets.append(left.cpu()[mask])
        scores.append(score.cpu()[mask])
    return Selected(
        torch.cat(keys).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(scores).numpy(),
    )
