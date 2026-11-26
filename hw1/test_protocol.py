"""Regression checks for STL-10, freezing, sampling and pseudo-labels."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import CLASSES, COUNTS, SIZE, epoch_order, images, split_train
from engine import Selected, confident_predictions, evaluate
from models import NAMES, Classifier


class ProtocolTests(unittest.TestCase):
    def test_metrics_use_true_rows_and_macro_average(self) -> None:
        truth = torch.tensor([0, *range(CLASSES)])
        prediction = truth.clone()
        prediction[0] = 1
        logits = torch.nn.functional.one_hot(prediction, CLASSES).float()
        loader = DataLoader(TensorDataset(logits, truth), batch_size=4)
        metrics, observed = evaluate(
            nn.Identity(), loader, torch.device("cpu")
        )
        np.testing.assert_array_equal(observed, prediction.numpy())
        self.assertEqual(metrics["confusion"][0][1], 1)
        self.assertEqual(metrics["confusion"][1][0], 0)
        self.assertAlmostEqual(metrics["accuracy"], 10 / 11)
        self.assertAlmostEqual(metrics["macro_f1"], (8 + 4 / 3) / 10, places=6)

    def test_stanford_column_major_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "stl10_binary"
            folder.mkdir()
            rgb = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            rgb[2, 9] = [12, 130, 244]
            rgb.transpose(2, 1, 0).tofile(folder / "train_X.bin")
            with patch.dict(COUNTS, train=1):
                np.testing.assert_array_equal(images(root, "train")[0], rgb)

    def test_stratified_disjoint_split(self) -> None:
        targets = np.tile(np.arange(CLASSES), 500)
        train, val = split_train(targets, seed=42)
        np.testing.assert_array_equal(
            np.bincount(targets[train]), np.full(CLASSES, 400)
        )
        np.testing.assert_array_equal(
            np.bincount(targets[val]), np.full(CLASSES, 100)
        )
        np.testing.assert_array_equal(
            np.sort(np.r_[train, val]), np.arange(5_000)
        )
        np.testing.assert_array_equal(split_train(targets, seed=42)[1], val)

    def test_paired_batch_budgets_and_indices(self) -> None:
        base = epoch_order(4_000, 0, 64, 142).reshape(-1, 64)
        mixed = epoch_order(4_000, 173, 64, 142).reshape(-1, 64)
        self.assertEqual(base.shape, (63, 64))
        self.assertEqual(base.shape, mixed.shape)
        np.testing.assert_array_equal(base[:, :48], mixed[:, :48])
        self.assertTrue(
            np.all((mixed[:, 48:] >= 4_000) & (mixed[:, 48:] < 4_173))
        )
        self.assertEqual(len(np.unique(base)), 4_000)

    def test_freezing_includes_batch_norm_and_all_parameters(self) -> None:
        for name in NAMES:
            with self.subTest(model=name):
                model = Classifier(name, pretrained=False)
                optimizer = model.optimizer()
                params = [
                    id(p)
                    for group in optimizer.param_groups
                    for p in group["params"]
                ]
                self.assertCountEqual(
                    params, [id(p) for p in model.parameters()]
                )
                first_bn = next(
                    m for m in model.modules() if isinstance(m, nn.BatchNorm2d)
                )
                counts = []
                for stage in range(4):
                    model.set_stage(stage)
                    self.assertEqual(first_bn.training, stage == 3)
                    counts.append(
                        sum(
                            p.numel()
                            for p in model.parameters()
                            if p.requires_grad
                        )
                    )
                self.assertTrue(all(a < b for a, b in zip(counts, counts[1:])))
                self.assertEqual(
                    counts[-1], sum(p.numel() for p in model.parameters())
                )
                model.set_stage(3, fixed_bn=True)
                self.assertFalse(first_bn.training)
                self.assertTrue(first_bn.weight.requires_grad)

    def test_two_views_must_agree_and_both_pass(self) -> None:
        class Lookup(nn.Module):
            def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                # The first pixel selects a class. Flipping swaps the views.
                scores = torch.tensor([[9.0, 0.0], [0.0, 9.0], [0.2, 0.0]])
                return scores[pixels[:, 0, 0, 0].long()]

        pixels = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 1]]).view(
            4, 1, 1, 2
        )
        data = TensorDataset(pixels, torch.tensor([91, 24, 62, 12]))
        found = confident_predictions(
            Lookup(), DataLoader(data, 3), torch.device("cpu"), 0.9
        )
        np.testing.assert_array_equal(found.keys, [91, 12])
        np.testing.assert_array_equal(found.targets, [0, 1])
        ranked = Selected(
            np.array([4, 7, 9]),
            np.array([0, 1, 0]),
            np.array([0.91, 0.92, 0.97]),
        )
        np.testing.assert_array_equal(ranked.capped(1).keys, [9, 7])


if __name__ == "__main__":
    unittest.main()
