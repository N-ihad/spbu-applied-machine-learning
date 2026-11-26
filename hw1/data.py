"""STL-10 binary files, training splits and image sampling."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import STL10
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
COUNTS = {"train": 5_000, "test": 8_000, "unlabeled": 100_000}
SIZE = 96
CLASSES = 10
ImageTransform = Callable[[Image.Image], torch.Tensor]


def prepare(root: Path) -> None:
    folder = root / STL10.base_folder
    files = STL10.train_list + STL10.test_list

    def verified() -> bool:
        return all(
            check_integrity(str(folder / name), md5) for name, md5 in files
        )

    if not verified() or not (folder / "class_names.txt").is_file():
        download_and_extract_archive(URL, str(root), md5=STL10.tgz_md5)
        if not verified():
            raise OSError(
                "STL-10 binary files failed their official MD5 checks."
            )
    print(f"Verified STL-10 in {folder}")


def images(root: Path, split: str) -> np.ndarray:
    path = root / STL10.base_folder / f"{split}_X.bin"
    # Stanford stores each channel in column-major order.
    data = np.memmap(
        path, dtype=np.uint8, mode="r", shape=(COUNTS[split], 3, SIZE, SIZE)
    )
    return data.transpose(0, 3, 2, 1)


def labels(root: Path, split: str) -> np.ndarray:
    path = root / STL10.base_folder / f"{split}_y.bin"
    return np.fromfile(path, dtype=np.uint8).astype(np.int64) - 1


def split_train(
    targets: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(CLASSES):
        indices = rng.permutation(np.flatnonzero(targets == label))
        validation.append(indices[:100])
        train.append(indices[100:])
    return np.concatenate(train), np.concatenate(validation)


class ImageSubset(Dataset):
    def __init__(
        self,
        pixels: np.ndarray,
        indices: np.ndarray,
        targets: np.ndarray,
        transform: ImageTransform,
    ) -> None:
        if len(indices) != len(targets):
            raise ValueError(
                "Each selected image needs one aligned target or index."
            )
        self.pixels = pixels
        self.indices = indices
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.pixels[self.indices[index]])
        return self.transform(image), int(self.targets[index])


def augmentation(stage: int) -> ImageTransform:
    if stage == 0:
        return T.Compose([T.RandomHorizontalFlip(), T.PILToTensor()])
    scale = (0.85, 0.75, 0.65)[stage - 1]
    jitter = (0.08, 0.15, 0.22)[stage - 1]
    return T.Compose(
        [
            T.RandomResizedCrop(SIZE, scale=(scale, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter
            ),
            T.PILToTensor(),
        ]
    )


def epoch_order(
    labeled: int, pseudo: int, batch_size: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = (labeled + batch_size - 1) // batch_size
    padding = steps * batch_size - labeled
    order = np.concatenate(
        (rng.permutation(labeled), rng.choice(labeled, padding))
    )
    batches = order.reshape(steps, batch_size)
    if pseudo:
        # Both arms share the first 3/4 of every batch and the update budget.
        extra = batch_size // 4
        batches[:, -extra:] = labeled + rng.integers(
            pseudo, size=(steps, extra)
        )
    return batches.ravel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and verify official STL-10."
    )
    parser.add_argument("--data", type=Path, default=DATA)
    prepare(parser.parse_args().data)
