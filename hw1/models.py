"""Torchvision classifiers with progressive unfreezing and per-block rates."""

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from data import CLASSES

NAMES = ("resnet18", "mobilenet_v3_small")
BLOCK_NAMES = ("early", "middle", "late", "head")
RATES = (3e-5, 8e-5, 2e-4, 1e-3)


class Classifier(nn.Module):
    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        if name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights, progress=False)
            net.fc = nn.Linear(net.fc.in_features, CLASSES)
            self.blocks = (
                (net.conv1, net.bn1, net.layer1, net.layer2),
                (net.layer3,),
                (net.layer4,),
                (net.fc,),
            )
        elif name == "mobilenet_v3_small":
            weights = (
                MobileNet_V3_Small_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            net = mobilenet_v3_small(weights=weights, progress=False)
            net.classifier[-1] = nn.Linear(
                net.classifier[-1].in_features, CLASSES
            )
            layers = tuple(net.features)
            self.blocks = (
                layers[:4],
                layers[4:9],
                layers[9:],
                (net.classifier,),
            )
        else:
            raise ValueError(f"Unknown model: {name}")
        self.network = net
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        normalized = (pixels.float().div(255) - self.mean) / self.std
        return self.network(normalized)

    def set_stage(self, stage: int, fixed_bn: bool = False) -> None:
        self.train()
        for depth, block in enumerate(self.blocks):
            active = depth >= len(self.blocks) - 1 - stage
            for module in block:
                module.requires_grad_(active)
                # Frozen BatchNorm layers must keep their running statistics.
                module.train(active)
        if fixed_bn:
            # Affine weights in active blocks still receive gradients.
            for layer in self.network.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()

    def optimizer(self) -> torch.optim.AdamW:
        groups = []
        for block, rate in zip(self.blocks, RATES, strict=True):
            params = [p for layer in block for p in layer.parameters()]
            groups.append({"params": params, "lr": rate})
        return torch.optim.AdamW(groups, weight_decay=1e-4)
