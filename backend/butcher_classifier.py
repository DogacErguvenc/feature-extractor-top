import torch
import torch.nn as nn
from torchvision import models


class ButcherClassifier(nn.Module):
    """
    butcher-vision ile aynı sınıflandırıcı:
    ResNet18 backbone + kasaba özel linear head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        features = self.backbone(x)
        logits = self.head(features)

        if return_features:
            return logits, features

        return logits
