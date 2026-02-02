import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNet3Head(nn.Module):
    """
    DenseNet121 encoder + 3 binary heads:
      - fracture (0/1)
      - projection: S=1, D=0
      - hardware (0/1)
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.densenet121(weights=weights)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_feats = base.classifier.in_features

        self.head_frac = nn.Linear(in_feats, 1)
        self.head_proj = nn.Linear(in_feats, 1)
        self.head_hw   = nn.Linear(in_feats, 1)

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = self.pool(f).flatten(1)
        return {
            "fracture":   self.head_frac(f),
            "projection": self.head_proj(f),
            "hardware":   self.head_hw(f),
        }


def build_model_v2(pretrained: bool = False) -> nn.Module:
    return DenseNet3Head(pretrained=pretrained)

