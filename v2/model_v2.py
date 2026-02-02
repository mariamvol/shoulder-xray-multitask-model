import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNet121_3Head(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_feats = base.classifier.in_features

        self.head_frac = nn.Linear(in_feats, 1)  # fracture
        self.head_proj = nn.Linear(in_feats, 1)  # projection (S=1, D=0)
        self.head_hw   = nn.Linear(in_feats, 1)  # hardware

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = self.pool(f).flatten(1)
        return {
            "fracture": self.head_frac(f),
            "projection": self.head_proj(f),
            "hardware": self.head_hw(f),
        }


def build_model():
    return DenseNet121_3Head()
