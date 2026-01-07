import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResEmoteNet(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=8):
        super(ResEmoteNet, self).__init__()
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError
        
        self.header = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.header(features)
        return out, features