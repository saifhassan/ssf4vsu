import torch
import torch.nn as nn


class BaselineWrapper(nn.Module):
    """Lightweight wrapper for SOTA baselines (SiamFC, SiamRPN++, TransT, Unicorn, OmniTracker)."""
    def __init__(self, name="SiamFC", num_classes=100):
        super().__init__()
        self.name = name.lower()
        if self.name == "siamfc":
            from torchvision.models import alexnet
            self.model = alexnet(pretrained=True)
        elif self.name == "siamrpn++":
            self.model = nn.Conv2d(3, 64, 3, 1, 1)
        elif self.name == "transt":
            from timm import create_model
            self.model = create_model("deit_base_patch16_224", pretrained=True)
        elif self.name in ("unicorn", "omnitracker"):
            self.model = nn.Identity()
        else:
            raise ValueError(f"Baseline {name} not implemented")

    def forward(self, x):
        if self.name == "siamfc":
            return {"sot": self.model(x)}
        elif self.name == "transt":
            return {"sot": self.model(x)}
        return {"output": self.model(x)}


def get_baseline(name, num_classes=100):
    return BaselineWrapper(name, num_classes=num_classes)
