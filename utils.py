import os
import torch
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_checkpoint(model, optimizer, epoch, path):
    """Save model + optimizer state."""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"✅ Saved checkpoint at {path}")


def load_checkpoint(model, optimizer, path, device="cuda"):
    """Load model + optimizer state."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"🔄 Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def visualize_bbox(image, bbox, label=None, save_path=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    rect = patches.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3],
        linewidth=2, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)
    if label is not None:
        plt.text(bbox[0], bbox[1] - 5, f"ID:{label}", color="red")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_mask(image, mask, save_path=None):
    img = image.permute(1, 2, 0).cpu().numpy()
    m = mask.squeeze().cpu().numpy()
    plt.imshow(img)
    plt.imshow(m, cmap="Reds", alpha=0.5)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
