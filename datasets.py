"""
Unified dataset for SSF4VSU: frames, target_prior, bbox, labels, mask, task_type.
Resolution 640x360 (default) or 1280x720; ImageNet normalization; task-specific augmentations.
"""
import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ImageNet mean/std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default resolution (640x360 or 1280x720)
DEFAULT_RESOLUTION = (640, 360)


def get_transforms(augment=True, resolution=DEFAULT_RESOLUTION, task="SOT"):
    """
    Preprocessing: resize, optional augmentations, ToTensor, ImageNet normalize.
    """
    H, W = resolution[1], resolution[0]
    base = [
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if not augment:
        return transforms.Compose(base)
    # Common
    aug_list = [
        transforms.Resize((H, W)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(aug_list)


def bbox_to_prior_map(bbox, H, W):
    """Convert bbox [x, y, w, h] (normalized 0-1 or pixel) to binary spatial map [1, H, W]."""
    prior = torch.zeros(1, H, W)
    if bbox[2] <= 0 or bbox[3] <= 0:
        return prior
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    if x <= 1 and y <= 1 and w <= 1 and h <= 1:
        x, y, w, h = x * W, y * H, w * W, h * H
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    prior[:, y1:y2, x1:x2] = 1.0
    return prior


class MultiTaskDataset(Dataset):
    """
    Unified dataset for SOT, MOT, VOS, MOTS.
    Returns task_type for loss routing; target_prior for unified embedding (zeros for MOT/MOTS if no target).
    """

    def __init__(
        self,
        root,
        task_type="SOT",
        augment=True,
        seq_len=8,
        ssl=True,
        resolution=DEFAULT_RESOLUTION,
    ):
        self.root = root
        self.task_type = task_type.upper()
        self.seq_len = seq_len
        self.ssl = ssl
        self.resolution = resolution
        self.H, self.W = resolution[1], resolution[0]
        self.samples = self._load_samples(root)
        self.transform = get_transforms(augment, resolution, self.task_type)

    def _load_samples(self, root):
        samples = []
        if not os.path.isdir(root):
            return samples
        for video_dir in sorted(os.listdir(root)):
            vpath = os.path.join(root, video_dir)
            if not os.path.isdir(vpath):
                continue
            frames = sorted(glob.glob(os.path.join(vpath, "*.jpg")) + glob.glob(os.path.join(vpath, "*.png")))
            ann_path = os.path.join(vpath, "annots.txt")
            if os.path.exists(ann_path):
                samples.append((frames, ann_path))
            else:
                samples.append((frames, None))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_annotations(self, ann_path):
        if ann_path is None:
            return {}
        annots = {}
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                frame_idx = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                label = int(parts[5]) if len(parts) > 5 else -1
                mask_path = parts[6].strip() if len(parts) > 6 else None
                annots[frame_idx] = {"bbox": bbox, "label": label, "mask_path": mask_path}
        return annots

    def __getitem__(self, idx):
        frames, ann_path = self.samples[idx]
        annots = self._load_annotations(ann_path) if ann_path else {}

        if len(frames) >= self.seq_len:
            start = random.randint(0, len(frames) - self.seq_len)
            frame_paths = frames[start : start + self.seq_len]
        else:
            frame_paths = frames

        imgs, bboxes, labels, masks, target_priors = [], [], [], [], []
        first_bbox = None
        first_mask = None

        for i, fp in enumerate(frame_paths):
            img = Image.open(fp).convert("RGB")
            img_t = self.transform(img)
            imgs.append(img_t)

            if i in annots:
                ann = annots[i]
                bbox = ann["bbox"]
                label = ann["label"]
                if first_bbox is None:
                    first_bbox = bbox
                bboxes.append(bbox)
                labels.append(label)
                if ann.get("mask_path") and os.path.exists(ann["mask_path"]):
                    mask_img = Image.open(ann["mask_path"]).convert("L")
                    mask_t = transforms.Resize((self.H, self.W))(transforms.ToTensor()(mask_img))
                    masks.append(mask_t)
                    if first_mask is None:
                        first_mask = mask_t
                    # Target prior for SOT/VOS: first-frame mask or bbox as spatial map
                    if self.task_type in ("SOT", "VOS") and first_mask is not None:
                        prior = first_mask
                    else:
                        prior = bbox_to_prior_map(bbox, self.H, self.W)
                else:
                    masks.append(torch.zeros(1, self.H, self.W))
                    prior = bbox_to_prior_map(bbox, self.H, self.W)
                target_priors.append(prior)
            else:
                bboxes.append([0.0, 0.0, 0.0, 0.0])
                labels.append(-1)
                masks.append(torch.zeros(1, self.H, self.W))
                # MOT/MOTS: neutral prior (zeros); SOT/VOS: use first frame prior if available
                if self.task_type in ("MOT", "MOTS"):
                    target_priors.append(torch.zeros(1, self.H, self.W))
                else:
                    target_priors.append(first_mask if first_mask is not None else torch.zeros(1, self.H, self.W))

        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        target_priors = torch.stack(target_priors)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        ssl_z1, ssl_z2 = None, None
        if self.ssl and len(frame_paths) > 0:
            aug_transform = get_transforms(True, self.resolution, self.task_type)
            z1_frames = [aug_transform(Image.open(fp).convert("RGB")) for fp in frame_paths]
            z2_frames = [aug_transform(Image.open(fp).convert("RGB")) for fp in frame_paths]
            ssl_z1 = torch.stack(z1_frames)
            ssl_z2 = torch.stack(z2_frames)

        return {
            "frames": imgs,
            "target_prior": target_priors,
            "bbox": bboxes,
            "labels": labels,
            "mask": masks,
            "task_type": self.task_type,
            "ssl_z1": ssl_z1,
            "ssl_z2": ssl_z2,
        }


class MultiTaskDatasetCombined(Dataset):
    """
    Combined dataset with task-balanced sampling.
    Sampling proportions: ~60% MOT, 25% SOT, 10% VOS, 5% MOTS.
    Each item has task_type for loss routing.
    """
    TASK_WEIGHTS = {"SOT": 0.25, "MOT": 0.60, "VOS": 0.10, "MOTS": 0.05}

    def __init__(self, roots_by_task, augment=True, seq_len=8, ssl=True, resolution=DEFAULT_RESOLUTION):
        self.datasets = {}
        self.task_keys = []
        self.cumulative = []
        cum = 0
        for task, root in roots_by_task.items():
            if not os.path.isdir(root):
                continue
            ds = MultiTaskDataset(root, task_type=task, augment=augment, seq_len=seq_len, ssl=ssl, resolution=resolution)
            if len(ds) > 0:
                self.datasets[task] = ds
                self.task_keys.append(task)
                cum += len(ds)
                self.cumulative.append(cum)
        self.total = cum

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        idx = idx % self.total
        for i, task in enumerate(self.task_keys):
            if idx < self.cumulative[i]:
                start = self.cumulative[i - 1] if i > 0 else 0
                return self.datasets[task][idx - start]
        return self.datasets[self.task_keys[0]][0]
