"""
Two-stage training: Stage 1 (SOT+MOT detection), Stage 2 (VOS+MOTS segmentation), then joint fine-tune.
Warm-up + multi-step LR decay; task-conditioned loss routing; gradient clipping.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SSF4VSU
from losses import TotalLoss
from datasets import MultiTaskDataset, MultiTaskDatasetCombined
from utils import save_checkpoint, AverageMeter


def _get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def get_warmup_multistep_scheduler(optimizer, warmup_epochs, total_epochs, milestones=None):
    """Warm-up then multi-step decay (0.1x at milestones)."""
    if milestones is None:
        milestones = [int(0.5 * total_epochs), int(0.8 * total_epochs)]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        factor = 1.0
        for m in milestones:
            if epoch >= m:
                factor *= 0.1
        return factor

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------
#  Training one epoch
# ----------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, stage=1, grad_clip=5.0):
    model.train()
    loss_meters = {k: AverageMeter() for k in ["total", "det", "mask", "ssl", "tcm"]}

    for batch in dataloader:
        x_seq = batch["frames"].to(device)
        target_prior = batch.get("target_prior")
        if target_prior is not None:
            target_prior = target_prior.to(device)
        targets = {
            "bbox": batch["bbox"].to(device),
            "labels": batch["labels"].to(device),
            "mask": batch["mask"].to(device),
        }
        ssl_emb = None
        if "ssl_z1" in batch and batch["ssl_z1"] is not None:
            ssl_emb = (batch["ssl_z1"].to(device), batch["ssl_z2"].to(device))
        task_type = batch.get("task_type", "SOT")
        if isinstance(task_type, (list, tuple)):
            task_type = task_type[0]

        outputs = model(x_seq, target_prior=target_prior)
        losses = criterion(outputs, targets, task_type=task_type, ssl_emb=ssl_emb, stage=stage)

        optimizer.zero_grad()
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in loss_meters:
            v = losses[k]
            loss_meters[k].update(v.item() if torch.is_tensor(v) else v, x_seq.size(0))

    return {k: v.avg for k, v in loss_meters.items()}


# ----------------------------
#  Validation
# ----------------------------
@torch.no_grad()
def validate(model, dataloader, criterion, device, stage=1):
    model.eval()
    loss_meters = {k: AverageMeter() for k in ["total", "det", "mask", "ssl", "tcm"]}

    for batch in dataloader:
        x_seq = batch["frames"].to(device)
        target_prior = batch.get("target_prior")
        if target_prior is not None:
            target_prior = target_prior.to(device)
        targets = {
            "bbox": batch["bbox"].to(device),
            "labels": batch["labels"].to(device),
            "mask": batch["mask"].to(device),
        }
        ssl_emb = None
        if "ssl_z1" in batch and batch["ssl_z1"] is not None:
            ssl_emb = (batch["ssl_z1"].to(device), batch["ssl_z2"].to(device))
        task_type = batch.get("task_type", "SOT")
        if isinstance(task_type, (list, tuple)):
            task_type = task_type[0]

        outputs = model(x_seq, target_prior=target_prior)
        losses = criterion(outputs, targets, task_type=task_type, ssl_emb=ssl_emb, stage=stage)

        for k in loss_meters:
            v = losses[k]
            loss_meters[k].update(v.item() if torch.is_tensor(v) else v, x_seq.size(0))

    return {k: v.avg for k, v in loss_meters.items()}


# ----------------------------
#  Main: two-stage training
# ----------------------------
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = config.get("resolution", (640, 360))
    stage1_epochs = config.get("stage1_epochs", 50)
    stage2_epochs = config.get("stage2_epochs", 20)
    finetune_epochs = config.get("finetune_epochs", 5)
    warmup_epochs = config.get("warmup_epochs", 5)
    batch_size = config.get("batch_size", 8)
    lr_backbone = config.get("lr_backbone", 1e-4)
    lr_heads = config.get("lr_heads", 1e-3)
    roots_by_task = config.get("roots_by_task", None)

    # Datasets: per-task roots or single folder
    if roots_by_task:
        train_ds = MultiTaskDatasetCombined(
            roots_by_task,
            augment=True,
            seq_len=config.get("seq_len", 8),
            ssl=config.get("ssl", True),
            resolution=resolution,
        )
        val_roots = config.get("val_roots_by_task", roots_by_task)
        val_ds = MultiTaskDatasetCombined(
            val_roots,
            augment=False,
            seq_len=config.get("seq_len", 8),
            ssl=False,
            resolution=resolution,
        ) if val_roots else None
    else:
        train_ds = MultiTaskDataset(
            config["train_data"],
            task_type=config.get("task_type", "SOT"),
            augment=True,
            seq_len=config.get("seq_len", 8),
            ssl=config.get("ssl", True),
            resolution=resolution,
        )
        val_ds = MultiTaskDataset(
            config["val_data"],
            task_type=config.get("task_type", "SOT"),
            augment=False,
            seq_len=config.get("seq_len", 8),
            ssl=False,
            resolution=resolution,
        ) if config.get("val_data") else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
    ) if val_ds and len(val_ds) > 0 else None

    model = SSF4VSU(
        backbone_type=config.get("backbone", "resnet50"),
        embed_dim=config.get("embed_dim", 256),
        num_classes=config.get("num_classes", 100),
        prior_alpha=config.get("prior_alpha", 1.0),
    ).to(device)

    criterion = TotalLoss(
        lambda_det=config.get("lambda_det", 1.0),
        lambda_mask=config.get("lambda_mask", 1.0),
        lambda_ssl=config.get("lambda_ssl", 0.5),
        lambda_tcm=config.get("lambda_tcm", 0.5),
    )

    # Param groups: backbone vs heads (different learning rates)
    backbone_params = list(model.backbone.parameters()) + list(model.fpn.parameters())
    other_params = [p for n, p in model.named_parameters() if p not in set(backbone_params)]
    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": lr_backbone},
        {"params": other_params, "lr": lr_heads},
        {"params": criterion.parameters(), "lr": lr_heads},
        weight_decay=1e-4,
    )
    grad_clip = config.get("grad_clip", 5.0)
    best_val_loss = float("inf")
    start_epoch = 0

    # ---------- Stage 1: SOT + MOT (detection only) ----------
    total_s1 = stage1_epochs
    scheduler_s1 = get_warmup_multistep_scheduler(optimizer, warmup_epochs, total_s1)
    for epoch in range(start_epoch, stage1_epochs):
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, stage=1, grad_clip=grad_clip
        )
        scheduler_s1.step()
        print(f"[Stage 1] Epoch {epoch+1}/{stage1_epochs} LR={_get_lr(optimizer):.2e} Train {train_losses}")
        if val_loader:
            val_losses = validate(model, val_loader, criterion, device, stage=1)
            print(f"  Val {val_losses}")
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                save_checkpoint(model, optimizer, epoch, config["checkpoint_path"])
    start_epoch = stage1_epochs

    # ---------- Stage 2: VOS + MOTS (segmentation); reduce backbone LR ----------
    lr_backbone_s2 = config.get("lr_backbone_stage2", 1e-5)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone_s2},
            {"params": other_params, "lr": lr_heads},
            {"params": list(criterion.parameters()), "lr": lr_heads},
        ],
        weight_decay=1e-4,
    )
    total_s2 = stage2_epochs
    scheduler_s2 = get_warmup_multistep_scheduler(optimizer, min(2, stage2_epochs), total_s2)
    for epoch in range(start_epoch, start_epoch + stage2_epochs):
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, stage=2, grad_clip=grad_clip
        )
        scheduler_s2.step()
        print(f"[Stage 2] Epoch {epoch+1}/{start_epoch + stage2_epochs} LR={_get_lr(optimizer):.2e} Train {train_losses}")
        if val_loader:
            val_losses = validate(model, val_loader, criterion, device, stage=2)
            print(f"  Val {val_losses}")
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                save_checkpoint(model, optimizer, epoch, config["checkpoint_path"])
    start_epoch += stage2_epochs

    # ---------- Joint fine-tune (all tasks, low LR) ----------
    lr_finetune = config.get("lr_finetune", 1e-5)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr_finetune,
        weight_decay=1e-4,
    )
    for epoch in range(start_epoch, start_epoch + finetune_epochs):
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, stage=3, grad_clip=grad_clip
        )
        print(f"[Fine-tune] Epoch {epoch+1}/{start_epoch + finetune_epochs} Train {train_losses}")
        if val_loader:
            val_losses = validate(model, val_loader, criterion, device, stage=3)
            print(f"  Val {val_losses}")
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                save_checkpoint(model, optimizer, epoch, config["checkpoint_path"])

    print("Training complete. Best Val Loss:", best_val_loss)


if __name__ == "__main__":
    config = {
        "train_data": "./data/train/",
        "val_data": "./data/val/",
        "backbone": "resnet50",
        "embed_dim": 256,
        "num_classes": 100,
        "resolution": (640, 360),
        "batch_size": 8,
        "stage1_epochs": 50,
        "stage2_epochs": 20,
        "finetune_epochs": 5,
        "warmup_epochs": 5,
        "lr_backbone": 1e-4,
        "lr_heads": 1e-3,
        "lr_backbone_stage2": 1e-5,
        "lr_finetune": 1e-5,
        "lambda_det": 1.0,
        "lambda_mask": 1.0,
        "lambda_ssl": 0.5,
        "lambda_tcm": 0.5,
        "grad_clip": 5.0,
        "checkpoint_path": "./checkpoints/ssf4vsu_best.pth",
    }
    main(config)
