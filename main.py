"""
SSF4VSU entry point: train (two-stage + fine-tune) or eval (per-task metrics).
"""
import argparse
import torch

from model import SSF4VSU
from train import main as train_main
from evaluate import compute_sot_metrics, compute_mot_metrics, compute_vos_metrics, compute_mots_metrics
from datasets import MultiTaskDataset, DEFAULT_RESOLUTION
from utils import load_checkpoint


def run_evaluation(model, dataset, device, task="SOT"):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            if frames.dim() == 4:
                frames = frames.unsqueeze(0)
            target_prior = batch.get("target_prior")
            if target_prior is not None:
                target_prior = target_prior.to(device)
                if target_prior.dim() == 4:
                    target_prior = target_prior.unsqueeze(0)
            outputs = model(frames, target_prior=target_prior)

            if task == "SOT":
                pred_bbox = outputs["sot"].cpu().numpy()[0]
                if pred_bbox.ndim == 4:
                    pred_bbox = pred_bbox[0, :, 0, 0]
                gt_bbox = batch["bbox"][0, -1].numpy()
                all_preds.append(pred_bbox)
                all_gts.append(gt_bbox)
            elif task == "MOT":
                preds = [(i, b.tolist()) for i, b in enumerate(outputs["mot"].cpu().numpy())]
                gts = [(int(lbl), bb.tolist()) for lbl, bb in zip(batch["labels"][0], batch["bbox"][0])]
                all_preds.append(preds)
                all_gts.append(gts)
            elif task == "VOS":
                all_preds.append(outputs["vos"].cpu())
                all_gts.append(batch["mask"])
            elif task == "MOTS":
                all_preds.append(outputs["mots"].cpu())
                all_gts.append(batch["mask"])

    if task == "SOT":
        return compute_sot_metrics(all_preds, all_gts)
    elif task == "MOT":
        return compute_mot_metrics(dict(enumerate(all_preds)), dict(enumerate(all_gts)))
    elif task == "VOS":
        return compute_vos_metrics(all_preds, all_gts)
    elif task == "MOTS":
        pred_masks = list(all_preds)
        gt_masks = list(all_gts)
        pred_ids = list(range(len(pred_masks)))
        gt_ids = list(range(len(gt_masks)))
        return compute_mots_metrics(pred_masks, gt_masks, pred_ids, gt_ids)
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSF4VSU Framework")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"])
    parser.add_argument("--task", type=str, default="SOT", choices=["SOT", "MOT", "VOS", "MOTS"])
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/ssf4vsu_best.pth")
    parser.add_argument("--data", type=str, default="./data/val/")
    parser.add_argument("--resolution", type=str, default="640,360", help="H,W e.g. 640,360 or 1280,720")
    parser.add_argument("--roots_by_task", type=str, default=None, help="comma-separated SOT:path,MOT:path,... for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = tuple(map(int, args.resolution.split(",")))

    if args.mode == "train":
        roots_by_task = None
        if args.roots_by_task:
            roots_by_task = {}
            for part in args.roots_by_task.split(","):
                k, v = part.split(":", 1)
                roots_by_task[k.strip()] = v.strip()
        config = {
            "train_data": "./data/train/",
            "val_data": "./data/val/",
            "roots_by_task": roots_by_task,
            "backbone": "resnet50",
            "embed_dim": 256,
            "num_classes": 100,
            "resolution": res,
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
            "seq_len": 8,
            "ssl": True,
            "checkpoint_path": args.checkpoint,
        }
        train_main(config)

    elif args.mode == "eval":
        dataset = MultiTaskDataset(
            args.data,
            task_type=args.task,
            augment=False,
            seq_len=8,
            ssl=False,
            resolution=res,
        )
        model = SSF4VSU(backbone_type="resnet50", embed_dim=256, num_classes=100)
        load_checkpoint(model, None, args.checkpoint, device)
        model.to(device)
        metrics = run_evaluation(model, dataset, device, task=args.task)
        print(f"✅ {args.task} Evaluation Results:", metrics)
