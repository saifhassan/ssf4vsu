"""
Multi-task and SSL losses. Task-conditional routing: L_mask = 0 for SOT/MOT.
L_total = λ_det L_det + λ_mask L_mask + λ_SSL L_SSL + λ_TCM L_TCM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """L_det = L_bbox + L_cls (Smooth L1 + CE)."""
    def __init__(self):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred_bbox, gt_bbox, pred_logits, gt_labels):
        if pred_bbox.numel() == 0 or gt_bbox.numel() == 0:
            return torch.tensor(0.0, device=pred_bbox.device)
        loss_bbox = self.reg_loss(pred_bbox, gt_bbox)
        if pred_logits.numel() > 0 and gt_labels.numel() > 0 and (gt_labels >= 0).any():
            valid = gt_labels >= 0
            loss_cls = self.cls_loss(pred_logits[valid], gt_labels[valid])
        else:
            loss_cls = torch.tensor(0.0, device=pred_bbox.device)
        return loss_bbox + loss_cls


class SegmentationLoss(nn.Module):
    """L_mask = BCE + Dice."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, gt_mask):
        if pred_mask.numel() == 0 or gt_mask.numel() == 0:
            return torch.tensor(0.0, device=pred_mask.device)
        bce = self.bce(pred_mask, gt_mask)
        smooth = 1e-6
        probs = torch.sigmoid(pred_mask)
        intersection = (probs * gt_mask).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (probs.sum() + gt_mask.sum() + smooth)
        return bce + dice


class SSLLoss(nn.Module):
    """Contrastive SSL (InfoNCE-style). L_SSL for refinement loop."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        if z1 is None or z2 is None or z1.numel() == 0:
            return torch.tensor(0.0, device=z1.device if z1 is not None else torch.device("cpu"))
        B, D = z1.shape
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(B, device=z1.device)
        return self.ce(logits, labels)


class TotalLoss(nn.Module):
    """
    L_total = λ_det L_det + λ_mask L_mask + λ_SSL L_SSL + λ_TCM L_TCM.
    L_mask = 0 for SOT/MOT batches; L_det optional for VOS-only when no boxes.
    """

    def __init__(self, lambda_det=1.0, lambda_mask=1.0, lambda_ssl=0.5, lambda_tcm=0.5):
        super().__init__()
        self.det_loss = DetectionLoss()
        self.seg_loss = SegmentationLoss()
        self.ssl_loss = SSLLoss()
        self.lambda_det = nn.Parameter(torch.tensor(lambda_det))
        self.lambda_mask = nn.Parameter(torch.tensor(lambda_mask))
        self.lambda_ssl = nn.Parameter(torch.tensor(lambda_ssl))
        self.lambda_tcm = nn.Parameter(torch.tensor(lambda_tcm))

    def forward(self, outputs, targets, task_type="SOT", ssl_emb=None, stage=1):
        device = outputs["sot"].device
        loss_det = torch.tensor(0.0, device=device)
        loss_mask = torch.tensor(0.0, device=device)

        # stage=0: SSL refinement loop — only L_SSL + L_TCM (no task losses)
        if stage == 0:
            loss_ssl = torch.tensor(0.0, device=device)
            if ssl_emb is not None:
                z1, z2 = ssl_emb
                if z1 is not None and z2 is not None:
                    if z1.dim() == 3:
                        z1 = z1[:, -1].reshape(z1.size(0), -1)
                        z2 = z2[:, -1].reshape(z2.size(0), -1)
                    loss_ssl = self.ssl_loss(z1, z2)
            loss_tcm = outputs.get("tcm_loss", torch.tensor(0.0, device=device))
            if not isinstance(loss_tcm, torch.Tensor):
                loss_tcm = torch.tensor(loss_tcm, device=device)
            total = self.lambda_ssl * loss_ssl + self.lambda_tcm * loss_tcm
            return {"total": total, "det": loss_det, "mask": loss_mask, "ssl": loss_ssl, "tcm": loss_tcm}

        if task_type in ("SOT", "MOT", "MOTS"):
            bbox = targets.get("bbox")
            labels = targets.get("labels")
            if bbox is not None:
                pred_bbox = outputs["sot"]
                pred_cls = outputs["mot"]
                if pred_bbox.dim() == 4:
                    pred_bbox = pred_bbox.mean(dim=(2, 3))
                bbox_last = bbox[:, -1] if bbox.dim() == 3 else bbox
                labels_last = labels[:, -1] if labels.dim() > 1 else labels
                if pred_bbox.numel() > 0 and bbox_last.numel() > 0:
                    loss_det = self.det_loss(pred_bbox, bbox_last, pred_cls, labels_last)

        if stage >= 2 and task_type in ("VOS", "MOTS"):
            mask = targets.get("mask")
            if mask is not None:
                if outputs["vos"].shape == mask.shape:
                    loss_mask = self.seg_loss(outputs["vos"], mask)
                elif outputs["mots"].numel() > 0:
                    m = mask[:, -1] if mask.dim() == 5 else mask
                    if m.dim() == 3:
                        m = m.unsqueeze(1)
                    loss_mask = self.seg_loss(outputs["mots"][:, :1], m)

        loss_ssl = torch.tensor(0.0, device=device)
        if ssl_emb is not None:
            z1, z2 = ssl_emb
            if z1 is not None and z2 is not None:
                if z1.dim() == 3:
                    z1 = z1[:, -1].reshape(z1.size(0), -1)
                    z2 = z2[:, -1].reshape(z2.size(0), -1)
                loss_ssl = self.ssl_loss(z1, z2)

        loss_tcm = outputs.get("tcm_loss", torch.tensor(0.0, device=device))
        if not isinstance(loss_tcm, torch.Tensor):
            loss_tcm = torch.tensor(loss_tcm, device=device)

        total = (
            self.lambda_det * loss_det
            + self.lambda_mask * loss_mask
            + self.lambda_ssl * loss_ssl
            + self.lambda_tcm * loss_tcm
        )
        return {
            "total": total,
            "det": loss_det,
            "mask": loss_mask,
            "ssl": loss_ssl,
            "tcm": loss_tcm,
        }
