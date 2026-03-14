import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# ----------------------------
#  Single Object Tracking (SOT)
# ----------------------------
def compute_sot_metrics(pred_bboxes, gt_bboxes, threshold=20):
    """
    Success (AUC), Precision@20, Normalized Precision.
    """
    ious, precisions, norm_precisions = [], [], []
    for p, g in zip(pred_bboxes, gt_bboxes):
        # IoU
        x1 = max(p[0], g[0])
        y1 = max(p[1], g[1])
        x2 = min(p[0]+p[2], g[0]+g[2])
        y2 = min(p[1]+p[3], g[1]+g[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = p[2]*p[3] + g[2]*g[3] - inter
        iou = inter/union if union > 0 else 0
        ious.append(iou)

        # Precision@20
        center_p = (p[0]+p[2]/2, p[1]+p[3]/2)
        center_g = (g[0]+g[2]/2, g[1]+g[3]/2)
        dist = np.linalg.norm(np.array(center_p)-np.array(center_g))
        precisions.append(dist < threshold)

        # Normalized Precision (relative to GT size)
        norm_thr = np.sqrt(g[2]**2 + g[3]**2)/10
        norm_precisions.append(dist < norm_thr)

    auc = np.mean(ious)
    prec20 = np.mean(precisions)
    norm_prec = np.mean(norm_precisions)
    return {"AUC": auc, "Precision@20": prec20, "NormPrec": norm_prec}


# ----------------------------
#  Multi-Object Tracking (MOT)
# ----------------------------
def compute_mot_metrics(pred_tracks, gt_tracks):
    """
    Computes MOTA, IDF1, MT, ML, FP, FN, ID switches.
    pred_tracks: dict {frame: [(id, bbox), ...]}
    gt_tracks:   dict {frame: [(id, bbox), ...]}
    """
    # For simplicity, we compute counts
    FP, FN, IDs = 0, 0, 0
    matches, total_gt = 0, 0
    id_mapping = {}

    for frame in gt_tracks:
        gt_objs = gt_tracks[frame]
        pred_objs = pred_tracks.get(frame, [])
        total_gt += len(gt_objs)

        matched_ids = []
        for gt_id, gt_box in gt_objs:
            best_iou, best_pid = 0, None
            for pid, pbox in pred_objs:
                # IoU
                x1 = max(pbox[0], gt_box[0])
                y1 = max(pbox[1], gt_box[1])
                x2 = min(pbox[0]+pbox[2], gt_box[0]+gt_box[2])
                y2 = min(pbox[1]+pbox[3], gt_box[1]+gt_box[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                union = pbox[2]*pbox[3] + gt_box[2]*gt_box[3] - inter
                iou = inter/union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_pid = iou, pid

            if best_iou > 0.5:
                matches += 1
                if gt_id in id_mapping and id_mapping[gt_id] != best_pid:
                    IDs += 1
                id_mapping[gt_id] = best_pid
                matched_ids.append(best_pid)
            else:
                FN += 1

        FP += len(pred_objs) - len(matched_ids)

    mota = 1 - (FP + FN + IDs) / max(1, total_gt)
    idf1 = 2 * matches / (total_gt + len(id_mapping)) if total_gt > 0 else 0
    return {"MOTA": mota, "IDF1": idf1, "FP": FP, "FN": FN, "IDs": IDs}


# ----------------------------
#  Video Object Segmentation (VOS)
# ----------------------------
def compute_vos_metrics(pred_masks, gt_masks):
    """
    Jaccard (J), F-measure (F), J&F mean.
    """
    J, F_scores = [], []
    for p, g in zip(pred_masks, gt_masks):
        # IoU (J)
        inter = (p * g).sum()
        union = (p + g).clamp(max=1).sum()
        j = inter / union if union > 0 else 0
        J.append(j.item())

        # Boundary F-measure (F from torch.nn.functional for max_pool2d)
        p_edge = p - F.max_pool2d(p, 3, 1, 1)
        g_edge = g - F.max_pool2d(g, 3, 1, 1)
        tp = (p_edge * g_edge).sum()
        fp = (p_edge * (1-g_edge)).sum()
        fn = ((1-p_edge) * g_edge).sum()
        prec = tp / (tp+fp+1e-6)
        rec = tp / (tp+fn+1e-6)
        f = 2*prec*rec / (prec+rec+1e-6)
        F_scores.append(f.item())

    Jm, Fm = np.mean(J), np.mean(F_scores)
    return {"J": Jm, "F": Fm, "J&F": (Jm+Fm)/2}


# ----------------------------
#  MOTS Evaluation
# ----------------------------
def compute_mots_metrics(pred_masks, gt_masks, pred_ids, gt_ids):
    """
    Computes sMOTSA, MOTSA, MOTSP, IDF1.
    """
    TP, FP, FN = 0, 0, 0
    IoUs = []
    for p, g, pid, gid in zip(pred_masks, gt_masks, pred_ids, gt_ids):
        inter = (p * g).sum().item()
        union = (p + g).clamp(max=1).sum().item()
        iou = inter / union if union > 0 else 0
        if iou > 0.5:
            TP += 1
            IoUs.append(iou)
        else:
            FP += 1
            FN += 1

    motsp = np.mean(IoUs) if IoUs else 0
    motsa = (TP - FP - FN) / max(1, TP+FN)
    smotsa = motsa * motsp
    idf1 = 2*TP / (2*TP + FP + FN) if TP > 0 else 0
    return {"sMOTSA": smotsa, "MOTSA": motsa, "MOTSP": motsp, "IDF1": idf1}
