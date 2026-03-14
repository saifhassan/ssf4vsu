"""
SSF4VSU Model: Aligned with thesis 05Method.tex
- Shared backbone + FPN (P3, P4, P5, P6)
- Unified embedding: U = F_cur + alpha * P (target prior, broadcast addition)
- TAM (Temporal Attention Module), TCM (Temporal Consistency Module)
- FAM (Feature Aggregation Module) fuses TAM output with backbone/FPN features
- Unified heads for SOT, MOT, VOS, MOTS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet normalization (thesis: zero mean, unit variance)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------
#  Backbone (shared CNN)
# ----------------------------
class Backbone(nn.Module):
    """
    Shared backbone B(·). Extracts multi-level features for FPN.
    ResNet stages -> C2, C3, C4, C5 (256, 512, 1024, 2048 channels).
    """
    def __init__(self, backbone_type="resnet50", pretrained=True):
        super().__init__()
        if backbone_type == "resnet50":
            from torchvision.models import resnet50
            net = resnet50(pretrained=pretrained)
            self.stage1 = nn.Sequential(*list(net.children())[:5])  # conv1..layer1 -> C2
            self.stage2 = net.layer2   # C3
            self.stage3 = net.layer3   # C4
            self.stage4 = net.layer4   # C5
            self.out_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

    def forward(self, x):
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return [c2, c3, c4, c5]


# ----------------------------
#  Feature Pyramid Network (FPN)
# ----------------------------
class FPN(nn.Module):
    """
    FPN: top-down + lateral connections. Produces P3, P4, P5, P6 (all 256-d).
    Thesis: "FPN generates feature maps at levels P3, P4, P5, P6."
    """
    def __init__(self, in_dims=(256, 512, 1024, 2048), out_dim=256):
        super().__init__()
        self.lateral_c2 = nn.Conv2d(in_dims[0], out_dim, 1)
        self.lateral_c3 = nn.Conv2d(in_dims[1], out_dim, 1)
        self.lateral_c4 = nn.Conv2d(in_dims[2], out_dim, 1)
        self.lateral_c5 = nn.Conv2d(in_dims[3], out_dim, 1)
        self.p6 = nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1)
        self.out_dim = out_dim

    def forward(self, features):
        c2, c3, c4, c5 = features
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p6 = self.p6(p5)
        return p3, p4, p5, p6


# ----------------------------
#  Unified Embedding (thesis: U = F_cur + alpha * P)
# ----------------------------
class UnifiedEmbedding(nn.Module):
    """
    Unified embedding: U_{ijc} = F_cur_{ijc} + alpha * P_{ij}.
    Target prior P is broadcast-added to current frame features.
    """
    def __init__(self, in_dim=256, embed_dim=256, alpha=1.0):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, embed_dim, kernel_size=1)
        self.alpha = alpha

    def forward(self, F_cur, target_prior=None):
        """
        F_cur: [B, C, H, W], target_prior: [B, 1, H_in, W_in] or [B, 1, H, W] or None
        """
        x = self.proj(F_cur)
        if target_prior is not None:
            if target_prior.shape[2:] != x.shape[2:]:
                P = F.interpolate(target_prior, size=x.shape[2:], mode="bilinear", align_corners=False)
            else:
                P = target_prior
            x = x + self.alpha * P
        return x


# ----------------------------
#  Temporal Attention Module (TAM)
# ----------------------------
class TemporalAttentionModule(nn.Module):
    """
    TAM: Q from current, K/V from reference. Attention = softmax(QK^T/sqrt(d)) V.
    """
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, F_cur, F_ref):
        B, C, H, W = F_cur.shape
        cur = F_cur.flatten(2).permute(0, 2, 1)
        ref = F_ref.flatten(2).permute(0, 2, 1)
        Q = self.q_proj(cur)
        K = self.k_proj(ref)
        V = self.v_proj(ref)
        def split_heads(x):
            B, N, C = x.shape
            return x.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out).permute(0, 2, 1).view(B, C, H, W)
        return out + F_cur


# ----------------------------
#  Temporal Consistency Module (TCM)
# ----------------------------
class TemporalConsistencyModule(nn.Module):
    """
    TCM: L_TCM = (1/N) sum_i ||z_t^(i) - z_{t-1}^(i)||^2 on embeddings.
    Implemented as feature-level L2 between consecutive frames.
    """
    def __init__(self):
        super().__init__()

    def forward(self, F_t, F_next):
        return (F_t - F_next).pow(2).mean()


# ----------------------------
#  Feature Aggregation Module (FAM)
# ----------------------------
class FeatureAggregationModule(nn.Module):
    """
    FAM: fuses TAM output with backbone/FPN features (and SSL cues).
    Produces final refined feature for unified heads.
    """
    def __init__(self, embed_dim=256, fpn_dim=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim + fpn_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, tam_out, fpn_feat):
        if tam_out.shape[2:] != fpn_feat.shape[2:]:
            fpn_feat = F.interpolate(fpn_feat, size=tam_out.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([tam_out, fpn_feat], dim=1)
        return self.fuse(x) + tam_out


# ----------------------------
#  Task-specific Heads
# ----------------------------
class TaskHeads(nn.Module):
    """
    Unified heads: detection (SOT/MOT), segmentation (VOS/MOTS).
    """
    def __init__(self, embed_dim=256, num_classes=100):
        super().__init__()
        self.det_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 4, kernel_size=1),
        )
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
        )
        self.mots_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, 1),
        )

    def forward(self, feat_map, pooled_feat):
        sot_out = self.det_head(feat_map)
        mot_out = self.cls_head(pooled_feat)
        vos_out = self.seg_head(feat_map)
        mots_out = self.mots_head(feat_map)
        return {"sot": sot_out, "mot": mot_out, "vos": vos_out, "mots": mots_out}


# ----------------------------
#  Full SSF4VSU Model
# ----------------------------
class SSF4VSU(nn.Module):
    """
    SSF4VSU: Shared backbone + FPN -> Unified embedding (with target prior)
    -> TAM -> FAM -> Unified heads. TCM as loss.
    """

    def __init__(self, backbone_type="resnet50", embed_dim=256, num_classes=100, prior_alpha=1.0):
        super().__init__()
        self.backbone = Backbone(backbone_type)
        self.fpn = FPN(self.backbone.out_dims, out_dim=256)
        self.embedding = UnifiedEmbedding(256, embed_dim, alpha=prior_alpha)
        self.tam = TemporalAttentionModule(embed_dim)
        self.tcm = TemporalConsistencyModule()
        self.fam = FeatureAggregationModule(embed_dim, fpn_dim=256)
        self.heads = TaskHeads(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def _forward_stream(self, x, target_prior=None):
        feats = self.backbone(x)
        p3, p4, p5, p6 = self.fpn(feats)
        U = self.embedding(p5, target_prior)
        return U, p5

    def forward(self, x_seq, target_prior=None):
        B, T, C, H, W = x_seq.shape
        if target_prior is not None and target_prior.dim() == 4:
            target_prior = target_prior[:, None, ...].expand(-1, T, -1, -1, -1)
        feat_seq = []
        fpn_seq = []
        for t in range(T):
            pt = target_prior[:, t] if target_prior is not None else None
            U_t, p5_t = self._forward_stream(x_seq[:, t], pt)
            feat_seq.append(U_t)
            fpn_seq.append(p5_t)
        if T > 1:
            attn_feat = self.tam(feat_seq[-1], feat_seq[-2])
        else:
            attn_feat = feat_seq[-1]
        F_prime = self.fam(attn_feat, fpn_seq[-1])
        tcm_loss = torch.tensor(0.0, device=x_seq.device)
        for t in range(T - 1):
            tcm_loss = tcm_loss + self.tcm(feat_seq[t], feat_seq[t + 1])
        if T > 1:
            tcm_loss = tcm_loss / (T - 1)
        pooled = F.adaptive_avg_pool2d(F_prime, (1, 1)).flatten(1)
        outputs = self.heads(F_prime, pooled)
        outputs["tcm_loss"] = tcm_loss
        outputs["pooled"] = pooled
        return outputs
