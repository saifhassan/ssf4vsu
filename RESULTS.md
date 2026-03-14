# Benchmark Results (from thesis)

All results below are from the PhD thesis evaluation of SSF4VSU. Single model, no task-specific tuning; same checkpoint for all tasks. Input resolution: 1280×720 (or 640×360); online inference.

---

## Single-Object Tracking (SOT)

### LaSOT & TrackingNet

| Tracker | LaSOT AUC (%) | LaSOT Prec@20 (%) | TrackingNet AUC (%) | TrackingNet Prec (%) |
|:--------|--------------:|------------------:|--------------------:|---------------------:|
| SiamFC | 33.6 | 33.9 | 57.1 | 66.3 |
| SiamRPN++ | 49.6 | 49.1 | 73.3 | 69.4 |
| TransT | 64.9 | 69.0 | 81.4 | 80.3 |
| Siam R-CNN | 64.8 | – | 81.2 | 80.0 |
| KeepTrack | 67.1 | 70.2 | – | – |
| Unicorn | 68.5 | 74.1 | 83.0 | 82.2 |
| OmniTracker | 65.9 | 69.3 | 80.2 | 77.7 |
| OmniTracker-L | 69.1 | 75.4 | 83.4 | 82.3 |
| **SSF4VSU (Ours)** | **71.3** | **77.8** | **84.5** | **83.0** |

**LaSOT success and precision curves:** [View plot (PDF)](docs/results/lasot_ope_plots.pdf)

---

## Multi-Object Tracking (MOT)

### MOT17 (test set)

| Tracker | MOTA↑ | IDF1↑ | HOTA↑ | MT↑ | ML↓ | FP↓ | FN↓ | IDs↓ |
|:--------|-----:|------:|------:|----:|----:|----:|----:|-----:|
| CenterTrack | 67.8 | 64.7 | 52.2 | 34.6% | 24.6% | 18498 | 160332 | 3039 |
| TraDes | 69.1 | 63.9 | 52.7 | 36.4% | 21.5% | 20892 | 150060 | 3555 |
| ByteTrack | 80.3 | 77.3 | 63.1 | – | – | 25491 | 83721 | 2196 |
| FairMOT | 73.7 | 72.3 | 59.3 | 43.2% | 17.3% | 27507 | 117477 | 3303 |
| TransMOT | 76.7 | 75.1 | 61.7 | 51.0% | 16.4% | 36231 | 93150 | 2346 |
| Unicorn | 77.2 | 75.5 | 61.7 | 58.7% | 11.2% | 50087 | 73349 | 5379 |
| OmniTracker-L | 79.1 | 75.6 | 62.3 | – | – | 87192 | 1968 | – |
| **SSF4VSU (Ours)** | **81.9** | **80.3** | **65.2** | **60.0%** | **12.3%** | **48000** | **72000** | **4520** |

**MOT17 result figures:** [MOTA vs IDF1 tradeoff (PDF)](docs/results/mot17_tradeoff_scatter.pdf) · [MOTA, IDF1, HOTA bars (PDF)](docs/results/mot17_grouped_bars.pdf) · [FP/FN/ID breakdown (PDF)](docs/results/mot17_error_breakdown.pdf)

### BDD100K (validation)

| Tracker | mMOTSA↑ | mIDF1↑ | MOTA↑ | IDF1↑ | FN↓ | FP↓ | IDs↓ | MT↑ | ML↓ |
|:--------|-------:|-------:|------:|------:|----:|----:|-----:|----:|----:|
| Yu et al. | 26.3 | 44.7 | 58.3 | 68.2 | 213220 | 100230 | 14674 | 16299 | 6017 |
| QDTrack | 35.5 | 52.3 | 64.3 | 72.3 | 201041 | 80054 | 10790 | 17353 | 5167 |
| ByteTrack | 40.1 | 55.8 | 69.6 | 71.3 | 169073 | 63869 | 15466 | 18057 | 5107 |
| Unicorn | 41.2 | 54.0 | 66.6 | 71.3 | 95454 | 41648 | 10876 | 10296 | 2505 |
| **SSF4VSU (Ours)** | **43.5** | **57.0** | **71.0** | **74.5** | **90000** | **40000** | **11500** | **11000** | **3048** |

**BDD100K MOT result figures:** [Error composition (PDF)](docs/results/bdd100k_error_composition.pdf) · [MT vs ML (PDF)](docs/results/bdd100k_mt_ml.pdf) · [Grouped metrics (PDF)](docs/results/bdd100k_grouped_metrics.pdf)

---

## Video Object Segmentation (VOS)

### DAVIS-2016 & DAVIS-2017

| Method | DAVIS-2016 J&F | J Mean | F Mean | DAVIS-2017 J&F | J Mean | F Mean |
|:-------|---------------:|-------:|-------:|----------------:|-------:|-------:|
| STM | 89.3 | 88.7 | 89.9 | 81.8 | 79.2 | 84.3 |
| HMMN | 90.8 | 89.6 | 92.0 | 84.7 | 81.9 | 87.5 |
| STCN | 91.6 | 90.8 | 92.5 | 85.4 | 82.2 | 88.6 |
| XMem | 92.0 | 90.7 | 93.2 | 87.7 | 84.0 | 91.4 |
| ISVOS | 92.8 | 91.8 | 93.8 | 88.2 | 84.5 | 91.9 |
| SiamMask | 69.8 | 71.7 | 67.8 | 56.4 | 54.3 | 58.5 |
| Unicorn | 87.4 | 86.5 | 88.2 | 69.2 | 65.2 | 73.2 |
| OmniTracker-L | 88.5 | 87.3 | 89.7 | 71.0 | 66.8 | 75.2 |
| **SSF4VSU (Ours)** | **93.3** | **92.4** | **94.4** | **89.0** | **86.7** | **93.4** |

**DAVIS-2016/2017 normalized radar (J&F, J Mean, F Mean):** [View plot (PDF)](docs/results/davis_radar_top5_norm.pdf)

---

## Multi-Object Tracking and Segmentation (MOTS)

### MOTS20 (pedestrians)

| Method | sMOTSA↑ | MOTSA↑ | MOTSP↑ | IDF1↑ | ID Sw.↓ |
|:-------|--------:|-------:|-------:|------:|--------:|
| TrackRCNN | 40.6 | 55.2 | 76.1 | 42.2 | 567 |
| PointTrack V2 | 62.3 | – | – | 42.9 | 541 |
| TrackFormer | 54.9 | – | – | 63.6 | 278 |
| Unicorn | 65.3 | – | – | 65.9 | 398 |
| **SSF4VSU (Ours)** | **69.0** | **74.5** | **82.0** | **70.5** | **357** |

### BDD100K MOTS (validation)

| Method | mMOTSA↑ | mMOTSP↑ | mIDF1↑ | ID Sw.↓ | mAP↑ |
|:-------|--------:|--------:|-------:|--------:|-----:|
| QDTrack-mots-fix | 23.5 | 66.3 | 44.5 | 973 | 25.5 |
| PCAN | 27.4 | 66.7 | 45.1 | **876** | 26.6 |
| Unicorn | 29.6 | 67.7 | 44.2 | 1731 | 32.1 |
| **SSF4VSU (Ours)** | **31.2** | **70.2** | **46.0** | 1462 | **33.7** |

**BDD100K MOTS result figures:** [Radar (mMOTSA, mMOTSP, mIDF1, mAP) (PDF)](docs/results/bdd100k_mots_radar.pdf) · [ID switches comparison (PDF)](docs/results/bdd100k_mots_idsw_line.pdf)

---

## Ablation Studies

Full model vs. ablated variants (TAM, TCM, SSL removed) on LaSOT, MOT17, DAVIS-2017, MOTS20.

### LaSOT (SOT)

| Model | Success (AUC) | Precision |
|:------|--------------:|----------:|
| **Full SSF4VSU** | **71.3** | **77.8** |
| – TAM | 68.5 | 73.0 |
| – TCM | 67.0 | 72.6 |
| – SSL | 69.7 | 74.5 |
| – TAM & TCM | 65.0 | 70.5 |

### MOT17 (MOT)

| Model | MOTA | IDF1 |
|:------|-----:|-----:|
| **Full SSF4VSU** | **81.9** | **80.3** |
| – TAM | 78.1 | 78.2 |
| – TCM | 76.6 | 75.2 |
| – SSL | 78.9 | 78.8 |
| – TAM & TCM | 73.6 | 70.2 |

### DAVIS-2017 (VOS)

| Model | J&F (%) |
|:------|--------:|
| **Full SSF4VSU** | **89.0** |
| – TAM | 84.9 |
| – TCM | 85.7 |
| – SSL | 87.0 |
| – TAM & TCM | 81.7 |

### MOTS20 (MOTS)

| Model | sMOTSA (%) | IDF1 (%) |
|:------|-----------:|---------:|
| **Full SSF4VSU** | **69.0** | **70.5** |
| – TAM | 67.0 | 67.0 |
| – TCM | 65.0 | 64.0 |
| – SSL | 68.0 | 68.5 |
| – TAM & TCM | 62.0 | 60.0 |

---

## Result figures (summary)

Result plots are linked above next to each dataset table. All files live in [docs/results/](docs/results/): `lasot_ope_plots.pdf`, `mot17_tradeoff_scatter.pdf`, `mot17_grouped_bars.pdf`, `mot17_error_breakdown.pdf`, `bdd100k_error_composition.pdf`, `bdd100k_mt_ml.pdf`, `bdd100k_grouped_metrics.pdf`, `davis_radar_top5_norm.pdf`, `bdd100k_mots_radar.pdf`, `bdd100k_mots_idsw_line.pdf`. Optional ablation figures can be added under [docs/ablation/](docs/ablation/).

---

*Source: PhD thesis Chapter 5 (Results). See paper/thesis for experimental details, citations, and full discussion.*
