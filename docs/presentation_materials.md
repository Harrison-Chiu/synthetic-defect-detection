# Presentation Materials — S6 Final

> For teammates preparing slides. Report is in English, 10-15 min + 3 min Q&A.

## Key Figures (ready to use)

### Dataset & Defects
| Figure | Path | Description |
|--------|------|-------------|
| Defect comparison | `docs/figures/stage6/param_tuning/s6_defect_comparison_newparam.png` | 2×4 grid: normal/bend/displace/remesh at 2 elevations |
| Defectmask preview | `docs/figures/stage6/param_tuning/s6_defectmask_preview.png` | 3×3 patches with red mask overlay |
| Scale comparison | `docs/figures/stage6/param_tuning/s6_scale_comparison.png` | 9×3 grid showing different part scales |

### Training Results
| Figure | Path | Description |
|--------|------|-------------|
| Training curves | `docs/figures/stage6/training/s6_training_curves.png` | 2×2: loss, per-class IoU, LR, throughput |
| KPI comparison | `docs/figures/stage6/training/s6_kpi_comparison.png` | S3 vs S5 vs S6 bar chart |
| Prediction panel | `docs/figures/stage6/training/s6_pred_panel.png` | RGB/GT/pred/heatmap for each defect type |
| Per-state perf | `docs/figures/stage6/training/s6_per_state.png` | Detection rate + instance IoU boxplot |
| Confusion matrix | `docs/figures/stage6/training/s6_confusion.png` | Row-normalized per-state confusion |

### Full HTML Report
- `output/runs/s6_final/report.html` — comprehensive single-file report with all visualizations

## Key Numbers

| Metric | S3 Baseline | S5 | **S6 Final** |
|--------|-------------|-----|-------------|
| Test mIoU | 0.712 | 0.722 | **0.732** |
| Test defect IoU | 0.362 | 0.390 | **0.443** |
| Pixel accuracy | 94.1% | — | **95.02%** |
| Best val defect IoU | — | 0.390 | **0.468** |

## Slide Outline Suggestion (English)

### 1. Introduction (2 min)
- Task: industrial part defect detection (QC + sorting)
- Challenge: no public dataset for our specific parts → custom Blender synthetic data
- Goal: pixel-level defect segmentation, not just classification

### 2. Dataset (3 min)
- 4 part types from McMaster-Carr CAD models
- 3 defect types: bend (mechanical deformation), displace (surface damage), remesh (topology corruption)
- 2,880 rendered patches (multi-view × multi-lighting × defect states)
- Runtime scene generation with collision detection and domain randomization
- **Use figures**: defect comparison, scale comparison

### 3. Model Architecture (3 min)
- Custom encoder-decoder (not U-Net/YOLO copy)
- Dual-head: semantic segmentation (3-class) + defect heatmap (sigmoid)
- Interior-ignore supervision: W≈0 inside defective parts → learn boundary features
- ~3.3M parameters, trains in ~6 minutes on single GPU

### 4. Results (3 min)
- Defect IoU: 0.362 → 0.443 (+22% over baseline)
- Remesh & displace well detected; bend near noise floor (honest analysis)
- Model capacity saturates at bc=8 → data quality matters more than model size
- **Use figures**: training curves, KPI comparison, prediction panel, confusion matrix

### 5. Conclusion (1 min)
- Custom dataset + custom model + iterative improvement
- Key insight: interior-ignore supervision is crucial for defect boundary detection
- Limitation: bend deformation too subtle for pixel-level detection

## Dataset Statistics
- Patches: 2,880 (720 per state)
- Defectmasks: 45 (for supervision)
- Scene generation: deterministic `f(index, config)` → reproducible
- Train/val/test: disjoint index ranges (no leakage by design)
- Parts per scene: 3-6, scale 50-75%

## Model Details
- Architecture: DefectSegNet (encoder-decoder + skip connections)
- base_c=32, depth=4 (4 downsampling levels)
- Heads: sem3 (3-class CE), defect (binary BCE, pos_weight=8)
- Checkpoint selection: val defect IoU (not mIoU)
- Training: 4000 steps, batch=8, Adam lr=1e-3 → ReduceLROnPlateau
