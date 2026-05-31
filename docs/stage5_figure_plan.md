# Stage 5 Figure Plan — Final (post-discussion)

> Last updated: 2026-06-01
> Status: **design finalized**, pending script update + generation.
> Script: `scripts/gen_stage5_figures.py` (needs rewrite to match this plan)
> Output: `docs/figures/stage5/`
> Language: **all English** (no CJK — avoids matplotlib tofu)

---

## Overview

15–16 figures total. All defect metrics at `thr=0.7` unless noted.

| # | Filename | Category | Purpose |
|---|---|---|---|
| 00 | dataset_showcase | 1. Dataset | Show synthetic data diversity |
| 01 | supervision_TW | 2. Method | Visualize interior-ignore T/W |
| 02 | diff_threshold | 2. Method | How defect masks are generated |
| 03 | kpi_vs_s3 | 4. Quantitative | Headline improvement over S3 |
| 04+05 | sweep_capacity | 6. Ablation | Width + depth sweep (one figure, two subplots) |
| 06 | thr_sweep | 7. Error analysis | defect_thr trade-off |
| 07 | posweight_sweep | 6. Ablation | pos_weight sweep |
| 08 | training_curves | 3. Training dynamics | Loss / IoU / LR / loss components |
| 09 | per_type | 4+7. Quant + Error | Per-defect-type with delta |
| 10 | pred_panel | 5. Qualitative | Strategic sample selection |
| 11 | fp_fn_cases | 5. Qualitative | Representative FP and FN |
| 12 | confusion | 4. Quantitative | 5-row confusion matrix |
| 14 | bend_honesty | 7. Error analysis | Bend vs FP across thresholds |
| 15a | pr_curve | 4. Quantitative | Overall precision-recall |
| 15b | pr_curve_pertype | 4+7. Quant + Error | Per-type precision-recall |

Dropped: 13 (inst_iou_box, redundant with 09), sorting_panel (part head doesn't distinguish part types → overclaim risk; multi-part ability shown via pred_panel instead).

---

## Detailed Design

### Fig 00 — Dataset Showcase

**Purpose**: Show the Blender synthetic dataset — the project's biggest selling point — in one figure.

**Layout**: 3 stacked sub-figures, all sharing column width.

**Sub-A "Defect States"** (2R × 5C):
- Column axis = defect state: normal / remesh / displace / bend_heavy / bend_light
- Row 1 = patch (single-part crop from render, white/transparent bg)
- Row 2 = scene (same defect state placed into a full multi-part scene)
- Vertical alignment: reader looks down to see "this patch → in context"

**Sub-B "Domain Randomization"** (1R × 4C):
- Column axis = background type: HDRI env A / HDRI env B / Procedural / Solid
- Same scene composition, only background changes
- Shows lighting and background variety

**Sub-C "Part Types"** (1R × 4C):
- Column axis = part type: socket_head / pan_head / hex_nut / flange_nut
- One normal part each, close-up crop

**Annotations**: Sub-titles for each sub-figure. No per-cell captions needed if column headers are clear.

**Data source**: runtime generation (specific scene indices chosen for variety).

---

### Fig 01 — Supervision T/W

**Purpose**: Visualize the core S5 mechanism — interior-ignore via weight map.

**Layout**: 3R × 4C grid.
- Row axis = one scene each (strategically selected: R1=remesh, R2=bend, R3=displace)
- Column axis:
  - C1: RGB scene (original)
  - C2: Semantic GT (bg=black, normal=green, defect=red — entire defective part is red)
  - C3: Target T (deformation region=bright, hot colormap 0→black 1→yellow)
  - C4: Weight W (Gaussian falloff, viridis colormap 0→dark purple high→yellow)

**Key annotation**: Arrow on C4 pointing to dark interior of defective part + label "W≈0: interior ignored". This is the visual punchline of S5.

**Contrast**: C2 (entire part red) vs C3 (only deformation region bright) — reader sees the difference between old supervision (S3) and new (S5).

---

### Fig 02 — Diff Threshold

**Purpose**: Show how defect masks are generated (normal vs defect patch subtraction).

**Reuse**: Existing `docs/figures/route_a_diff_threshold.png`. No changes needed.

---

### Fig 03 — KPI vs S3 Baseline

**Purpose**: One chart summarizing S5 improvement.

**Layout**: Grouped bar chart, 3 groups side by side.
- X axis = metric name: mIoU / defect IoU / remesh inst IoU
- Each group = 2 bars: gray (S3) + blue (S5)
- Y axis = score, range 0–0.8
- Bar top labels: exact values (e.g. "0.390")
- Annotation arrow on defect IoU group: "+0.028" showing improvement magnitude

**Subtitle**: "Test set, thr=0.7"

---

### Fig 04+05 — Capacity Sweep (combined, 2 subplots)

**Purpose**: Show (1) width saturation → over-parameterized, (2) depth > width.

**Layout**: 1R × 2C, two subplots sharing visual style (same Y axis range, same font).

**Left subplot (04): Width sweep**
- Line chart. X axis = base_c (2,4,6,8,16,24,32). Y axis = defect IoU.
- Each point annotated with param count ("13K", "52K", ... "3.31M")
- Vertical dashed line at bc=8 + label "saturation"
- Gray horizontal band across bc8–bc32 range (~0.287–0.299) to emphasize flatness
- **Corner note**: "thr=0.5, pw=8, 3000 steps" (different from headline)

**Right subplot (05): Depth sweep**
- Bar chart. 3 bars: depth=2 / 3 / 4.
- Bar top labels: defect IoU + param count (e.g. "0.324\n52K")
- Horizontal dashed line at IoU=0.177, label "bc=4 (same 52K params)" → visual comparison depth=3 (52K, 0.324) vs bc=4 (52K, 0.177)

**Shared title**: "Model capacity: width saturates at bc=8; depth more efficient than width"

---

### Fig 06 — Threshold Sweep

**Purpose**: Show defect_thr trade-off (raise thr → suppress FP but defect IoU also changes).

**Layout**: Dual Y-axis line chart.
- X axis = thr (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
- Left Y axis = defect IoU (blue line with circle markers)
- Right Y axis = normal FP rate (red line with square markers)
- Vertical dashed line at thr=0.7 + label "selected"
- Both lines labeled with values at thr=0.7

---

### Fig 07 — Pos_weight Sweep

**Purpose**: Show pw=5 is optimal.

**Layout**: Bar chart, 3 bars (pw=3 / 5 / 8).
- pw=5 bar in dark blue, others in light gray
- Bar top labels: defect IoU value
- Subtitle: "thr=0.7, bc=8, 3000 steps"

---

### Fig 08 — Training Curves (enhanced) ⭐

**Purpose**: Show (1) convergence, (2) no overfitting, (3) LR schedule, (4) loss decomposition. Highest information density figure.

**Layout**: 2R × 2C grid, all 4 subplots sharing X axis (step, 0–15000) — vertically aligned so reader can correlate events across subplots.

| Position | Subplot | Y axis | Lines |
|---|---|---|---|
| Top-left | Loss | L_total | Train (yellow) / Val (purple). Annotations: (a) "plateau ~6000 steps", (b) "val ≤ train: BN eval mode" |
| Top-right | Val IoU per class | IoU | bg (gray dashed) / normal (green dashed) / defect (red solid). Defect line has dot at peak "@6800: 0.408" |
| Bottom-left | Learning rate | LR (log scale) | Green solid, staircase shape. Step numbers at each drop |
| Bottom-right | Loss components | loss value | L_part_CE (orange) / L_defect_BCE (blue) / L_defect_Dice (cyan). Shows which head dominates total loss |

**Note on mIoU**: Not plotted (it's just the mean of the 3 class lines — reader can infer it; removing it reduces visual clutter).

**Future**: If S6 adds per-type eval to training loop, bottom-right can be replaced with per-type det_rate curves (bend/displace/remesh vs step).

---

### Fig 09 — Per-Type Performance

**Purpose**: Show difficulty gap across defect types; honestly expose bend inflation.

**Layout**: Horizontal grouped bar chart.
- Y axis = 3 defect types: remesh_heavy / displace_heavy / bend_heavy (top to bottom)
- X axis = percentage (0–100%)
- Each type has 3 bars stacked vertically:
  - Dark blue: det_rate
  - Light red: normal FP rate (same value 6.2% for all — the baseline)
  - Dark green: delta (det − FP) = real discriminative ability
- Bar right-end labels with values

**Key annotation**: Bend's delta bar is very short (+1.2%), with label "≈ noise floor".

---

### Fig 10 — Prediction Panel (strategic selection)

**Purpose**: Qualitative results showing model performance on representative scenes.

**Layout**: 4R × 4C grid.
- Column axis: C1=RGB / C2=GT (3-class mask, bg=black normal=green defect=red) / C3=Pred (same colormap) / C4=P(defect) heatmap (jet, 0→blue 1→red)
- Row axis (strategically selected from test cache):
  - R1: remesh success — scene with remesh defect, correctly detected
  - R2: displace success — scene with displace defect, correctly detected
  - R3: bend miss — scene with bend defect, model fails to detect (typical FN)
  - R4: all-normal scene — no defects, model doesn't false-alarm. Choose a multi-part scene to also demonstrate instance separation ability.

**Selection logic**: Filter test cache by defect type → sort by defect IoU → pick best (R1,R2) / worst (R3) / clean (R4).

**Row labels**: Left side of each row: "remesh (success)", "displace (success)", "bend (miss)", "normal (no false alarm)".

---

### Fig 11 — FP/FN Cases

**Purpose**: Representative false positive and false negative examples.

**Layout**: 2R × 3C.

| | C1: RGB | C2: GT overlay | C3: Pred overlay |
|---|---|---|---|
| R1: FP case | All-normal scene | Green translucent on normal regions | Red translucent on defect predictions → red on good parts = false alarm |
| R2: FN case | Scene with defect | Red translucent on defect GT | Green overlay → defective part shown as normal = miss |

**Selection**: R1 = scene with highest normal FP rate. R2 = bend miss (bend is the primary FN source).

---

### Fig 12 — Confusion Matrix

**Purpose**: Overall prediction error distribution.

**Layout**: Heatmap, 5R × 3C, row-normalized (each row sums to 1).
- Row axis (GT state, 5 rows): normal / bend_light / bend_heavy / displace_heavy / remesh_heavy
- Column axis (predicted class, 3 columns): bg / normal_part / defective_part
- Cell values: percentage (e.g. "28.6%")
- Colormap: Blues (dark diagonal = good)

**Why 5 rows**: Separating bend_light and bend_heavy reveals whether light vs heavy bend differ in detectability.

---

### Fig 14 — Bend Honesty ⭐

**Purpose**: Proactively expose the biggest weakness = bonus points from professor.

**Layout**: Dual Y-axis line chart.
- X axis = thr (0.5 / 0.6 / 0.7), 3 points
- Left Y axis (%): 3 lines
  - Red solid: bend_heavy det_rate
  - Red dashed: bend_light det_rate
  - Gray solid: normal det_rate (= FP baseline)
- Each thr point annotated with delta value (bend_heavy − normal)

**Key annotations**:
- At thr=0.5: arrow → "28.6% ← inflated by FP (normal=22.2%)"
- At thr=0.7: "7.4% ≈ FP floor (normal=6.2%), delta=1.2%"
- Bottom caption: "Bend detection rate tracks normal FP rate — no real discriminative power for bend deformation."

**Visual punchline**: Three lines descending roughly in parallel → bend "detection" is just FP spillover.

---

### Fig 15a — PR Curve (overall)

**Purpose**: Standard precision-recall curve for defect detection.

**Layout**: Single line chart.
- X axis = recall (0–1), Y axis = precision (0–1)
- Blue line: sweep sigmoid threshold 0.01–0.99, compute defect-class precision & recall at each
- Label AUC value
- Large dot at thr=0.7: "selected operating point (P=xx%, R=xx%)"
- Smaller dot at thr=0.5 for comparison

---

### Fig 15b — PR Curve (per-type)

**Purpose**: Show how detection capability differs by defect type.

**Layout**: Single chart, 3 lines.
- X axis = recall (0–1), Y axis = precision (0–1)
- 3 lines: remesh (blue) / displace (orange) / bend (red)
- Legend in corner
- Bend's curve will hug the axes (very low precision at any recall) — this is expected and honest

**Note**: Per-type PR requires per-instance matching (a defective part's pixels are the "positive set" for that type). The script needs to implement this carefully — global defect-class PR ≠ per-type PR.

---

## Generation Plan

1. Update `scripts/gen_stage5_figures.py` to match this spec
2. Run once: `python scripts/gen_stage5_figures.py` → all figures to `docs/figures/stage5/`
3. Visual review with Harrison
4. If S6 per-type eval completes, update Fig 08 bottom-right subplot
