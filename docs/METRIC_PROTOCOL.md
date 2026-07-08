# Metric Protocol (BV2)

Evaluation protocol for **published BV2 paper tables**. Training and MG-WSF use the **validation split** for model and fusion selection; **final reported numbers use the test split only**.

---

## Splits

| Split | Samples | Use |
|-------|--------:|-----|
| train | 1911 | Training |
| val | 625 | Validation metrics, checkpoint selection, MG-WSF |
| test | **584** | **Final reporting only** |

---

## Input shapes

| Tensor | Shape |
|--------|-------|
| Echo (runtime STFT) | **2 × 256 × 256** |
| Depth target (model) | **1 × 256 × 256** (normalized by 30 m) |

`max_depth = 30.0 m`

---

## Valid-depth evaluation protocol

Pixels contribute to metrics only when **ground-truth depth ≥ valid_min_depth**:

```
valid_mask = (gt_meters >= 0.5)
```

- `valid_min_depth = 0.5 m`
- Predictions and ground truth are in **meters** before masking (denormalize model output if needed).

This is the standard valid-depth protocol for BV2 paper tables — not a hidden trick or alternate split logic.

---

## Metrics

Six metrics with **sample-mean aggregation**:

| Metric | Definition |
|--------|------------|
| **RMSE** | Root mean square error (meters) |
| **REL** | Mean absolute relative error |
| **log10** | Mean \|log10(gt) − log10(pred)\| |
| **δ1** | Fraction with max(gt/pred, pred/gt) < 1.25 |
| **δ2** | Fraction with max(gt/pred, pred/gt) < 1.25² |
| **δ3** | Fraction with max(gt/pred, pred/gt) < 1.25³ |

**Aggregation:** compute per-image metrics on valid pixels → average over all test samples.

---

## MG-WSF rows (Table 4, BV2)

| Stage | Description |
|-------|-------------|
| Before fusion | Single best checkpoint |
| After MG-WSF | Validation-guided fusion of multiple donors |

Both rows are evaluated on the **test split** with the same valid-depth protocol above.

---

## Reproduction checklist

1. Released pretrained checkpoints ([CHECKPOINTS.md](CHECKPOINTS.md))
2. BV2 data + index ([DATA.md](DATA.md))
3. Test evaluation for final numbers
4. Compare to [`results/paper_results_bv2.csv`](../results/paper_results_bv2.csv)

See [RESULTS_REPRODUCTION.md](RESULTS_REPRODUCTION.md).
