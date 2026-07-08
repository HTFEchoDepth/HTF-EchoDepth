# Metric Protocol (BV2)

Metric definitions and aggregation details for BV2 testing.

---

## Splits

| Split | Samples | Use |
|-------|--------:|-----|
| train | 1911 | Development |
| val | 625 | Validation |
| test | **584** | Testing |

---

## Input shapes

| Tensor | Shape |
|--------|-------|
| Echo (runtime STFT) | **2 × 256 × 256** |
| Depth target (model) | **1 × 256 × 256** |

Predictions and ground truth are measured in meters before metric computation.

---

## Metrics

Six metrics are reported:

| Metric | Definition |
|--------|------------|
| **RMSE** | Root mean square error (meters) |
| **REL** | Mean absolute relative error |
| **log10** | Mean \|log10(gt) − log10(pred)\| |
| **δ1** | Fraction with max(gt/pred, pred/gt) < 1.25 |
| **δ2** | Fraction with max(gt/pred, pred/gt) < 1.25² |
| **δ3** | Fraction with max(gt/pred, pred/gt) < 1.25³ |

For RMSE, REL, and log10, lower values are better. For δ1, δ2, and δ3, higher values are better.

Metrics are computed by the evaluation utilities in `htf_echodepth.metrics` and aggregated over the BV2 test split.

---

## Testing checklist

1. Local checkpoint ([CHECKPOINTS.md](CHECKPOINTS.md))
2. BV2 data + index ([DATA.md](DATA.md))
3. Testing command from `scripts/eval_bv2.py`
4. Compare to [`results/paper_results_bv2.csv`](../results/paper_results_bv2.csv)
