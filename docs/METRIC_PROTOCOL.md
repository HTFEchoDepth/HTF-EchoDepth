# Metric Protocol (BV2)

Evaluation protocol for **BV2 paper tables**. Training and MG-WSF use the validation split for model and fusion selection; reported numbers are computed on the BV2 test split.

---

## Splits

| Split | Samples | Use |
|-------|--------:|-----|
| train | 1911 | Training |
| val | 625 | Validation metrics, checkpoint selection, MG-WSF |
| test | **584** | Reporting |

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

Metrics are computed by the evaluation utilities in `htf_echodepth.metrics` and aggregated over the BV2 test split following the paper setting.

---

## MG-WSF rows (Table 4, BV2)

| Stage | Description |
|-------|-------------|
| Before fusion | Single best checkpoint |
| After MG-WSF | Validation-guided fusion of multiple donors |

Both rows are evaluated with the same BV2 paper setting.

---

## Reproduction checklist

1. Released pretrained checkpoints ([CHECKPOINTS.md](CHECKPOINTS.md))
2. BV2 data + index ([DATA.md](DATA.md))
3. Test evaluation for final numbers
4. Compare to [`results/paper_results_bv2.csv`](../results/paper_results_bv2.csv)

See [RESULTS_REPRODUCTION.md](RESULTS_REPRODUCTION.md).
