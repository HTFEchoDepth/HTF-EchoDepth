# Results Reproduction (BV2)

Two workflows for reproducing published BV2 numbers. **Exact Table 1 reproduction should use released pretrained checkpoints** (Workflow A).

References: [`results/paper_results_bv2.csv`](../results/paper_results_bv2.csv) · [METRIC_PROTOCOL.md](METRIC_PROTOCOL.md) · [CHECKPOINTS.md](CHECKPOINTS.md)

Baseline rows marked *reported* are reference numbers from prior literature and are **not** re-evaluated in this repository.

---

## A. Evaluate a pretrained checkpoint

Recommended for matching Table 1.

1. **Prepare BV2 data** — [DATA.md](DATA.md)
2. **Build index**
   ```bash
   python scripts/build_bv2_index.py \
     --raw-root "${HTF_BV2_DATA_ROOT}" \
     --out-dir data/bv2_index
   ```
3. **Validate samples**
   ```bash
   python scripts/validate_bv2_data.py \
     --data-root "${HTF_BV2_DATA_ROOT}" \
     --index-file data/bv2_index/test_index.csv \
     --num-samples 3
   ```
4. **Place checkpoint** (outside Git):
   ```
   checkpoints/htf_echodepth_bv2_mgwsf.pth
   ```
5. **Run test evaluation**
   ```bash
   python scripts/eval_bv2.py \
     --config configs/bv2/eval_htf_echodepth_bv2.yaml \
     --data-root "${HTF_BV2_DATA_ROOT}" \
     --index-file data/bv2_index/test_index.csv \
     --checkpoint checkpoints/htf_echodepth_bv2_mgwsf.pth \
     --output-dir outputs/eval_pretrained
   ```
6. **Compare** to Table 1 / CSV (allow minor rounding).

---

## B. Train and construct the final fused model

Full pipeline: train → validation metric logging → metric-role candidates → MG-WSF → evaluate fused checkpoint.

### B1. Train

```bash
python scripts/train_bv2.py \
  --config configs/bv2/train_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --output-dir outputs/bv2_run
```

Training outputs (under `--output-dir`):

- `val_metrics.csv` — per-epoch validation metrics
- `candidate_registry.csv` — portable candidate list (**validation only**, no test metrics)
- Metric-role checkpoints: `best_rmse.pth`, `best_rel.pth`, `best_log10.pth`, `best_delta1.pth`, `best_delta2.pth`, `best_delta3.pth`, `best_wbrs.pth`
- Optional: `last.pth`

Details: [TRAINING_RECIPE.md](TRAINING_RECIPE.md)

### B2. MG-WSF fusion

MG-WSF is part of the **training and model fusion** workflow — not a separate experimental track. It:

1. Reads `candidate_registry.csv`
2. Applies **MRCR** on validation records
3. Runs **VGFS** (validation-guided weight search) on the **validation split**
4. Performs **weight-space fusion** → **single fused checkpoint**

```bash
python scripts/run_mg_wsf_bv2.py \
  --config configs/bv2/mg_wsf_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --candidate-registry outputs/bv2_run/candidate_registry.csv \
  --output-dir outputs/bv2_run/fused \
  --save-fused-checkpoint
```

**Requirements:**

- **≥ 2 distinct validation-selected donors** after MRCR (from a full training run with diverse metric-role improvements).
- `--allow-degenerate-fusion` is **only for smoke/debug** when MRCR collapses to a single donor. It performs copy-through tagging, **not** real fusion — **do not treat as performance evidence**.

### B3. Evaluate fused checkpoint (test split — final reporting only)

```bash
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint outputs/bv2_run/fused/htf_echodepth_mg_wsf_fused.pth \
  --output-dir outputs/eval_fused
```

---

## MG-WSF implementation notes

### Donor pool size

MRCR typically retains a small specialist pool (paper-scale: on the order of a few donors). The release config does not hard-limit donor count; donors come from `candidate_registry.csv`.

### Weight grid for 4+ donors

VGFS uses a full simplex grid for **k ≤ 3** donors. For **k > 3**, the release implementation uses a **simple equal-weight fallback** rather than an exhaustive grid search. This is an engineering simplification — **not** a performance claim. For exact paper numbers, use the **released pretrained fused checkpoint** (Workflow A).

---

## Expected tolerances

| Comparison | Expectation |
|------------|-------------|
| Released checkpoint + protocol | Rounded match on all six metrics |
| Train + fuse from scratch | Small drift possible |
| Smoke / single-donor fusion | Engineering test only |
| Baselines marked *reported* | Not re-evaluated here |

---

## Out of scope

- Full internal experiment matrix
- Qualitative figure assets (optional separate release)
