# Training and Checkpoint Fusion with MG-WSF

Training follows the paper configuration. The paper is **coming soon**.

Configs:

- `configs/bv2/train_htf_echodepth_bv2.yaml`
- `configs/bv2/mg_wsf_bv2.yaml`

---

## Train

```bash
python scripts/train_bv2.py \
  --config configs/bv2/train_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --output-dir outputs/bv2_run
```

Training writes validation-selected candidate checkpoints and `candidate_registry.csv` under `--output-dir`.

---

## MG-WSF

```bash
python scripts/run_mg_wsf_bv2.py \
  --config configs/bv2/mg_wsf_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --candidate-registry outputs/bv2_run/candidate_registry.csv \
  --output-dir outputs/bv2_run/fused \
  --save-fused-checkpoint
```

MG-WSF constructs the final fused checkpoint from validation-selected candidates.

---

## Evaluate

```bash
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint outputs/bv2_run/fused/htf_echodepth_mg_wsf_fused.pth
```

See also [CHECKPOINTS.md](CHECKPOINTS.md) and [RESULTS_REPRODUCTION.md](RESULTS_REPRODUCTION.md).
