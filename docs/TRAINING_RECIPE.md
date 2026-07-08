# Training and Checkpoint Fusion with MG-WSF

Paper-aligned **BatVision V2** training for HTF-EchoDepth, followed by validation-guided **MG-WSF** to produce the final fused checkpoint.

Config: `configs/bv2/train_htf_echodepth_bv2.yaml` · Fusion config: `configs/bv2/mg_wsf_bv2.yaml`

---

## Paper-style training settings

| Item | Setting |
|------|---------|
| Epochs | 80 |
| Seed | 42 |
| Batch size | 32 |
| Optimizer | AdamW |
| Weight decay | 1×10⁻² |
| Learning rate | 3×10⁻⁴ |
| LR schedule | 3-epoch warmup + cosine decay to 5×10⁻⁵ |
| Loss | L1 + thresholded log-ratio hinge |
| λ_lrh | 0.003 |
| Input | 2×256×256 runtime STFT spectrogram |
| Depth target | Normalized depth map |
| Checkpoint selection | **Validation split** (metric-role bests + WBRS) |
| Reporting | **Test split** |

```bash
python scripts/train_bv2.py \
  --config configs/bv2/train_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --output-dir outputs/bv2_run
```

## Training output structure

After a full run, `--output-dir` (example: `outputs/bv2_run/`) contains:

```
outputs/bv2_run/
├── best_rmse.pth
├── best_rel.pth
├── best_log10.pth
├── best_delta1.pth
├── best_delta2.pth
├── best_delta3.pth
├── best_wbrs.pth
├── last.pth                      # if save_last enabled
├── val_metrics.csv
├── candidate_registry.csv
└── fused/                        # created by run_mg_wsf_bv2.py
    ├── htf_echodepth_mg_wsf_fused.pth
    └── fusion_result.json
```

### candidate_registry.csv

Generated from validation metrics. Columns include `role`, `metric_name`, `checkpoint_path` (relative), `epoch`, and validation metric values.

### Metric-role checkpoints

Each `best_*.pth` is saved when its validation metric improves. These files are the donor pool for MG-WSF (via MRCR).

---

## MG-WSF (post-training)

MG-WSF constructs the **final fused checkpoint** after training:

```bash
python scripts/run_mg_wsf_bv2.py \
  --config configs/bv2/mg_wsf_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --candidate-registry outputs/bv2_run/candidate_registry.csv \
  --output-dir outputs/bv2_run/fused \
  --save-fused-checkpoint
```

Pipeline: **MRCR** (retain validation specialists) → **VGFS** (validation weight search) → **weight-space fusion** → single `.pth`.

- Uses validation-selected MRCR donors for weight-space fusion.

Evaluate the fused model on the **test split** ([METRIC_PROTOCOL.md](METRIC_PROTOCOL.md)).

---

## Train from scratch vs pretrained

Stochastic training and checkpoint tie-breaking may cause small drift vs `results/paper_results_bv2.csv`. Released pretrained checkpoints are described in [CHECKPOINTS.md](CHECKPOINTS.md).

---

## Entry points

| Script | Purpose |
|--------|---------|
| `scripts/train_bv2.py` | BV2 training + candidate retention |
| `scripts/run_mg_wsf_bv2.py` | MG-WSF fusion from `candidate_registry.csv` |
| `scripts/eval_bv2.py` | BV2 test evaluation |

Shell wrappers: `train_bv2.sh`, `eval_bv2.sh`, `run_mg_wsf_bv2.sh`
