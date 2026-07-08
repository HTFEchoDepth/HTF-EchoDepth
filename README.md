# HTF-EchoDepth

**Hierarchical Time-Frequency Echo Depth Estimation with Metric-Guided Weight-Space Fusion**

> **Paper status:** Under review. Citation placeholder below — update when the paper is published.

Official open-source release of **HTF-EchoDepth** for **BatVision V2 (BV2)** — the implementation used in our main paper experiments.

---

## Overview

HTF-EchoDepth estimates dense depth maps from **binaural echoes** using a hierarchical time–frequency encoder–decoder. The network combines local time–frequency encoding (**LTFE**), intra-stage multi-scale residual enhancement (**IMRE**), and latent dual-path encoding (**LDPE**) within a U-Net-style backbone. After training, **metric-guided weight-space fusion (MG-WSF)** combines validation-selected checkpoints into a single model for final test reporting.

![HTF-EchoDepth overview](assets/figures/framework.png)

**This release focuses on the BV2 implementation used in the main paper experiments.** Spectrograms are computed **on the fly** from `.wav` files in the dataloader (runtime STFT). **Checkpoints are not stored in Git.** Exact Table 1 reproduction is recommended using **released pretrained checkpoints**; training code is provided, but stochastic training may lead to small numerical variations. Internal diagnostic scripts and exploratory branches are **not** included. Experimental extensions such as BiMamba, and additional datasets such as Replica support, may be released in future updates.

---

## Repository scope

| Included | Not in this release (may come in future updates) |
|----------|--------------------------------------------------|
| BV2 data tools, dataloader, runtime STFT | Other datasets (e.g. Replica) |
| HTF-EchoDepth model + paper configs | Experimental extensions (e.g. BiMamba) |
| Train / eval / MG-WSF scripts | Exploratory branches |
| Reference BV2 results (`results/`) | Dataset or checkpoint binaries |

```
configs/bv2/       Paper-style BV2 configs
htf_echodepth/     Model, data, loss, metrics, fusion
scripts/           Train, eval, MG-WSF, index/validate tools
docs/              Data, metrics, checkpoints, recipes
results/           Published BV2 reference numbers
assets/figures/    Overview figure (upload separately)
```

---

## Installation / environment

```bash
git clone <repository-url>
cd HTF-EchoDepth
pip install -r requirements.txt
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export HTF_BV2_DATA_ROOT=/path/to/bv2_processed
export HTF_CHECKPOINT_ROOT=/path/to/checkpoints
```

Conda users: see `environment.yml`.  
Reference stack used in our local verification: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

---

## BV2 data preparation

Dataset files are **not** hosted in this repository.

1. Obtain **BatVision V2** processed data (see [docs/DATA.md](docs/DATA.md)).
2. Build portable index CSVs (relative paths only):

```bash
python scripts/build_bv2_index.py \
  --raw-root "${HTF_BV2_DATA_ROOT}" \
  --out-dir data/bv2_index
```

3. Validate a few samples:

```bash
python scripts/validate_bv2_data.py \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --num-samples 3
```

Expected echo shape: **2 × 256 × 256**. Depth maps: `.npy` uint16 mm → meters in the loader.

---

## Pretrained checkpoints

**Checkpoints are not committed to Git.** Place released weights under:

```
checkpoints/
├── htf_echodepth_bv2_before_mgwsf.pth
└── htf_echodepth_bv2_mgwsf.pth
```

Release policy: available upon request / released separately — see [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).  
Paper checkpoint compatibility is handled automatically by `htf_echodepth.models.compatibility`.

---

## Evaluation

Test-split evaluation uses the **valid-depth protocol** (`valid_min_depth = 0.5 m`, `max_depth = 30 m`). See [docs/METRIC_PROTOCOL.md](docs/METRIC_PROTOCOL.md).

```bash
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint checkpoints/htf_echodepth_bv2_mgwsf.pth \
  --output-dir outputs/eval_bv2
```

Or: `bash scripts/eval_bv2.sh --help`

---

## Training and model fusion

BV2 training follows the paper recipe (80 epochs, AdamW, validation-based checkpoint selection). During training the script:

1. Logs **validation metrics** each epoch (`val_metrics.csv`).
2. Saves **metric-role candidate checkpoints** when validation improves (`best_rmse.pth`, `best_rel.pth`, `best_log10.pth`, `best_delta1.pth`, `best_delta2.pth`, `best_delta3.pth`, `best_wbrs.pth`).
3. Writes **`candidate_registry.csv`** — a portable record of candidates and their validation scores (no test-set selection).

After training, **MG-WSF** reads `candidate_registry.csv`, applies **MRCR** (metric-role retention) and **VGFS** (validation-guided weight search) on the **validation split**, then performs **weight-space fusion** to produce a **single fused checkpoint**.

```bash
# 1. Train (paper-style config)
python scripts/train_bv2.py \
  --config configs/bv2/train_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --output-dir outputs/bv2_run

# 2. Fuse validation-selected donors → single checkpoint
python scripts/run_mg_wsf_bv2.py \
  --config configs/bv2/mg_wsf_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --candidate-registry outputs/bv2_run/candidate_registry.csv \
  --output-dir outputs/bv2_run/fused \
  --save-fused-checkpoint

# 3. Evaluate fused model on test split (final reporting only)
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint outputs/bv2_run/fused/htf_echodepth_mg_wsf_fused.pth
```

**Notes:**

- MG-WSF uses **validation only** for donor selection and fusion weights; the **test set is for final reporting**.
- Meaningful fusion requires **≥ 2 distinct validation-selected donors** from training.
- `--allow-degenerate-fusion` is for **smoke/debug runs only** (single donor copy-through) — **not** performance evidence.

Full workflows: [docs/RESULTS_REPRODUCTION.md](docs/RESULTS_REPRODUCTION.md) · Recipe details: [docs/TRAINING_RECIPE.md](docs/TRAINING_RECIPE.md)

---

## Expected BV2 results (Table 1)

BV2 test split (**N = 584**), valid-depth protocol (gt ≥ 0.5 m). Baseline rows are **reported results from prior work**. **HTF-EchoDepth** is our evaluated result (after MG-WSF). For exact reproduction, use the released pretrained checkpoint.

| Method | RMSE ↓ | REL ↓ | log10 ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|--------|-------:|------:|--------:|-----:|-----:|-----:|
| Echo-Net *(reported)* | 2.878 | 0.521 | 0.197 | 0.430 | 0.629 | 0.765 |
| Bat-Net *(reported)* | 2.676 | 0.432 | 0.160 | 0.497 | 0.717 | 0.835 |
| AHMF-Net *(reported)* | 2.195 | 0.430 | 0.074 | 0.502 | 0.718 | 0.838 |
| **HTF-EchoDepth** *(ours)* | **2.187** | **0.382** | **0.154** | **0.516** | **0.724** | **0.840** |

Also see [`results/paper_results_bv2.csv`](results/paper_results_bv2.csv).

---

## Reproduction notes

| Goal | Recommended path |
|------|------------------|
| Match Table 1 numbers | Released pretrained checkpoint + [RESULTS_REPRODUCTION.md](docs/RESULTS_REPRODUCTION.md) workflow A |
| Retrain + fuse from scratch | Workflow B in [RESULTS_REPRODUCTION.md](docs/RESULTS_REPRODUCTION.md) |
| Metric definitions | [METRIC_PROTOCOL.md](docs/METRIC_PROTOCOL.md) |
| Paper ↔ code naming | [NAMING_MAP.md](docs/NAMING_MAP.md) |

Training from scratch may show small drift vs published CSV due to stochasticity and tie-breaking.

---

## License

See [LICENSE](LICENSE).

---

## Citation

```bibtex
% TODO: replace with final paper citation when available
@article{htf_echodepth2026,
  title   = {HTF-EchoDepth: Hierarchical Time-Frequency Echo Depth Estimation with Metric-Guided Weight-Space Fusion},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```
