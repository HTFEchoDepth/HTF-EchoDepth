# HTF-EchoDepth

**Hierarchical Time-Frequency Feature Encoding for Dense Depth Estimation from Binaural Echoes**

> **Paper status:** Under review. Citation will be updated after publication.

## Overview

This repository provides the official implementation of **HTF-EchoDepth**, a framework for dense depth estimation from binaural echoes. HTF-EchoDepth treats echo spectrograms as structured time–frequency observations and organizes their representations through hierarchical feature encoding at local, stage, and latent levels. It further uses validation-guided weight-space fusion to improve the final model while preserving single-model inference.

![HTF-EchoDepth overview](assets/figures/framework.png)

## Main results on BV2

We evaluate HTF-EchoDepth on the **BatVision V2 (BV2)** test split using the paper evaluation setting. Baseline rows are reported results from prior work, and HTF-EchoDepth is our evaluated result after model fusion.

| Method | RMSE ↓ | REL ↓ | log10 ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|--------|-------:|------:|--------:|-----:|-----:|-----:|
| Echo-Net *(reported)* | 2.878 | 0.521 | 0.197 | 0.430 | 0.629 | 0.765 |
| Bat-Net *(reported)* | 2.676 | 0.432 | 0.160 | 0.497 | 0.717 | 0.835 |
| AHMF-Net *(reported)* | 2.195 | 0.430 | 0.074 | 0.502 | 0.718 | 0.838 |
| **HTF-EchoDepth** *(ours)* | **2.187** | **0.382** | **0.154** | **0.516** | **0.724** | **0.840** |

Also see [`results/paper_results_bv2.csv`](results/paper_results_bv2.csv). Exact reproduction is recommended using the released pretrained checkpoint.
Metric definitions and the BV2 evaluation protocol are described in [`docs/METRIC_PROTOCOL.md`](docs/METRIC_PROTOCOL.md).

## Installation

```bash
git clone https://github.com/AIMHIGH-WU/HTF-EchoDepth.git
cd HTF-EchoDepth

pip install -r requirements.txt
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Conda users may also refer to [`environment.yml`](environment.yml). The reference environment used in our local verification is summarized in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Data

This project uses **BatVision V2 (BV2)** from the **Audio-Visual BatVision Dataset**. We thank the BatVision Dataset authors for releasing the dataset and enabling research on real-world echo-based scene understanding.

Please download the dataset from the official sources and follow their license and citation requirements:

- Project page: [Audio-Visual BatVision Dataset](https://amandinebtto.github.io/Batvision-Dataset/)
- Official GitHub: [AmandineBtto/Batvision-Dataset](https://github.com/AmandineBtto/Batvision-Dataset)
- Paper: *The Audio-Visual BatVision Dataset for Research on Sight and Sound*, IROS 2023

More details are provided in [`docs/DATA.md`](docs/DATA.md).

## Evaluation

Pretrained checkpoints are released separately and should be placed under:

```text
checkpoints/
├── htf_echodepth_bv2_before_mgwsf.pth
└── htf_echodepth_bv2_mgwsf.pth
```

Evaluate the fused checkpoint on the BV2 test split:

```bash
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint checkpoints/htf_echodepth_bv2_mgwsf.pth \
  --output-dir outputs/eval_bv2
```

Metric definitions and the BV2 evaluation protocol are described in [`docs/METRIC_PROTOCOL.md`](docs/METRIC_PROTOCOL.md). Checkpoint usage is described in [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

## Training and model fusion

BV2 training follows the paper recipe with validation-based checkpoint selection and final model fusion. Training saves validation-selected candidate checkpoints, and MG-WSF constructs the final fused model from these candidates.

```bash
python scripts/train_bv2.py \
  --config configs/bv2/train_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --output-dir outputs/bv2_run
```

Then construct the final fused checkpoint:

```bash
python scripts/run_mg_wsf_bv2.py \
  --config configs/bv2/mg_wsf_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --candidate-registry outputs/bv2_run/candidate_registry.csv \
  --output-dir outputs/bv2_run/fused \
  --save-fused-checkpoint
```

Finally evaluate the fused checkpoint:

```bash
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint outputs/bv2_run/fused/htf_echodepth_mg_wsf_fused.pth
```

More details are available in [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md) and [`docs/RESULTS_REPRODUCTION.md`](docs/RESULTS_REPRODUCTION.md). Additional datasets or experimental extensions may be supported in future updates.

## Citation

If you find this repository useful, please cite our paper after publication.

```bibtex
% TODO: replace with final paper citation when available
@article{htf_echodepth2026,
  title   = {HTF-EchoDepth: Hierarchical Time-Frequency Feature Encoding for Dense Depth Estimation from Binaural Echoes},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

Please also cite the BatVision Dataset paper when using BV2:

```bibtex
@inproceedings{brunetto2023batvision,
  title     = {The Audio-Visual BatVision Dataset for Research on Sight and Sound},
  author    = {Brunetto, Amandine and Hornauer, Sascha and Yu, Stella X. and Moutarde, Fabien},
  booktitle = {2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2023}
}
```

See [`LICENSE`](LICENSE) for licensing information.
