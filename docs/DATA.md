# Data

This release uses **BatVision V2 (BV2)** from the **Audio-Visual BatVision Dataset**. We thank the BatVision Dataset authors for releasing the dataset and enabling research on real-world echo-based scene understanding.

Official resources:

- Project page: [Audio-Visual BatVision Dataset](https://amandinebtto.github.io/Batvision-Dataset/)
- Official GitHub: [AmandineBtto/Batvision-Dataset](https://github.com/AmandineBtto/Batvision-Dataset)
- Paper: *The Audio-Visual BatVision Dataset for Research on Sight and Sound*, IROS 2023

Please follow the dataset license and citation requirements from the official BatVision Dataset release.

This document describes the **BV2** data layout expected by HTF-EchoDepth. Dataset files are distributed through the official BatVision Dataset resources. This repository provides scripts to build portable indices and validate samples.

---

## Design principle: raw WAV + runtime STFT

- **Echo format:** binaural `.wav`; spectrograms are computed at load time.
- **STFT is computed at runtime** in `htf_echodepth.data.BV2Dataset` on every load.
- STFT parameters are fixed in `htf_echodepth/data/bv2_dataset.py` (paper training path).

---

## Expected raw layout

```
${HTF_BV2_DATA_ROOT}/
├── <scene_name>/
│   ├── audio/
│   │   └── audio_<id>.wav      # binaural PCM, 44.1 kHz
│   ├── depth/
│   │   └── depth_<id>.npy      # uint16, millimeters
│   ├── train.csv               # official per-scene split
│   ├── val.csv
│   └── test.csv
└── data/bv2_index/             # generated (recommended location)
    ├── train_index.csv
    ├── val_index.csv
    ├── test_index.csv
    └── dataset_manifest.json
```

### Split sizes

| Split | Samples |
|-------|--------:|
| train | 1911 |
| val | 625 |
| test | 584 |

### Depth on disk

- Format: **NumPy `.npy`**, **uint16**, values in **millimeters**
- Loader: converts depth values from millimeters to meters and prepares model targets.

### Runtime STFT

| Parameter | Value |
|-----------|-------|
| Truncation | Paper range setting with the audio sample rate |
| `n_fft` | 512 |
| `win_length` | 64 |
| `hop_length` | 16 |
| Output resize | **256 × 256** per channel |
| **Echo tensor shape** | **2 × 256 × 256** |
| **Depth tensor shape** | **1 × 256 × 256** |

---

## Build index

Creates CSV indices with relative paths and leaves source data in place:

```bash
python scripts/build_bv2_index.py \
  --raw-root "${HTF_BV2_DATA_ROOT}" \
  --out-dir data/bv2_index
```

Expected counts: train **1911**, val **625**, test **584**.

Index columns: `sample_id`, `scene`, `split`, `audio_relpath`, `depth_relpath`.

---

## Validate data

```bash
python scripts/validate_bv2_data.py \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --num-samples 3
```

Checks: wav readable, runtime STFT runs, echo shape **2×256×256**, depth npy loads and converts to meters.

---

## Load in Python

```python
from htf_echodepth.data import BV2Dataset

ds = BV2Dataset(
    data_root="${HTF_BV2_DATA_ROOT}",
    index_file="data/bv2_index/test_index.csv",
)
sample = ds[0]
assert sample["echo_spectrogram"].shape == (2, 256, 256)
assert sample["depth"].shape == (1, 256, 256)
```

---

## Environment variable

| Variable | Description |
|----------|-------------|
| `HTF_BV2_DATA_ROOT` | BV2 processed data root |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_bv2_index.py` | Per-scene CSV → unified relative index CSVs |
| `scripts/validate_bv2_data.py` | Check files + dataloader output shapes |

Evaluation metrics and aggregation are described in [METRIC_PROTOCOL.md](METRIC_PROTOCOL.md). Checkpoint usage is described in [CHECKPOINTS.md](CHECKPOINTS.md).
