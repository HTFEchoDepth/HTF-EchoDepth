# Environment

Reference software stack used during local verification of this BV2 release. Your environment may differ slightly; adjust CUDA/PyTorch builds to match your GPU driver.

---

## Reference configuration

| Component | Version / note |
|-----------|----------------|
| OS | Linux (Ubuntu-compatible), x86_64 |
| Python | 3.10.x |
| PyTorch | 2.11 + CUDA 12.6 |
| torchaudio | 2.11 + CUDA 12.6 |
| GPU (verified) | NVIDIA RTX 4090 |
| librosa | 0.11.x |
| numpy | 2.2.x |
| scipy | 1.15.x |
| PyYAML | 6.0.x |

---

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Conda alternative: `environment.yml`

---

## Runtime variables

| Variable | Purpose |
|----------|---------|
| `HTF_BV2_DATA_ROOT` | BV2 processed data root |
| `HTF_CHECKPOINT_ROOT` | Directory for downloaded `.pth` files (outside Git) |
| `PYTHONPATH` | Repository root (so `import htf_echodepth` works) |

---

## CPU-only note

Scripts fall back to CPU when CUDA is unavailable. Full BV2 training is intended for GPU.
