# Checkpoints (BV2)

Checkpoint loading utilities are provided for local BV2 testing. Place your checkpoint under a local `checkpoints/` directory or pass its path with `--checkpoint`.

---

## Local layout

Example:

```
checkpoints/
└── checkpoint.pth
```

---

## Loading & compatibility

Compatible checkpoints can be loaded through the public helper:

```python
from htf_echodepth.models import build_htf_echodepth
from htf_echodepth.utils.checkpoint import load_checkpoint

model = build_htf_echodepth()
load_checkpoint(model, "checkpoints/checkpoint.pth")
```

Lower-level helper: `htf_echodepth.models.compatibility.load_model_state`

---

## Testing

```bash
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint checkpoints/checkpoint.pth
```

---

## Environment variable

```bash
export HTF_CHECKPOINT_ROOT=/path/to/checkpoints
```

Then reference `${HTF_CHECKPOINT_ROOT}/checkpoint.pth` in your commands.
