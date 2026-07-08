# Checkpoints (BV2)

Pretrained BV2 checkpoints are distributed separately and should be placed in a local checkpoint directory.

---

## Release policy

| Item | Policy |
|------|--------|
| Storage | Local checkpoint directory |
| Distribution | Released separately / available upon request |
| Request | Open an issue if the download link is unavailable |

---

## Expected layout

Place downloaded weights under a local directory (example):

```
checkpoints/
├── htf_echodepth_bv2_before_mgwsf.pth   # main model before MG-WSF
└── htf_echodepth_bv2_mgwsf.pth          # final fused model (Table 1)
```

Optional training bundles may also include metric-role donor checkpoints and a fusion recipe JSON. The pre-fused release file is sufficient for Table 1 evaluation.

---

## Loading & compatibility

Paper checkpoints may use legacy `state_dict` key layouts (e.g. `module.` prefixes). The public API handles this automatically:

```python
from htf_echodepth.models import build_htf_echodepth
from htf_echodepth.utils.checkpoint import load_checkpoint

model = build_htf_echodepth()
load_checkpoint(model, "checkpoints/htf_echodepth_bv2_mgwsf.pth")
```

Lower-level helper: `htf_echodepth.models.compatibility.load_model_state`

---

## Evaluation

```bash
python scripts/eval_bv2.py \
  --config configs/bv2/eval_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-file data/bv2_index/test_index.csv \
  --checkpoint checkpoints/htf_echodepth_bv2_mgwsf.pth
```

See [RESULTS_REPRODUCTION.md](RESULTS_REPRODUCTION.md) for the full workflow.

---

## Train-from-scratch vs pretrained

| Goal | Recommendation |
|------|----------------|
| **Table 1 evaluation** | Use **released pretrained checkpoints** |
| Research / ablation retrain | Supported; expect small numerical variation |

Published reference values: [`results/paper_results_bv2.csv`](../results/paper_results_bv2.csv)

---

## Environment variable

```bash
export HTF_CHECKPOINT_ROOT=/path/to/checkpoints
```

Then reference `${HTF_CHECKPOINT_ROOT}/htf_echodepth_bv2_mgwsf.pth` in your commands.
