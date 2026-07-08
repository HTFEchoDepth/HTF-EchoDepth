# Naming Map — Paper ↔ Implementation

Public naming follows **paper terminology**. Recommended entry points are **`HTFEchoDepth`** and the documented scripts.

---

## Model entry point

| Paper name | Implementation |
|------------|----------------|
| **HTF-EchoDepth** | `htf_echodepth.models.HTFEchoDepth` · `build_htf_echodepth()` |

---

## Architectural components (paper-facing)

| Acronym | Full name | Public module |
|---------|-----------|---------------|
| **LTFE** | Local Time-Frequency Encoding Block | `htf_echodepth/models/ltfe.py` |
| **FPC** | Frequency-Position Calibration | Inside LTFE blocks |
| **TFDE** | Time-Frequency Decoupled Encoding | Inside LTFE blocks |
| **CDE** | Cross-Frequency Dependency Enhancement | Inside LTFE blocks |
| **IMRE** | Intra-stage Multi-scale Residual Enhancement | `htf_echodepth/models/imre.py` |
| **LDPE** | Latent Dual-Path Encoding | `htf_echodepth/models/ldpe.py` |

U-Net encoder–decoder helpers: `htf_echodepth/models/unet_backbone.py`

---

## Fusion components

| Term | Full name | Module |
|------|-----------|--------|
| **WBRS** | Weighted Balanced Relative Score | `htf_echodepth/fusion/wbrs.py` |
| **MRCR** | Metric-Role Candidate Retention | `htf_echodepth/fusion/candidate_retention.py` |
| **VGFS** | Validation-Guided Fusion Selection | `htf_echodepth/fusion/fusion_selection.py` |
| **MG-WSF** | Metric-Guided Weight-Space Fusion | `htf_echodepth/fusion/mg_wsf.py` |

Training candidate logging: `htf_echodepth/fusion/candidate_manager.py`

---

## BV2 config

```
configs/bv2/eval_htf_echodepth_bv2.yaml
```

---

## Checkpoint compatibility

The public `HTFEchoDepth` class wraps the released architecture. Checkpoint loading goes through `htf_echodepth.models.compatibility` (e.g. stripping `module.` prefixes), with `load_checkpoint()` as the recommended helper:

```python
from htf_echodepth.models import build_htf_echodepth
from htf_echodepth.utils.checkpoint import load_checkpoint

model = build_htf_echodepth()
load_checkpoint(model, "checkpoints/checkpoint.pth")
```

---

## Backbone components

Low-level block implementations live under `htf_echodepth/models/backbone/` for **checkpoint fidelity** and `state_dict` compatibility with released weights. Use the public API above in application code and documentation.
