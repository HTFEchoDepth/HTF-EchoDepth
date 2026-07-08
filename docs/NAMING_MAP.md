# Naming Map — Paper ↔ Implementation

Public naming follows **paper terminology**. Normal users should import **`HTFEchoDepth`** and the documented scripts — not low-level backbone modules.

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

## Training & fusion (paper-facing)

| Term | Full name | Module |
|------|-----------|--------|
| **WBRS** | Weighted Balanced Relative Score | `htf_echodepth/fusion/wbrs.py` |
| **MRCR** | Metric-Role Candidate Retention | `htf_echodepth/fusion/candidate_retention.py` |
| **VGFS** | Validation-Guided Fusion Selection | `htf_echodepth/fusion/fusion_selection.py` |
| **MG-WSF** | Metric-Guided Weight-Space Fusion | `htf_echodepth/fusion/mg_wsf.py` |

Training candidate logging: `htf_echodepth/fusion/candidate_manager.py`

---

## BV2 configs

```
configs/bv2/train_htf_echodepth_bv2.yaml
configs/bv2/eval_htf_echodepth_bv2.yaml
configs/bv2/mg_wsf_bv2.yaml
```

---

## Checkpoint compatibility (internal, automatic)

Released pretrained checkpoints were saved from the research codebase. The public `HTFEchoDepth` class wraps the same architecture. Loading goes through `htf_echodepth.models.compatibility` (e.g. stripping `module.` prefixes) — **end users normally call** `load_checkpoint()` only:

```python
from htf_echodepth.models import build_htf_echodepth
from htf_echodepth.utils.checkpoint import load_checkpoint

model = build_htf_echodepth()
load_checkpoint(model, "checkpoints/htf_echodepth_bv2_mgwsf.pth")
```

---

## Backbone internals (not public API)

Low-level block implementations live under `htf_echodepth/models/backbone/` for **checkpoint fidelity**. These modules exist only to preserve `state_dict` compatibility with released weights. Use the public API above in application code and documentation.
