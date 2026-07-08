# Training and Model Fusion

Training and MG-WSF follow the configuration used for the reported results. The manuscript is currently under review, and the citation will be updated after publication.

The public entry points are kept for consistency with the project workflow:

```bash
python scripts/train_bv2.py \
  --config configs/bv2/train_htf_echodepth_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --output-dir outputs/bv2_run

python scripts/run_mg_wsf_bv2.py \
  --config configs/bv2/mg_wsf_bv2.yaml \
  --data-root "${HTF_BV2_DATA_ROOT}" \
  --index-dir data/bv2_index \
  --candidate-registry outputs/bv2_run/candidate_registry.csv \
  --output-dir outputs/bv2_run/fused \
  --save-fused-checkpoint
```

Full training and model-fusion details will be released soon.
