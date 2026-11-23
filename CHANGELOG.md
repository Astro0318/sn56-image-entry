# Changelog — sn56-image-entry

## 2026-11-23 — I8 deterministic `last.safetensors`

- Original change in `ops/docker/scripts/image_toolkit_entrypoint.py`
- After training, promote the evaluator-preferred adapter to `last.safetensors`
- Selection order mirrors `docs/miner_image.md` Output contract
- CPU unit test: `scripts/test_promote_last.py`

Not yet: empty-prompt share (I4), checkpoint selection (I2).
