# Changelog — sn56-image-entry

## 2026-11-24 — I5 time-budget steps + stronger I2 pick

- Scale AI-Toolkit `train.steps` from `--hours-to-complete` (`STEPS_PER_HOUR`, default 900)
- When promoting, prefer second-half numbered checkpoints if `last.safetensors` is absent
- Public clone/licence verified for https://github.com/Astro0318/sn56-image-entry

## 2026-11-23 — I4 empty-prompt share + I2 light checkpoint policy

- `EMPTY_PROMPT_RATE` (default **0.45**) sets `caption_dropout_rate` on all dataset blocks
- Aligns training closer to eval’s 0.5 captioned / 0.5 empty-prompt mix
- `SAVE_EVERY_STEPS` (default 100) + `MAX_STEP_SAVES` (default 8) keep denser checkpoints
- CPU tests extended in `scripts/test_promote_last.py`

## 2026-11-23 — I8 deterministic `last.safetensors`

- Original change in `ops/docker/scripts/image_toolkit_entrypoint.py`
- After training, promote the evaluator-preferred adapter to `last.safetensors`
- Selection order mirrors `docs/miner_image.md` Output contract
- CPU unit test: `scripts/test_promote_last.py`
