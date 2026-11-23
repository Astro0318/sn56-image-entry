# SN56 image entry (baseline starter)

Minimal **image-tournament** training repository for Bittensor netuid 56 (Gradients).

Built from official files in [`gradients-ai/G.O.D`](https://github.com/gradients-ai/G.O.D)
commit `5bf3ea0289ae9b97da4a8436135b330f1f7da09c`.

## Status (2026-11-23)

- Structural layout matches the image guide (LICENSE/NOTICE/Dockerfile/entrypoint/templates).
- Original deltas on top of stock G.O.D baseline:
  - **I8** promote `last.safetensors`
  - **I4** `EMPTY_PROMPT_RATE` (default 0.45) caption dropout
  - **I2-light** denser step checkpoints (`SAVE_EVERY_STEPS` / `MAX_STEP_SAVES`)
- Still far from crown-level custom Comfy winners (~37k+ LOC).
  See `/root/sn56-work/study/IMAGE_WINNER_GAP.md`.
- **Do not** copy winner source here (duplicate-elimination risk + fee burn).

## Required layout

- `LICENSE.md` + `NOTICE` — identical to G.O.D
- `ops/docker/standalone-image-toolkit-trainer.dockerfile`
- `ops/docker/scripts/image_toolkit_entrypoint.py`
- `core/training_templates/`

## Checks (CPU-only)

```bash
./scripts/cpu_safe_checks.sh
```

## GPU local test (needs NVIDIA + credentials; not on this VPS)

From a G.O.D checkout with this repo’s files present, after editing the script:

```bash
./ops/examples/training/run_image_task.sh
```

## Before paying the 0.4 τ fee

Replace this baseline with an **original** competitive recipe, publish to GitHub,
pin a 40-char commit, then register. With low TAO, skip fee until that is done.
