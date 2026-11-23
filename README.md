# SN56 image entry (baseline starter)

Minimal **image-tournament** training repository for Bittensor netuid 56 (Gradients).

Built from official files in [`gradients-ai/G.O.D`](https://github.com/gradients-ai/G.O.D)
commit `5bf3ea0289ae9b97da4a8436135b330f1f7da09c`.

## Status (2026-11-23)

- Structural layout matches the image guide (LICENSE/NOTICE/Dockerfile/entrypoint/templates).
- This is the **stock baseline**, not a competitive crown challenger.
- Latest published image winner is a large custom ComfyUI trainer (~37k+ LOC).
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
