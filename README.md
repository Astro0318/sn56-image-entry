# SN56 image tournament entry (starter)

Minimal training repository for Bittensor **netuid 56 (Gradients)** image tournament.

Built from the official starter files in
[`gradients-ai/G.O.D`](https://github.com/gradients-ai/G.O.D) at commit
`5bf3ea0289ae9b97da4a8436135b330f1f7da09c`.

## Required layout (image guide)

- `LICENSE.md` + `NOTICE` — identical to G.O.D
- `ops/docker/standalone-image-toolkit-trainer.dockerfile`
- `ops/docker/scripts/image_toolkit_entrypoint.py`
- `core/training_templates/` — AI Toolkit YAML templates

## Local GPU training test (needs NVIDIA GPU)

From a full G.O.D checkout with this repo's files present, after editing
credentials in the script:

```bash
./ops/examples/training/run_image_task.sh
```

On a CPU-only host that script cannot complete training (`--gpus all`).

## CUDA

Trainer hosts support CUDA **12.8 or lower**. The official Dockerfile pins
PyTorch cu128 wheels for musubi and uses `diagonalge/ai-toolkit:latest`.
