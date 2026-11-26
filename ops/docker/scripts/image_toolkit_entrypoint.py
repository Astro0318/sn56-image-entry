#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml


# Evaluator picks one .safetensors under checkpoints/ in this order
# (docs/miner_image.md Output contract): last.safetensors if unique;
# else highest trailing number; else the only file. We always promote the
# chosen adapter to last.safetensors so the pick is deterministic (I8).
_TRAILING_NUM = re.compile(r"(\d+)(?=\.safetensors$)", re.IGNORECASE)


CACHE_ROOT = Path("/cache")
CONFIG_TEMPLATE_ROOT = Path("/workspace/core/training_templates")
CONFIG_OUTPUT_ROOT = Path("/dataset/configs")
DATASET_ZIP_ROOT = CACHE_ROOT / "datasets"
IMAGE_DATASET_ROOT = Path("/dataset/images")
EXTRACT_ROOT = Path("/dataset/extracted")
CHECKPOINTS_ROOT = Path("/app/checkpoints")
AI_TOOLKIT_ROOT = Path("/app/ai-toolkit")
MUSUBI_ROOT = Path("/app/musubi-tuner")
MUSUBI_PYTHON = os.environ.get("MUSUBI_PYTHON", "/opt/musubi/bin/python")
MUSUBI_ACCELERATE = os.environ.get("MUSUBI_ACCELERATE", "/opt/musubi/bin/accelerate")
HIDREAM_DIT = CACHE_ROOT / "hf_cache" / "checkpoints" / "hidream_o1_image_bf16.safetensors"
# Trainers resolve some assets by hub repo id; the downloader pre-caches them here.
OFFLINE_HUB_ENV = {
    **os.environ,
    "HF_HUB_CACHE": str(CACHE_ROOT / "hf_cache"),
    "HF_HUB_OFFLINE": "1",
    "MODELS_PATH": str(CACHE_ROOT / "ai_toolkit_models"),
}

TEXT_TO_IMAGE = "text-to-image"
EDIT = "edit"
TEMPLATE_BY_MODEL_AND_MODE = {
    ("flux", TEXT_TO_IMAGE): "base_diffusion_flux.yaml",
    ("z-image", TEXT_TO_IMAGE): "base_diffusion_zimage.yaml",
    ("qwen-image", TEXT_TO_IMAGE): "base_diffusion_qwen_image.yaml",
    ("ideogram4", TEXT_TO_IMAGE): "base_diffusion_ideogram4.yaml",
    ("krea2", TEXT_TO_IMAGE): "base_diffusion_krea2.yaml",
    ("flux2-klein-4b", TEXT_TO_IMAGE): "base_diffusion_flux2_klein_4b.yaml",
    ("flux2-klein-4b", EDIT): "base_diffusion_flux2_klein_4b_edit.yaml",
    ("qwen-image-edit-2511", EDIT): "base_diffusion_qwen_image_edit_2511.yaml",
    ("hidream-o1", TEXT_TO_IMAGE): "base_diffusion_hidream_o1.yaml",
    ("moss-image-base", TEXT_TO_IMAGE): "base_diffusion_moss_image_base.yaml",
    ("moss-image-base", EDIT): "base_diffusion_moss_image_base_edit.yaml",
}
IDEOGRAM4_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-4B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted({model_type for model_type, _ in TEMPLATE_BY_MODEL_AND_MODE}),
    )
    parser.add_argument("--training-mode", choices=[TEXT_TO_IMAGE, EDIT], default=TEXT_TO_IMAGE)
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True)
    parser.add_argument("--trigger-word")
    return parser.parse_args()


def cached_model_path(model: str) -> Path:
    cache_name = model.replace("/", "--")
    return CACHE_ROOT / "models" / cache_name


def _manifest_path(root: Path, value: str, sample_id: str) -> Path:
    path = (root / value).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Invalid or missing manifest path for sample {sample_id}: {value}")
    return path


def _prepare_edit_dataset(extract_root: Path) -> tuple[Path, list[Path]]:
    from PIL import Image

    manifests = list(extract_root.rglob("manifest.jsonl"))
    if len(manifests) != 1:
        raise ValueError(f"Edit dataset must contain exactly one manifest.jsonl; found {len(manifests)}")

    manifest = manifests[0]
    dataset_root = manifest.parent
    targets_path = IMAGE_DATASET_ROOT / "targets"
    controls_root = IMAGE_DATASET_ROOT / "controls"
    targets_path.mkdir(parents=True)
    controls_root.mkdir(parents=True)

    expected_source_count: int | None = None
    sample_count = 0
    with manifest.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = sample.get("sample_id")
            instruction = sample.get("instruction")
            sources = sample.get("sources")
            target = sample.get("target")
            if not isinstance(sample_id, str) or not sample_id or "/" in sample_id or "\\" in sample_id:
                raise ValueError(f"Invalid sample_id on manifest line {line_number}")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Missing instruction for sample {sample_id}")
            if not isinstance(target, str):
                raise ValueError(f"Missing target for sample {sample_id}")
            if not isinstance(sources, list) or not 1 <= len(sources) <= 3 or not all(
                isinstance(source, str) for source in sources
            ):
                raise ValueError(f"Sample {sample_id} must have one to three source images")
            if expected_source_count is None:
                expected_source_count = len(sources)
            elif len(sources) != expected_source_count:
                raise ValueError("Every sample in an edit dataset must have the same number of sources")

            target_source = _manifest_path(dataset_root, target, sample_id)
            target_destination = targets_path / f"{sample_id}.png"
            with Image.open(target_source) as image:
                image.convert("RGB").save(target_destination)
            target_destination.with_suffix(".txt").write_text(instruction.strip() + "\n", encoding="utf-8")

            for index, source_value in enumerate(sources):
                control_path = controls_root / str(index)
                control_path.mkdir(exist_ok=True)
                with Image.open(_manifest_path(dataset_root, source_value, sample_id)) as image:
                    image.convert("RGB").save(control_path / target_destination.name)
            sample_count += 1

    if sample_count == 0:
        raise ValueError("Edit dataset manifest contains no samples")
    return targets_path, [controls_root / str(index) for index in range(expected_source_count or 0)]


def prepare_dataset(task_id: str, training_mode: str) -> tuple[Path, list[Path]]:
    zip_path = DATASET_ZIP_ROOT / f"{task_id}_tourn.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected cached image dataset zip at {zip_path}")

    if IMAGE_DATASET_ROOT.exists():
        shutil.rmtree(IMAGE_DATASET_ROOT)
    IMAGE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    extract_root = EXTRACT_ROOT / task_id
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        resolved_extract_root = extract_root.resolve()
        for member in archive.infolist():
            destination = (extract_root / member.filename).resolve()
            if destination != resolved_extract_root and resolved_extract_root not in destination.parents:
                raise ValueError(f"Unsafe path in image dataset zip: {member.filename}")
        archive.extractall(extract_root)

    if training_mode == EDIT:
        return _prepare_edit_dataset(extract_root)

    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".txt"}
    copied = 0
    for source in extract_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in supported_suffixes:
            destination = IMAGE_DATASET_ROOT / source.name
            if destination.exists():
                destination = IMAGE_DATASET_ROOT / f"{source.parent.name}_{source.name}"
            shutil.copy2(source, destination)
            copied += 1

    if copied == 0:
        raise RuntimeError(f"No supported image/caption files found in {zip_path}")
    # I1: every image must have a sibling .txt so the trainer/eval pairing is stable.
    ensure_image_caption_pairs(IMAGE_DATASET_ROOT)
    return IMAGE_DATASET_ROOT, []


def ensure_image_caption_pairs(dataset_root: Path) -> int:
    """Create empty .txt captions for images that lack them (I1).

    Eval mixes captioned and empty prompts; orphan images without captions are a
    common silent quality hit. Returns how many caption stubs were written.
    """
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    created = 0
    for image_path in sorted(dataset_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in image_suffixes:
            continue
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            caption_path.write_text("\n", encoding="utf-8")
            created += 1
    return created


def prepare_config(args: argparse.Namespace, dataset_path: Path, control_paths: list[Path]) -> Path:
    try:
        template_name = TEMPLATE_BY_MODEL_AND_MODE[(args.model_type, args.training_mode)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AI Toolkit training combination: {args.model_type}/{args.training_mode}"
        ) from exc
    template_path = CONFIG_TEMPLATE_ROOT / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Missing ai-toolkit template for {args.model_type}: {template_path}")

    with template_path.open() as file:
        config = yaml.safe_load(file)

    config_body = config.setdefault("config", {})
    config_body["name"] = args.expected_repo_name

    process = config_body.setdefault("process", [{}])[0]
    process["training_folder"] = str(CHECKPOINTS_ROOT / args.task_id)
    process["trigger_word"] = args.trigger_word
    train = process.setdefault("train", {})
    if steps_override := os.environ.get("TRAIN_STEPS_OVERRIDE"):
        train["steps"] = int(steps_override)
    else:
        # I5: scale step budget from the task's --hours-to-complete window.
        train["steps"] = steps_for_hours(float(args.hours_to_complete))

    datasets = process.setdefault("datasets", [{}])
    # Some templates nest a single dataset dict; normalise to a list of dataset maps.
    if isinstance(datasets, dict):
        datasets = [datasets]
        process["datasets"] = datasets
    datasets[0]["folder_path"] = str(dataset_path)
    if control_paths:
        datasets[0]["control_path"] = [str(path) for path in control_paths]
    else:
        datasets[0].pop("control_path", None)

    # I4: eval scores 0.5 captioned + 0.5 empty prompt. Stock templates often use
    # ~0.05 caption dropout; raise toward EMPTY_PROMPT_RATE (default 0.50).
    apply_empty_prompt_share(process, empty_prompt_rate_from_env())
    # I2: denser checkpoints so promote_last_safetensors has a late-run choice.
    apply_checkpoint_save_policy(process)
    # I7/I9: family-specific LR / LoRA / res / step nudges aimed at lower eval loss.
    apply_family_loss_overrides(process, args.model_type, float(args.hours_to_complete))

    model_config = process.setdefault("model", {})
    model_path = cached_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected cached base model at {model_path}")
    # Trainer container has no internet; ai-toolkit must load every component from
    # the local cache dir. Its ideogram4 loader uses the local branch when
    # <name_or_path>/<subfolder> exists, so point it at the downloaded snapshot.
    model_config["name_or_path"] = str(model_path)

    if args.model_type == "ideogram4":
        text_encoder_path = IDEOGRAM4_TEXT_ENCODER_CACHE
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Expected cached Ideogram 4 text encoder at {text_encoder_path}"
            )
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)

    elif args.model_type == "krea2":
        text_encoder_path = KREA2_TEXT_ENCODER_CACHE
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Expected cached Krea 2 text encoder at {text_encoder_path}"
            )
        vae_path = model_path / "vae"
        if not vae_path.exists():
            raise FileNotFoundError(f"Expected cached Krea 2 VAE at {vae_path}")
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)
        # Krea2Model appends the "vae" subfolder when loading the VAE.
        model_kwargs["vae_path"] = str(model_path)

    CONFIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_OUTPUT_ROOT / f"{args.task_id}.yaml"
    with output_path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return output_path


def prepare_hidream_edit_config(dataset_path: Path, control_paths: list[Path], task_id: str) -> Path:
    if len(control_paths) != 1:
        raise ValueError("HiDream O1 edit training requires exactly one source image per target")
    CONFIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = Path("/dataset/cache") / task_id
    output_path = CONFIG_OUTPUT_ROOT / f"{task_id}_hidream_edit.toml"
    output_path.write_text(
        "\n".join(
            [
                "[general]",
                "resolution = [1024, 1024]",
                'caption_extension = ".txt"',
                "batch_size = 1",
                "enable_bucket = true",
                "bucket_no_upscale = false",
                "",
                "[[datasets]]",
                f'image_directory = "{dataset_path}"',
                f'control_directory = "{control_paths[0]}"',
                f'cache_directory = "{cache_path}"',
                "num_repeats = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def run_hidream_edit(args: argparse.Namespace, dataset_path: Path, control_paths: list[Path]) -> None:
    if not HIDREAM_DIT.is_file():
        raise FileNotFoundError(f"Expected cached HiDream O1 checkpoint at {HIDREAM_DIT}")
    dataset_config = prepare_hidream_edit_config(dataset_path, control_paths, args.task_id)
    output_dir = CHECKPOINTS_ROOT / args.task_id / args.expected_repo_name
    output_dir.mkdir(parents=True, exist_ok=True)
    common = ["--dataset_config", str(dataset_config), "--batch_size", "1"]
    subprocess.run(
        [
            MUSUBI_PYTHON,
            "-c",
            "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'Musubi torch cannot use CUDA')",
        ],
        check=True,
    )
    subprocess.run(
        [MUSUBI_PYTHON, "src/musubi_tuner/hidream_o1_cache_pixel.py", *common],
        cwd=MUSUBI_ROOT,
        env=OFFLINE_HUB_ENV,
        check=True,
    )
    subprocess.run(
        [
            MUSUBI_PYTHON,
            "src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py",
            *common,
            "--dit",
            str(HIDREAM_DIT),
            "--model_type",
            "full",
        ],
        cwd=MUSUBI_ROOT,
        env=OFFLINE_HUB_ENV,
        check=True,
    )
    train_command = [
            MUSUBI_ACCELERATE,
            "launch",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            "bf16",
            "src/musubi_tuner/hidream_o1_train_network.py",
            "--dit",
            str(HIDREAM_DIT),
            "--dataset_config",
            str(dataset_config),
            "--model_type",
            "full",
            "--task",
            "i2i",
            "--sdpa",
            "--mixed_precision",
            "bf16",
            "--timestep_sampling",
            "uniform",
            "--weighting_scheme",
            "none",
            "--noise_scale_start",
            "8.0",
            "--noise_scale_end",
            "8.0",
            "--noise_clip_std",
            "0.0",
            "--optimizer_type",
            "adamw8bit",
            "--learning_rate",
            "4e-5",
            "--gradient_checkpointing",
            "--network_module",
            "networks.lora_hidream_o1",
            "--network_dim",
            "32",
            "--network_alpha",
            "32",
            "--network_args",
            "conv_dim=4",
            "conv_alpha=1",
            "--save_every_n_epochs",
            os.environ.get("SAVE_EVERY_N_EPOCHS", "1"),
            "--seed",
            os.environ.get("SEED", "42"),
            "--output_dir",
            str(output_dir),
            "--output_name",
            args.expected_repo_name,
        ]
    if max_train_epochs := os.environ.get("MAX_TRAIN_EPOCHS"):
        train_command.extend(["--max_train_epochs", max_train_epochs])
    elif os.environ.get("MAX_TRAIN_STEPS"):
        train_command.extend(["--max_train_steps", os.environ["MAX_TRAIN_STEPS"]])
    else:
        # I5: scale HiDream edit steps from the task hour budget as well.
        train_command.extend(["--max_train_steps", str(steps_for_hours(float(args.hours_to_complete)))])
    subprocess.run(
        train_command,
        cwd=MUSUBI_ROOT,
        env=OFFLINE_HUB_ENV,
        check=True,
    )


def empty_prompt_rate_from_env() -> float:
    # Match eval's 0.5/0.5 captioned vs empty mix by default (I4).
    raw = os.environ.get("EMPTY_PROMPT_RATE", "0.50")
    rate = float(raw)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"EMPTY_PROMPT_RATE must be in [0, 1], got {rate}")
    return rate


def steps_for_hours(hours: float) -> int:
    """Map task hours to a train step budget (I5).

    Image tasks are typically 0.5–1.5h. Default ~1100 steps/hour, clamped.
    Override with STEPS_PER_HOUR / MIN_TRAIN_STEPS / MAX_TRAIN_STEPS.
    """
    if hours <= 0:
        raise ValueError(f"hours_to_complete must be > 0, got {hours}")
    per_hour = float(os.environ.get("STEPS_PER_HOUR", "1100"))
    minimum = int(os.environ.get("MIN_TRAIN_STEPS", "400"))
    maximum = int(os.environ.get("MAX_TRAIN_STEPS_CAP", "2800"))
    return max(minimum, min(maximum, int(hours * per_hour)))


def apply_empty_prompt_share(process: dict, rate: float) -> None:
    """Set caption_dropout_rate on every dataset block (I4)."""
    datasets = process.get("datasets", [])
    if isinstance(datasets, dict):
        datasets = [datasets]
        process["datasets"] = datasets
    for dataset in datasets:
        if isinstance(dataset, dict):
            dataset["caption_dropout_rate"] = rate


def apply_checkpoint_save_policy(process: dict) -> None:
    """Keep denser step checkpoints for later last.safetensors promotion (I2)."""
    save = process.setdefault("save", {})
    save["save_every"] = int(os.environ.get("SAVE_EVERY_STEPS", "50"))
    save["max_step_saves_to_keep"] = int(os.environ.get("MAX_STEP_SAVES", "12"))


# Live 2026-11-23 field: R1 Z-Image / Klein / MOSS; later Qwen + HiDream.
_R1_MODEL_TYPES = frozenset({"z-image", "flux2-klein-4b", "moss-image-base"})


def apply_r1_family_overrides(process: dict, model_type: str) -> None:
    """Backward-compatible wrapper → apply_family_loss_overrides(hours=1.0)."""
    apply_family_loss_overrides(process, model_type, hours=1.0)


def _drop_high_res(dataset: dict, drop_from: int = 1024) -> None:
    resolution = dataset.get("resolution")
    if isinstance(resolution, list) and drop_from in resolution and len(resolution) > 1:
        dataset["resolution"] = [value for value in resolution if value < drop_from] or resolution


def apply_family_loss_overrides(process: dict, model_type: str, hours: float) -> None:
    """I7/I9: per-family knobs aimed at lower validator loss (original, not a winner paste)."""
    save = process.setdefault("save", {})
    train = process.setdefault("train", {})
    network = process.setdefault("network", {})
    datasets = process.get("datasets", [])
    if isinstance(datasets, dict):
        datasets = [datasets]
        process["datasets"] = datasets

    train["gradient_checkpointing"] = True
    if os.environ.get("TRAIN_LR_OVERRIDE"):
        train["lr"] = float(os.environ["TRAIN_LR_OVERRIDE"])

    # Denser saves help every family; R1/R3 get even denser.
    save["save_every"] = min(int(save.get("save_every", 50)), 50)
    save["max_step_saves_to_keep"] = max(int(save.get("max_step_saves_to_keep", 12)), 12)

    # Mild step uplift within the hour budget for short tournament windows.
    if hours > 0 and "TRAIN_STEPS_OVERRIDE" not in os.environ:
        steps = int(train.get("steps") or steps_for_hours(hours))
        if model_type in _R1_MODEL_TYPES | {"qwen-image", "hidream-o1"}:
            train["steps"] = min(int(os.environ.get("MAX_TRAIN_STEPS_CAP", "2800")), int(steps * 1.08))

    if model_type == "z-image":
        if "TRAIN_LR_OVERRIDE" not in os.environ:
            train["lr"] = float(os.environ.get("ZIMAGE_LR", "6e-5"))
        network["linear"] = int(os.environ.get("ZIMAGE_LINEAR", "48"))
        network["linear_alpha"] = int(os.environ.get("ZIMAGE_LINEAR_ALPHA", "48"))
        network.setdefault("conv", 16)
        network.setdefault("conv_alpha", 16)
        for dataset in datasets:
            if isinstance(dataset, dict):
                _drop_high_res(dataset, 1024)

    elif model_type == "flux2-klein-4b":
        if "TRAIN_LR_OVERRIDE" not in os.environ:
            train["lr"] = float(os.environ.get("KLEIN_LR", "7e-5"))
        network["linear"] = int(os.environ.get("KLEIN_LINEAR", "40"))
        network["linear_alpha"] = int(os.environ.get("KLEIN_LINEAR_ALPHA", "40"))
        for dataset in datasets:
            if isinstance(dataset, dict):
                _drop_high_res(dataset, 1024)

    elif model_type == "moss-image-base":
        if "TRAIN_LR_OVERRIDE" not in os.environ:
            train["lr"] = float(os.environ.get("MOSS_LR", "6e-5"))
        network["linear"] = int(os.environ.get("MOSS_LINEAR", "40"))
        network["linear_alpha"] = int(os.environ.get("MOSS_LINEAR_ALPHA", "40"))
        for dataset in datasets:
            if isinstance(dataset, dict):
                _drop_high_res(dataset, 1024)

    elif model_type == "qwen-image":
        # Live R2 losses sit ~0.11; keep moderate LR + drop 1024 for stability.
        if "TRAIN_LR_OVERRIDE" not in os.environ:
            train["lr"] = float(os.environ.get("QWEN_LR", "7e-5"))
        network["linear"] = int(os.environ.get("QWEN_LINEAR", "48"))
        network["linear_alpha"] = int(os.environ.get("QWEN_LINEAR_ALPHA", "48"))
        train["train_text_encoder"] = False
        for dataset in datasets:
            if isinstance(dataset, dict):
                _drop_high_res(dataset, 1024)

    elif model_type == "hidream-o1":
        # Live R3 contender ~0.25; avoid huge res + keep conservative LR.
        if "TRAIN_LR_OVERRIDE" not in os.environ:
            train["lr"] = float(os.environ.get("HIDREAM_LR", "3e-5"))
        network["linear"] = int(os.environ.get("HIDREAM_LINEAR", "40"))
        network["linear_alpha"] = int(os.environ.get("HIDREAM_LINEAR_ALPHA", "40"))
        save["save_every"] = min(int(save.get("save_every", 50)), 40)
        for dataset in datasets:
            if isinstance(dataset, dict):
                _drop_high_res(dataset, 2048)
                _drop_high_res(dataset, 1024)

    elif model_type in _R1_MODEL_TYPES:
        if "TRAIN_LR_OVERRIDE" not in os.environ:
            train["lr"] = float(os.environ.get("R1_TRAIN_LR", "7e-5"))
        for dataset in datasets:
            if isinstance(dataset, dict):
                _drop_high_res(dataset, 1024)


def training_deadline_seconds(hours: float) -> float | None:
    """I5b: wall-clock budget as a fraction of the task hour window.

    Returns None when DEADLINE_FRACTION=0 (disabled). Default uses 92% of the
    assigned hours so mid-run checkpoints can still be promoted.
    """
    raw = os.environ.get("DEADLINE_FRACTION", "0.92")
    fraction = float(raw)
    if fraction <= 0:
        return None
    if hours <= 0:
        raise ValueError(f"hours_to_complete must be > 0, got {hours}")
    return max(60.0, hours * 3600.0 * fraction)


def looks_like_oom(text: str) -> bool:
    lowered = text.lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "cudamemalloc",
        "cuda error: out of memory",
        "resource exhausted",
        "torch.cuda.outofmemoryerror",
    )
    return any(needle in lowered for needle in needles)


def shrink_config_for_oom_retry(config_path: Path) -> None:
    """I6: rewrite prepared YAML once — lower res ladder + fewer steps."""
    with config_path.open() as file:
        config = yaml.safe_load(file)
    process = config["config"]["process"][0]
    train = process.setdefault("train", {})
    steps = int(train.get("steps", 500))
    train["steps"] = max(200, steps // 2)
    datasets = process.get("datasets", [])
    if isinstance(datasets, dict):
        datasets = [datasets]
        process["datasets"] = datasets
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        resolution = dataset.get("resolution")
        if isinstance(resolution, list) and resolution:
            # Keep only the lowest bucket on retry.
            dataset["resolution"] = [min(resolution)]
        elif isinstance(resolution, int):
            dataset["resolution"] = max(256, resolution // 2)
    with config_path.open("w") as file:
        yaml.safe_dump(config, file, default_flow_style=False)


def _training_folder_from_config(config_path: Path) -> Path | None:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    try:
        folder = config["config"]["process"][0].get("training_folder")
    except (KeyError, IndexError, TypeError):
        return None
    return Path(folder) if folder else None


def run_ai_toolkit(config_path: Path, hours: float | None = None) -> None:
    """Run AI Toolkit; OOM retry (I6) + soft deadline leave-partial (I5b)."""
    command = ["python3", "run.py", str(config_path)]
    timeout = training_deadline_seconds(hours) if hours is not None else None

    def _run_once(cfg: Path) -> None:
        try:
            subprocess.run(
                command if cfg == config_path else ["python3", "run.py", str(cfg)],
                cwd=AI_TOOLKIT_ROOT,
                env=OFFLINE_HUB_ENV,
                check=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as timed_out:
            # Soft deadline: keep going if any checkpoint landed for promote.
            folder = _training_folder_from_config(cfg)
            if folder and list_safetensors(folder):
                return
            raise RuntimeError(
                f"Training hit soft deadline ({timeout:.0f}s) with no .safetensors under {folder}"
            ) from timed_out

    try:
        _run_once(config_path)
        return
    except subprocess.CalledProcessError as first_error:
        allow = os.environ.get("ALLOW_OOM_RETRY", "1") != "0"
        soft_kill = first_error.returncode in {137, 143}
        detail = str(first_error)
        if first_error.stderr:
            detail = f"{detail}\n{first_error.stderr}"
        if not allow or (not soft_kill and not looks_like_oom(detail)):
            raise
        shrink_config_for_oom_retry(config_path)
        _run_once(config_path)


def list_safetensors(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.safetensors") if path.is_file())


def select_lora_for_eval(files: list[Path]) -> Path | None:
    """Pick an adapter to promote, then the evaluator will prefer last.safetensors.

    Prefer an existing unique last.safetensors. Otherwise prefer the highest
    trailing step number, but skip very early saves (< mid-run) when later
    checkpoints exist (I2 stronger heuristic; still original).
    """
    if not files:
        return None
    lasts = [path for path in files if path.name.lower() == "last.safetensors"]
    if len(lasts) == 1:
        return lasts[0]
    numbered: list[tuple[int, Path]] = []
    for path in files:
        match = _TRAILING_NUM.search(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        numbered.sort(key=lambda item: item[0])
        max_step = numbered[-1][0]
        # Prefer ~80% through the run: late enough to fit, less terminal overfit (I2).
        floor = max(1, int(max_step * 0.55))
        target = max(floor, int(max_step * 0.80))
        late = [item for item in numbered if item[0] >= floor]
        pool = late or numbered
        return min(pool, key=lambda item: (abs(item[0] - target), -item[0]))[1]
    if len(files) == 1:
        return files[0]
    return None


def promote_last_safetensors(output_dir: Path) -> Path | None:
    """Ensure output_dir/last.safetensors is the adapter the evaluator will score."""
    files = list_safetensors(output_dir)
    chosen = select_lora_for_eval(files)
    if chosen is None:
        return None
    target = output_dir / "last.safetensors"
    if chosen.resolve() == target.resolve():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen, target)
    return target


def main() -> None:
    args = parse_args()
    dataset_path, control_paths = prepare_dataset(args.task_id, args.training_mode)
    output_dir = CHECKPOINTS_ROOT / args.task_id / args.expected_repo_name
    if (args.model_type, args.training_mode) == ("hidream-o1", EDIT):
        run_hidream_edit(args, dataset_path, control_paths)
    else:
        config_path = prepare_config(args, dataset_path, control_paths)
        run_ai_toolkit(config_path, hours=float(args.hours_to_complete))
    promoted = promote_last_safetensors(output_dir)
    if promoted is None:
        raise FileNotFoundError(
            f"No .safetensors found under {output_dir} after training; nothing for the evaluator"
        )


if __name__ == "__main__":
    main()
