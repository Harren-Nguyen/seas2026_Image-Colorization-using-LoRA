"""
Inference with base ECCV 2016 colorizer + LoRA adapter weights.

Usage:
    python lora_inference.py --input photo.jpg
    python lora_inference.py --input images/  --output results_lora/
    python lora_inference.py --input photo.jpg --no-gpu
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from model import ECCVColorizer
from inference import (
    download_weights,
    postprocess,
    preprocess,
    WEIGHTS_PATH,
    IMG_EXTS,
)
from lora_train import inject_lora, LORA_SAVE_PATH

# ---------------------------------------------------------------------------


def load_model_with_lora(
    weights_path: Path,
    lora_path: Path,
    device: torch.device,
    rank: int = 4,
    alpha: float = 1.0,
) -> ECCVColorizer:
    # 1. Load frozen base model
    base_model = ECCVColorizer()
    state = torch.load(weights_path, map_location=device)
    base_model.load_state_dict(state)

    for p in base_model.parameters():
        p.requires_grad_(False)

    # 2. Inject LoRA wrappers (same architecture as training)
    model = inject_lora(base_model, rank=rank, alpha=alpha)

    # 3. Load and apply LoRA weights on top
    lora_state = torch.load(lora_path, map_location=device)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)

    if unexpected:
        print(f"Warning: unexpected keys in LoRA checkpoint: {unexpected}", file=sys.stderr)

    return model.to(device).eval()


# ---------------------------------------------------------------------------

def colorize_image(img_path: Path, model: ECCVColorizer, device: torch.device):
    tens_l_orig, tens_l_256 = preprocess(img_path)
    with torch.no_grad():
        out_ab = model(tens_l_256.to(device))
    return postprocess(tens_l_orig, out_ab.cpu())


def main():
    parser = argparse.ArgumentParser(
        description="ECCV16 Colorization with LoRA — Inference"
    )
    parser.add_argument("--input",      required=True,
                        help="Input image or directory")
    parser.add_argument("--output",     default="output_lora",
                        help="Output directory (default: output_lora/)")
    parser.add_argument("--weights",    default=str(WEIGHTS_PATH),
                        help="Path to base model .pth weights")
    parser.add_argument("--lora",       default=str(LORA_SAVE_PATH),
                        help="Path to LoRA weights .pth")
    parser.add_argument("--rank",       type=int,   default=4,
                        help="LoRA rank (must match what was used in training)")
    parser.add_argument("--alpha",      type=float, default=1.0,
                        help="LoRA alpha (must match training)")
    parser.add_argument("--no-gpu",     action="store_true")
    args = parser.parse_args()

    # ---- Device ----
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Running on CPU")
    else:
        device = torch.device("cuda")
        print(f"Running on {torch.cuda.get_device_name(0)}")

    # ---- Weights ----
    weights_path = Path(args.weights)
    if not weights_path.exists():
        if weights_path == WEIGHTS_PATH:
            download_weights()
        else:
            print(f"Error: base weights not found at {weights_path}", file=sys.stderr)
            sys.exit(1)

    lora_path = Path(args.lora)
    if not lora_path.exists():
        print(f"Error: LoRA weights not found at {lora_path}\n"
              f"Run lora_train.py first.", file=sys.stderr)
        sys.exit(1)

    # ---- Model ----
    model = load_model_with_lora(
        weights_path, lora_path, device,
        rank=args.rank, alpha=args.alpha
    )
    print(f"Base model + LoRA loaded.  (rank={args.rank}, alpha={args.alpha})\n")

    # ---- Input images ----
    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir()
                        if p.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"No images found in {input_path}")
            sys.exit(0)
    else:
        images = [input_path]

    # ---- Output directory ----
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        print(f"Colorizing  {img_path.name} …", end=" ", flush=True)
        result   = colorize_image(img_path, model, device)
        out_path = out_dir / img_path.name
        Image.fromarray(result).save(out_path)
        print(f"→ {out_path}")

    print("\nFinished.")


if __name__ == "__main__":
    main()