"""
Inference script for ECCV 2016 colorization.

Usage:
    python inference.py --input photo.jpg
    python inference.py --input images/  --output results/
    python inference.py --input photo.jpg --no-gpu
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color
from skimage.transform import resize

from model import ECCVColorizer

# ---------------------------------------------------------------------------
WEIGHTS_URL  = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
WEIGHTS_PATH = Path("checkpoints/eccv16.pth")
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ---------------------------------------------------------------------------


def download_weights(dest: Path = WEIGHTS_PATH) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Weights found at {dest}")
        return

    print(f"Downloading pretrained weights …")

    def _progress(count, block, total):
        pct = min(int(count * block * 100 / total), 100) if total > 0 else 0
        print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(WEIGHTS_URL, dest, _progress)
    print(f"\r  Saved to {dest}   ")


def load_model(weights_path: Path, device: torch.device) -> ECCVColorizer:
    model = ECCVColorizer()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Pre / post processing
# ---------------------------------------------------------------------------

def preprocess(img_path: Path):
    """
    Returns:
        tens_l_orig : (1, 1, H, W)  — original-resolution L channel
        tens_l_256  : (1, 1, 256, 256) — resized L fed to the model
    """
    img_rgb = np.array(Image.open(img_path).convert("RGB")) / 255.0
    img_lab = color.rgb2lab(img_rgb)          # L: [0,100], a*b*: [-128, 127]
    img_l   = img_lab[:, :, 0]               # (H, W)

    img_l_256 = resize(img_l, (256, 256), anti_aliasing=True)

    tens_l_orig = torch.from_numpy(img_l).float()[None, None]       # (1,1,H,W)
    tens_l_256  = torch.from_numpy(img_l_256).float()[None, None]   # (1,1,256,256)
    return tens_l_orig, tens_l_256


def postprocess(tens_l_orig: torch.Tensor, out_ab: torch.Tensor) -> np.ndarray:
    """
    Upsample predicted a*b* to original L size, merge, convert to uint8 RGB.
    """
    H, W = tens_l_orig.shape[2:]
    out_ab_up = F.interpolate(out_ab, size=(H, W), mode="bilinear", align_corners=False)

    lab = torch.cat([tens_l_orig, out_ab_up], dim=1)   # (1, 3, H, W)
    lab_np = lab[0].permute(1, 2, 0).cpu().numpy()     # (H, W, 3)

    rgb = color.lab2rgb(lab_np)                         # [0, 1]
    return (rgb * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def colorize_image(img_path: Path, model: ECCVColorizer, device: torch.device) -> np.ndarray:
    tens_l_orig, tens_l_256 = preprocess(img_path)
    with torch.no_grad():
        out_ab = model(tens_l_256.to(device))
    return postprocess(tens_l_orig, out_ab.cpu())


def main():
    parser = argparse.ArgumentParser(description="ECCV 2016 Colorization — Inference")
    parser.add_argument("--input",   required=True,               help="Input image or directory")
    parser.add_argument("--output",  default="output",            help="Output directory (default: output/)")
    parser.add_argument("--weights", default=str(WEIGHTS_PATH),   help="Path to .pth weights")
    parser.add_argument("--no-gpu",  action="store_true",         help="Force CPU")
    args = parser.parse_args()

    # Device
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        if args.no_gpu:
            print("Running on CPU (--no-gpu)")
        else:
            print("CUDA not available, running on CPU")
    else:
        device = torch.device("cuda")
        print(f"Running on {torch.cuda.get_device_name(0)}")

    # Weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        if weights_path == WEIGHTS_PATH:
            download_weights()
        else:
            print(f"Error: weights not found at {weights_path}", file=sys.stderr)
            sys.exit(1)

    # Model
    model = load_model(weights_path, device)
    print("Model loaded.\n")

    # Input images
    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"No images found in {input_path}")
            sys.exit(0)
    else:
        images = [input_path]

    # Output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        print(f"Colorizing  {img_path.name} …", end=" ", flush=True)
        result = colorize_image(img_path, model, device)
        out_path = out_dir / img_path.name
        Image.fromarray(result).save(out_path)
        print(f"→ {out_path}")

    print("\nFinished.")


if __name__ == "__main__":
    main()
