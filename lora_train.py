"""
LoRA fine-tuning for ECCV 2016 colorization model.

Usage:
    python lora_train.py --train-dir /path/to/your/images/
    python lora_train.py --train-dir /path/to/your/images/ --epochs 10 --lr 1e-4
    python lora_train.py --train-dir /path/to/your/images/ --no-gpu
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import color
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader

from model import ECCVColorizer
from inference import download_weights, WEIGHTS_PATH

# ---------------------------------------------------------------------------
LORA_SAVE_PATH = Path("checkpoints/lora_weights.pth")
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRAConv2d(nn.Module):
    """
    Wraps an existing frozen Conv2d with a low-rank LoRA branch:
        output = W(x) + scale * (B @ A)(x)
    where A: (rank, in_ch, kH, kW)  B: (out_ch, rank, 1, 1)
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.conv  = conv                      # frozen base weight
        self.rank  = rank
        self.scale = alpha / rank

        in_ch  = conv.in_channels
        out_ch = conv.out_channels
        kH, kW = conv.kernel_size

        # A captures spatial/channel mixing; B projects up to out_ch
        self.lora_A = nn.Parameter(torch.empty(rank, in_ch, kH, kW))
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank, 1, 1))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze base conv
        for p in self.conv.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)

        # Low-rank branch: first convolve with A (same stride/pad as base conv)
        lora_x = F.conv2d(x, self.lora_A,
                           stride=self.conv.stride,
                           padding=self.conv.padding,
                           dilation=self.conv.dilation,
                           groups=self.conv.groups)
        # Then project up with B (1×1 conv, no padding needed)
        lora_x = F.conv2d(lora_x, self.lora_B)

        return base + self.scale * lora_x


def inject_lora(model: ECCVColorizer, rank: int = 4, alpha: float = 1.0) -> ECCVColorizer:
    """Replace every Conv2d in the model with a LoRAConv2d wrapper."""
    for name, module in list(model.named_modules()):
        # Navigate to parent so we can setattr
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        if isinstance(module, nn.Conv2d):
            setattr(parent, child_name, LoRAConv2d(module, rank=rank, alpha=alpha))

    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters (lora_A / lora_B) from the model."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ColorizationDataset(Dataset):
    """
    Loads RGB images, converts to Lab, and returns:
        L channel (1, 256, 256)  — model input
        ab channels (2, 256, 256) — regression target
    """

    def __init__(self, img_dir: Path):
        self.paths = sorted(p for p in img_dir.iterdir()
                            if p.suffix.lower() in IMG_EXTS)
        if not self.paths:
            raise ValueError(f"No images found in {img_dir}")
        print(f"Found {len(self.paths)} training images in {img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_rgb = np.array(Image.open(self.paths[idx]).convert("RGB")) / 255.0
        img_lab = color.rgb2lab(img_rgb)   # L:[0,100]  ab:[-128,127]

        img_l  = resize(img_lab[:, :, 0],  (256, 256), anti_aliasing=True)
        img_ab = resize(img_lab[:, :, 1:], (256, 256), anti_aliasing=True)

        tens_l  = torch.from_numpy(img_l).float().unsqueeze(0)          # (1,256,256)
        tens_ab = torch.from_numpy(img_ab).float().permute(2, 0, 1)     # (2,256,256)

        return tens_l, tens_ab


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Device ----
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Running on CPU")
    else:
        device = torch.device("cuda")
        print(f"Running on {torch.cuda.get_device_name(0)}")

    # ---- Base model ----
    weights_path = Path(args.weights)
    if not weights_path.exists():
        download_weights(weights_path)

    base_model = ECCVColorizer()
    state = torch.load(weights_path, map_location=device)
    base_model.load_state_dict(state)

    # Freeze everything first
    for p in base_model.parameters():
        p.requires_grad_(False)

    # Inject LoRA adapters (unfreezes only lora_A / lora_B)
    model = inject_lora(base_model, rank=args.rank, alpha=args.alpha).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  "
          f"({100 * trainable / total:.2f}%)\n")

    # ---- Data ----
    # TODO: Change --train-dir to your image folder when running
    train_dir = Path(args.train_dir)
    dataset   = ColorizationDataset(train_dir)
    loader    = DataLoader(dataset,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=args.num_workers,
                           pin_memory=(device.type == "cuda"))

    # ---- Optimizer & loss ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.MSELoss()

    # ---- Train ----
    save_path = Path(args.lora_save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, (tens_l, tens_ab) in enumerate(loader, 1):
            tens_l  = tens_l.to(device)
            tens_ab = tens_ab.to(device)

            optimizer.zero_grad()
            pred_ab = model(tens_l)
            target_ab = F.interpolate(tens_ab, size=pred_ab.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(pred_ab, target_ab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % max(1, len(loader) // 5) == 0:
                print(f"  Epoch {epoch}/{args.epochs}  "
                      f"step {step}/{len(loader)}  "
                      f"loss={loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}\n")

    # ---- Save LoRA weights only ----
    torch.save(lora_state_dict(model), save_path)
    print(f"LoRA weights saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for ECCV16 Colorizer")

    # TODO: set your training image directory here via --train-dir
    parser.add_argument("--train-dir",    required=True,
                        help="Directory of RGB training images (placeholder — change to your path)")
    parser.add_argument("--weights",      default=str(WEIGHTS_PATH),
                        help="Path to base model .pth weights")
    parser.add_argument("--lora-save",    default=str(LORA_SAVE_PATH),
                        help="Where to save LoRA weights")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--rank",         type=int,   default=9,
                        help="LoRA rank (higher = more capacity, more params)")
    parser.add_argument("--alpha",        type=float, default=9.0,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--no-gpu",       action="store_true")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()