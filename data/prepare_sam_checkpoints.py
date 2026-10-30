"""
Download script for SAM and MedSAM pretrained checkpoints.

This script is intended to be placed inside the `dataset/` folder.
It automatically detects the project root (parent of `dataset/`)
and saves checkpoints in `<project_root>/cp/`.

Usage:
    python dataset/prepare_sam_checkpoints.py --model sam_vit_b
    python dataset/prepare_sam_checkpoints.py --model sam_vit_h
    python dataset/prepare_sam_checkpoints.py --model medsam_vit_b
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


# ============================================================
# Automatic project root detection
# ============================================================
# This file is located at: <project_root>/dataset/download_checkpoints.py
# So: parent = <project_root>/dataset
#     parent.parent = <project_root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / 'cp'
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model registry
# ============================================================
MODEL_INFO = {
    'sam_vit_b': {
        'filename': 'sam_vit_b.pth',
        'url': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        'min_size_mb': 350,
    },
    'sam_vit_h': {
        'filename': 'sam_vit_h.pth',
        'url': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'min_size_mb': 2300,
    },
    'medsam_vit_b': {
        'filename': 'medsam_vit_b.pth',
        # Validated Google Drive direct link (requires gdown)
        'url': "https://drive.google.com/uc?id=1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_",
        'min_size_mb': 350,
    },
}


# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(
    description="Download SAM / MedSAM pretrained checkpoints."
)
parser.add_argument(
    '--model',
    type=str,
    required=True,
    choices=list(MODEL_INFO.keys()),
    help='Model identifier to download (sam_vit_b, sam_vit_h, medsam_vit_b)',
)
args = parser.parse_args()


# ============================================================
# Resolve target path
# ============================================================
model_data = MODEL_INFO[args.model]
CKPT_PATH = CKPT_DIR / model_data['filename']
MIN_SIZE_BYTES = model_data['min_size_mb'] * 1_000_000


print("=" * 70)
print(f" [INFO] Project root    : {PROJECT_ROOT}")
print(f" [INFO] Checkpoint dir  : {CKPT_DIR}")
print(f" [INFO] Target model    : {args.model}")
print(f" [INFO] Target filename : {model_data['filename']}")
print(f" [INFO] Target path     : {CKPT_PATH}")
print("=" * 70)


# ============================================================
# Download logic
# ============================================================
if CKPT_PATH.exists() and CKPT_PATH.stat().st_size > MIN_SIZE_BYTES:
    size_mb = CKPT_PATH.stat().st_size / 1e6
    print(f" [OK] File already exists ({size_mb:.2f} MB). Skipping download.")
else:
    print(f" [INFO] Downloading from: {model_data['url']}")
    print(f"        Expected size: ~{model_data['min_size_mb']} MB")

    # Attempt 1: gdown (best for Google Drive + supports resume)
    try:
        import gdown
        gdown.download(model_data['url'], str(CKPT_PATH), quiet=False, fuzzy=True)
    except ImportError:
        print(" [WARN] 'gdown' not installed. Falling back to wget.")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(CKPT_PATH), model_data['url']],
            check=True,
        )
    except Exception as e:
        print(f" [WARN] gdown failed ({e}). Falling back to wget.")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(CKPT_PATH), model_data['url']],
            check=True,
        )

    # ============================================================
    # Validation
    # ============================================================
    if CKPT_PATH.exists():
        size_mb = CKPT_PATH.stat().st_size / 1e6
        print(f" [OK] Downloaded: {CKPT_PATH} ({size_mb:.2f} MB)")
        if size_mb < model_data['min_size_mb']:
            print(f" [ERROR] File too small ({size_mb:.2f} MB). Download likely corrupted.")
            sys.exit(1)
    else:
        print(" [ERROR] Download failed — file does not exist.")
        sys.exit(1)


print("=" * 70)
print(f" [OK] Final status: ready at {CKPT_PATH}")
print(f" [OK] Size        : {CKPT_PATH.stat().st_size / 1e6:.2f} MB")
print("=" * 70)