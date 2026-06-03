"""Shared SAM3 helpers for the dry_sam3 experiments.

The torch/SAM3-specific pieces (model builder, single-prompt inference, per-class
mask union) plus the oriented PLY loader, factored out of the individual
capability scripts so they share one definition. The pure camera/projection math
is NOT here — it lives in the canonical home ``backend/scenes/reproject.py`` and
is imported below, so the dry-run scripts and the production registration check
project identically.

Run from the anaconda base where ``sam3`` + torch are installed (``build_processor``
imports them lazily). ``load_ply``/``union_mask`` are torch-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

# Canonical camera math lives in the voxa backend (pure numpy, no torch).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from scenes.reproject import ORIENTATION_PRESETS, euler_xyz_matrix  # noqa: E402


def build_processor(device: str = "cuda", confidence_threshold: float = 0.25):
    """Load the SAM3 image model + processor. Imports sam3 lazily (anaconda)."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    bpe = "/home/hendrik/anaconda3/lib/python3.12/site-packages/clip/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(device=device, load_from_HF=True, bpe_path=bpe)
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def segment(proc, pil_img: Image.Image, prompt: str):
    """Return (masks_bool [N,H,W], scores [N]) for one text prompt."""
    state = proc.set_image(pil_img)
    state = proc.set_text_prompt(prompt=prompt, state=state)
    masks = state["masks"]   # [N,1,H,W] bool
    scores = state["scores"]  # [N]
    if masks.numel() == 0:
        return (np.zeros((0, pil_img.height, pil_img.width), dtype=bool),
                np.zeros((0,), dtype=np.float32))
    return masks.squeeze(1).cpu().numpy().astype(bool), scores.cpu().numpy().astype(np.float32)


def union_mask(masks: np.ndarray, scores: np.ndarray):
    """Per-pixel best score across instances for one class."""
    if masks.shape[0] == 0:
        return None
    H, W = masks.shape[1], masks.shape[2]
    best = np.zeros((H, W), dtype=np.float32)
    for m, s in zip(masks, scores):
        best = np.maximum(best, m.astype(np.float32) * float(s))
    return best


def load_ply(path: Path, orientation: str = "Z+"):
    """Read xyz+rgb from a PLY and rotate by an orientation preset (default Z-up)."""
    p = PlyData.read(str(path))
    v = p["vertex"].data
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.uint8)
    R = euler_xyz_matrix(*ORIENTATION_PRESETS[orientation])
    return pts @ R.T, rgb
