"""SAM3 image inference: box (+ optional text) prompt -> best mask.

Modeled on voxa/scripts/dry_sam3/sam3_common.py. Imports sam3 lazily so the rest
of the app can be imported without CUDA. The SAM3 image processor's public API
supports text prompts (set_text_prompt) and box prompts (add_geometric_prompt);
there is no click-point path, so this MVP drives it with a 2D box.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

BPE_PATH = "/home/hendrik/anaconda3/lib/python3.12/site-packages/clip/bpe_simple_vocab_16e6.txt.gz"


def build_processor(device: str = "cuda", confidence_threshold: float = 0.25):
    """Load the SAM3 image model + processor (weights from HuggingFace)."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=device, load_from_HF=True, bpe_path=BPE_PATH)
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def segment_box(proc, pil_img: Image.Image, box_norm, text: str | None = None):
    """Run SAM3 with a positive box (+ optional text). Return (mask_bool[H,W], score).

    box_norm: [cx, cy, w, h] normalized to [0,1] (SAM3's expected box format).
    Returns the highest-scoring mask, or an all-False mask + 0.0 if none found.
    """
    W, H = pil_img.size
    state = proc.set_image(pil_img)
    if text:
        state = proc.set_text_prompt(prompt=text, state=state)
    state = proc.add_geometric_prompt(box=list(box_norm), label=True, state=state)

    masks = state["masks"]   # [N,1,H,W] bool
    scores = state["scores"]  # [N]
    if masks is None or masks.numel() == 0:
        return np.zeros((H, W), dtype=bool), 0.0

    masks = masks.squeeze(1).cpu().numpy().astype(bool)  # [N,H,W]
    scores = scores.cpu().numpy().astype(np.float32)
    best = int(np.argmax(scores))
    return masks[best], float(scores[best])


def _masks_scores(state):
    m = state["masks"]
    if m is None or m.shape[0] == 0:
        return np.zeros((0, 0, 0), bool), np.zeros((0,), np.float32)
    return (m.squeeze(1).cpu().numpy().astype(bool),
            state["scores"].cpu().numpy().astype(np.float32))


def segment_concept(proc, pil_img: Image.Image, text: str, min_score: float = 0.3):
    """SAM3 concept mode: one forward pass returns ALL instances of `text`.
    Returns [(mask_bool[H,W], score), ...] above min_score, highest score first."""
    state = proc.set_image(pil_img)
    state = proc.set_text_prompt(prompt=text, state=state)
    masks, scores = _masks_scores(state)
    out = [(masks[k], float(scores[k])) for k in range(masks.shape[0]) if scores[k] >= min_score]
    out.sort(key=lambda ms: ms[1], reverse=True)
    return out
