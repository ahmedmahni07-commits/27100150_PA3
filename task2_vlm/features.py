"""Cache frozen SigLIP patch features once (FP16), then free the vision encoder before training Qwen.

    python -m task2_vlm.features --config configs/task2.yaml

Features: SiglipVisionModel(...).last_hidden_state – the 196 final patch tokens in 14x14 raster order, with
SigLIP's final (post) layer norm applied exactly once inside the HF model (we do not re-apply it).
Preprocessing is the deterministic SiglipImageProcessor (RGB, bicubic resize to 224, rescale, normalise 0.5/0.5).
Output: outputs/task2/features.pt  {"feats": [N,196,768] fp16, "index": {image_id: row}} and a timing log.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common.utils import ensure_dir, load_cfg, peak_mem_gb, reset_peak_mem, save_json
from task2_vlm.data import all_images


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ensure_dir(cfg["paths"]["outputs"])
    ims = [i for _, i in all_images(cfg)]
    root = Path(cfg["data"]["image_dir"])
    proc = SiglipImageProcessor.from_pretrained(cfg["vision"])
    reset_peak_mem()
    t0 = time.perf_counter()
    enc = SiglipVisionModel.from_pretrained(cfg["vision"], dtype=torch.float16).to(device).eval()
    feats = torch.empty(len(ims), 196, 768, dtype=torch.float16)
    t_load = time.perf_counter() - t0
    args_batch = args.batch
    for s in range(0, len(ims), args_batch):
        batch = [Image.open(root / f"{i}.jpg").convert("RGB") for i in ims[s:s + args_batch]]
        px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        h = enc(pixel_values=px).last_hidden_state
        assert h.shape[1:] == (196, 768), h.shape
        feats[s:s + len(batch)] = h.cpu()
        if s % (args_batch * 20) == 0:
            print(s, len(ims), flush=True)
    total = time.perf_counter() - t0
    torch.save({"feats": feats, "index": {i: r for r, i in enumerate(ims)}}, out / "features.pt")
    save_json({"n_images": len(ims), "seconds_total": total, "seconds_model_load": t_load,
               "peak_gpu_mem_gb": peak_mem_gb(), "feature_file_mb": (out / "features.pt").stat().st_size / 1024**2,
               "dtype": "float16", "post_layernorm": "applied once (HF SiglipVisionTransformer.post_layernorm)"},
              ensure_dir(cfg["paths"]["results"]) / "feature_caching.json")
    del enc
    torch.cuda.empty_cache() if device == "cuda" else None


if __name__ == "__main__":
    main()
