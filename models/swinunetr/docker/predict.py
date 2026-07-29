#!/usr/bin/env python
"""
BraTS-GoAT SwinUNETR-SSL inference -- Docker submission entrypoint.

Region-based sigmoid model: the network outputs 3 overlapping binary regions
[TC, WT, ET] (NOT a 4-class softmax). Decode matches the training notebook
(predict_goat_swinunetr_ssl): sigmoid -> 0.5 threshold -> per-region
post-processing -> nesting -> integer label map.

Reads cases from --input-dir (one subfolder per subject, BraTS naming) and
writes one label map per subject to --output-dir as <subject_id>.nii.gz in the
original image space of the input.

Output labels: 0=BG, 1=NCR, 2=ED, 3=ET.

Post-processing (only these, matching the notebook):
    1. TTA        -- 8-view flip averaging of the sigmoid probabilities
    2. CC filter  -- per region, drop components smaller than CC_MIN_VOXELS
    3. Closing    -- per region, binary closing (iter=1, 6-connectivity)
    4. Nesting    -- enforce ET subset TC subset WT, applied LAST
"""

import argparse
import re
import sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import nibabel as nib
from scipy.ndimage import (
    label as scipy_label,
    binary_closing,
    generate_binary_structure,
)

import torch
import torch.nn as nn

import monai
from monai.data import (
    Dataset, DataLoader, decollate_batch, pad_list_data_collate,
    set_track_meta, MetaTensor,
)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    CropForegroundd, NormalizeIntensityd, Invertd,
)
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

torch.backends.cudnn.benchmark = True
set_track_meta(True)  # required so Invertd can undo crop + orientation


# ============================================================================
# Configuration -- must match the training checkpoint exactly
# ============================================================================

DEFAULT_WEIGHTS = "/opt/model/model.pth"

# ---- Modalities (order matters -- matches training channel order) ----------
IMAGE_KEYS   = ["t1n", "t1c", "t2w", "t2f"]

# ---- SwinUNETR architecture ------------------------------------------------
IN_CHANNELS  = 4
OUT_CHANNELS = 3               # 3 sigmoid regions [TC, WT, ET]
REGION_NAMES = ["TC", "WT", "ET"]
FEATURE_SIZE = 48
SPATIAL_DIMS = 3

# ---- Sliding-window inference ----------------------------------------------
SW_ROI     = (96, 96, 96)
SW_BATCH   = 4
SW_OVERLAP = 0.7
SW_MODE    = "gaussian"
USE_AMP    = torch.cuda.is_available()
AMP_DTYPE  = torch.float16

# ---- TTA: 8-view flip (axes over (B,C,D,H,W) -> spatial axes 2,3,4) --------
FLIP_AXES = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]

# ---- Post-processing --------------------------------------------------------
CC_MIN_VOXELS = {"TC": 50, "WT": 100, "ET": 30}   # drop components smaller than this, per region
CLOSE_ITER    = 1
CLOSE_STRUCT  = generate_binary_structure(3, 1)    # 6-connectivity

# ---- DataLoader -------------------------------------------------------------
NUM_WORKERS = 0   # safest inside a container


# ============================================================================
# Device / AMP
# ============================================================================

NUM_GPUS = torch.cuda.device_count()
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx():
    if USE_AMP and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True)
    return nullcontext()


# ============================================================================
# Case discovery
# ============================================================================

def find_modality(case_dir: Path, key: str):
    hits = (sorted(case_dir.glob(f"*-{key}.nii.gz")) or
            sorted(case_dir.glob(f"*-{key}.nii")) or
            sorted(case_dir.glob(f"*{key}*.nii*")))
    return str(hits[0]) if hits else None


def discover_cases(input_dir: Path):
    dicts, skipped = [], []
    for d in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        mods = {k: find_modality(d, k) for k in IMAGE_KEYS}
        miss = [k for k, v in mods.items() if v is None]
        if miss:
            skipped.append((d.name, f"missing: {miss}"))
            continue
        dicts.append({**mods, "subject_id": d.name})
    return dicts, skipped


# ============================================================================
# Transforms  (crop on raw intensities FIRST, then z-score -- matches training)
# ============================================================================

def build_transforms():
    val_transforms = Compose([
        LoadImaged(keys=IMAGE_KEYS),
        EnsureChannelFirstd(keys=IMAGE_KEYS),
        Orientationd(keys=IMAGE_KEYS, axcodes="RAS"),
        CropForegroundd(keys=IMAGE_KEYS, source_key="t1c", allow_smaller=True),
        NormalizeIntensityd(keys=IMAGE_KEYS, nonzero=True, channel_wise=True),
    ])
    post_invert = Invertd(
        keys="pred", transform=val_transforms, orig_keys="t1c",
        meta_keys="pred_meta_dict", orig_meta_keys="t1c_meta_dict",
        meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True,
    )
    return val_transforms, post_invert


# ============================================================================
# Model
# ============================================================================

def _detect_out_channels(sd):
    for k, v in sd.items():
        if k.endswith("out.conv.conv.weight") and hasattr(v, "ndim") and v.ndim == 5:
            return int(v.shape[0])
    return None


def load_model(weights_path: Path):
    model = SwinUNETR(
        in_channels    = IN_CHANNELS,
        out_channels   = OUT_CHANNELS,
        feature_size   = FEATURE_SIZE,
        use_checkpoint = False,
        spatial_dims   = SPATIAL_DIMS,
    ).to(DEVICE)

    sd = torch.load(str(weights_path), map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.")    for k in sd):
        sd = {k[7:]:  v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[10:]: v for k, v in sd.items()}

    ckpt_oc = _detect_out_channels(sd)
    if ckpt_oc is not None and ckpt_oc != OUT_CHANNELS:
        sys.exit(
            f"\nFATAL: checkpoint output head has {ckpt_oc} channels but "
            f"OUT_CHANNELS is {OUT_CHANNELS}.\n"
            f"This script expects a 3-region sigmoid head [TC, WT, ET].\n"
        )

    model.load_state_dict(sd)   # strict -- keys match the training checkpoint
    model.eval()
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"[model] DataParallel across {NUM_GPUS} GPUs", flush=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"[weights] loaded ({n/1e6:.2f}M params)", flush=True)
    return model


# ============================================================================
# TTA (8-flip on sigmoid probabilities)
# ============================================================================

def tta_inference(model, image):
    """image: (1, 4, H, W, D) on CPU. Returns (3, H, W, D) mean sigmoid probs on CPU."""
    probs_sum = None
    with torch.no_grad():
        for ax in FLIP_AXES:
            xi = torch.flip(image, ax) if ax else image
            xi = xi.to(DEVICE)
            with amp_ctx():
                logits = sliding_window_inference(
                    xi, roi_size=SW_ROI, sw_batch_size=SW_BATCH,
                    predictor=model, overlap=SW_OVERLAP, mode=SW_MODE,
                )
            p = torch.sigmoid(logits.float()).cpu()
            if ax:
                p = torch.flip(p, ax)
            probs_sum = p if probs_sum is None else probs_sum + p
    return (probs_sum / len(FLIP_AXES))[0]


# ============================================================================
# Post-processing
# ============================================================================

def _remove_small_cc(mask, min_voxels):
    if min_voxels <= 0 or mask.sum() == 0:
        return mask
    out = mask.copy()
    labeled, n_comp = scipy_label(out)
    if n_comp == 0:
        return out
    sizes = np.bincount(labeled.ravel())
    for cid in range(1, n_comp + 1):
        if sizes[cid] < min_voxels:
            out[labeled == cid] = 0
    return out


def _close(mask):
    if mask.sum() == 0:
        return mask
    return binary_closing(mask, structure=CLOSE_STRUCT, iterations=CLOSE_ITER).astype(np.uint8)


def postprocess_regions(seg):
    """seg: (3, H, W, D) binary [TC, WT, ET]. CC -> closing -> nesting (last)."""
    out = seg.copy()
    for i, r in enumerate(REGION_NAMES):
        out[i] = _remove_small_cc(out[i], CC_MIN_VOXELS[r])
        out[i] = _close(out[i])
    out[0] = np.maximum(out[0], out[2])   # TC must contain ET
    out[1] = np.maximum(out[1], out[0])   # WT must contain TC
    return out


def regions_to_labels(seg):
    """(3, H, W, D) binary [TC, WT, ET] -> integer map 1=NCR, 2=ED, 3=ET."""
    out = np.zeros(seg.shape[1:], dtype=np.uint8)
    out[seg[1] > 0] = 2   # WT -> ED (default)
    out[seg[0] > 0] = 1   # TC -> NCR
    out[seg[2] > 0] = 3   # ET -> ET
    return out


# ============================================================================
# Save (invert to original space, using the input's own affine/header)
# ============================================================================

def save_prediction(d0, label_map, sid, post_invert, output_dir: Path):
    t1c_mt = d0["t1c"]
    pred_mt = MetaTensor(
        torch.as_tensor(label_map)[None].float(),
        meta=t1c_mt.meta.copy() if hasattr(t1c_mt, "meta") else {},
        applied_operations=list(t1c_mt.applied_operations)
            if hasattr(t1c_mt, "applied_operations") else [],
    )
    d0["pred"] = pred_mt
    inv = post_invert(d0)
    pred_inv = inv["pred"]
    pred_np  = pred_inv.cpu().numpy() if isinstance(pred_inv, torch.Tensor) else np.asarray(pred_inv)
    pred_orig = np.squeeze(pred_np).astype(np.uint8)

    ref_path = t1c_mt.meta.get("filename_or_obj", None) if hasattr(t1c_mt, "meta") else None
    orig_img = nib.load(ref_path)
    if pred_orig.shape != orig_img.shape[:3]:
        raise ValueError(f"shape mismatch after invert: pred={pred_orig.shape} "
                         f"orig={orig_img.shape[:3]}")

    header = orig_img.header.copy()
    header.set_data_dtype(np.uint8)
    out_path = output_dir / f"{sid}.nii.gz"
    nib.save(nib.Nifti1Image(pred_orig, orig_img.affine, header), str(out_path))
    return out_path, pred_orig


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="BraTS-GoAT SwinUNETR-SSL inference")
    ap.add_argument("--input-dir",  default="/input")
    ap.add_argument("--output-dir", default="/output")
    ap.add_argument("--weights",    default=DEFAULT_WEIGHTS)
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MONAI {monai.__version__} | PyTorch {torch.__version__} | "
          f"CUDA {torch.cuda.is_available()} | device {DEVICE}", flush=True)
    print(f"TTA=8-flip  ROI={SW_ROI}  overlap={SW_OVERLAP}  batch={SW_BATCH}  "
          f"CC={CC_MIN_VOXELS}  close_iter={CLOSE_ITER}  nesting=on", flush=True)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"FATAL: weights not found at {weights_path}")

    cases, skipped = discover_cases(input_dir)
    print(f"[data] {len(cases)} case(s) found under {input_dir}", flush=True)
    for sid, reason in skipped:
        print(f"[data] skipped {sid}: {reason}", flush=True)
    if not cases:
        sys.exit(f"FATAL: no valid cases under {input_dir}")

    val_transforms, post_invert = build_transforms()
    loader = DataLoader(
        Dataset(cases, val_transforms),
        batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=pad_list_data_collate,
    )

    model = load_model(weights_path)

    n_done, failures = 0, []
    for idx, batch in enumerate(loader, 1):
        d0  = decollate_batch(batch)[0]
        sid = d0["subject_id"]
        try:
            vi = torch.cat([d0[k].unsqueeze(0) for k in IMAGE_KEYS], dim=1)  # (1,4,H,W,D) CPU

            probs     = tta_inference(model, vi)                    # (3,H,W,D) CPU
            raw_seg   = (probs > 0.5).numpy().astype(np.uint8)      # (3,H,W,D) cropped space
            pp_seg    = postprocess_regions(raw_seg)
            label_map = regions_to_labels(pp_seg)                   # (H,W,D) cropped space

            out_path, pred_orig = save_prediction(d0, label_map, sid, post_invert, output_dir)
            vox = {r: int((pred_orig == v).sum()) for r, v in zip(["NCR", "ED", "ET"], [1, 2, 3])}
            n_done += 1
            print(f"[{n_done}/{len(cases)}] {sid}  NCR={vox['NCR']} ED={vox['ED']} "
                  f"ET={vox['ET']}  -> {out_path.name}", flush=True)
        except Exception as e:
            failures.append((sid, str(e)))
            print(f"[{idx}/{len(cases)}] {sid}  FAILED: {e}", flush=True)

    print(f"[done] wrote {n_done}/{len(cases)} segmentation(s) to {output_dir}", flush=True)
    if failures:
        print(f"[done] {len(failures)} failure(s):", flush=True)
        for sid, err in failures:
            print(f"  {sid}: {err}", flush=True)


if __name__ == "__main__":
    main()
