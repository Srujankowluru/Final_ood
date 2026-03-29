"""
Prepare WildDash2 for OOD evaluation: build image list and OOD masks.

- Reads WildDash panoptic (panoptic.json + panoptic/*.png).
- Default: ID/OOD/ignore from JSON mapping (WildDash category_id -> 0=ID, 1=OOD, -1=ignore).
  Recommended: --mapping-json .../prashanth/process_datasets/id_ood_mapping.json --mapping-key wilddash
- Legacy: --legacy-cityscapes-19 uses only the 19 Cityscapes-aligned WildDash category_ids (no JSON).
- Writes: image_list, ood_masks/<stem>.png with 0=ID, 255=OOD, 128=ignore (eval skips 128).

Usage:
  python prepare_wilddash_ood_masks.py --wilddash-root /fastdata/groupL/datasets/wilddash \\
    --mapping-json /path/to/id_ood_mapping.json --out-mask-dir ./wilddash_ood_masks_json
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Legacy only: WildDash panoptic category_ids aligned with Cityscapes-19.
CITYSCAPES_19_LABEL_IDS = {
    7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33
}

DEFAULT_MAPPING_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "mapping",
    "id_ood_mapping.json",
)
IGNORE_MASK_VALUE = 128  # must match run_all_methods.mask_image_to_ood_labels


def load_panoptic_meta(panoptic_json_path):
    """Load panoptic.json and return file_name -> {segment_id: category_id} per image."""
    with open(panoptic_json_path, "r") as f:
        data = json.load(f)
    images = {im["id"]: im["file_name"] for im in data["images"]}
    seg_map = {}
    for ann in data["annotations"]:
        file_name = images.get(ann["image_id"])
        if file_name is None:
            continue
        seg_map[file_name] = {seg["id"]: seg["category_id"] for seg in ann["segments_info"]}
    return seg_map


def decode_panoptic_segment_id(rgb):
    """Decode segment ID from RGB panoptic PNG (COCO-style: R + G*256 + B*256^2)."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return r.astype(np.int32) + g.astype(np.int32) * 256 + b.astype(np.int32) * 256 * 256


def load_id_ood_mapping(path: str, key: str):
    with open(path, "r") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"Key '{key}' not in {path}; have {list(data.keys())}")
    return {int(k): int(v) for k, v in data[key].items()}


def category_flags_from_mapping(cat_ids, mapping):
    """Map panoptic category_id per pixel to -1/0/1; None or unknown category -> OOD (1)."""

    def flag(c):
        if c is None:
            return 1
        try:
            cid = int(c)
        except (TypeError, ValueError):
            return 1
        return mapping.get(cid, 1)

    flags = np.vectorize(flag, otypes=[np.int32])(cat_ids)
    out = np.where(
        flags == 0,
        np.uint8(0),
        np.where(flags == -1, np.uint8(IGNORE_MASK_VALUE), np.uint8(255)),
    )
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Prepare WildDash image list and OOD masks")
    parser.add_argument(
        "--wilddash-root",
        type=str,
        default="/fastdata/groupL/datasets/wilddash",
        help="Root of WildDash2 dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "training", "all"],
        help="Which split to use (validation.txt / training.txt / all stems with panoptic)",
    )
    parser.add_argument(
        "--out-list",
        type=str,
        default="wilddash_val_list.txt",
        help="Output path for image list (one path per line)",
    )
    parser.add_argument(
        "--out-mask-dir",
        type=str,
        default=None,
        help="Output directory for OOD masks (default: <wilddash_root>/ood_masks)",
    )
    parser.add_argument("--no-masks", action="store_true", help="Only write image list, do not generate masks")
    parser.add_argument(
        "--mapping-json",
        type=str,
        default=None,
        help=f"JSON with ID/OOD flags (default: {DEFAULT_MAPPING_JSON} unless --legacy-cityscapes-19)",
    )
    parser.add_argument(
        "--mapping-key",
        type=str,
        default="wilddash",
        help="Key inside JSON (default: wilddash)",
    )
    parser.add_argument(
        "--legacy-cityscapes-19",
        action="store_true",
        help="Use hardcoded Cityscapes-19 WildDash category_ids instead of JSON mapping",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.wilddash_root)
    images_dir = os.path.join(root, "images")
    panoptic_dir = os.path.join(root, "panoptic")
    panoptic_json = os.path.join(root, "panoptic.json")

    if args.out_mask_dir is None:
        args.out_mask_dir = os.path.join(root, "ood_masks")
    out_mask_dir = os.path.abspath(args.out_mask_dir)

    if not os.path.isdir(images_dir) or not os.path.isdir(panoptic_dir) or not os.path.isfile(panoptic_json):
        raise FileNotFoundError(
            f"WildDash structure not found under {root}. Need images/, panoptic/, panoptic.json"
        )

    # Stems to process
    if args.split == "all":
        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(panoptic_dir)
            if f.endswith(".png")
        )
    else:
        split_file = os.path.join(root, "random_split", f"{args.split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            stems = [line.strip() for line in f if line.strip()]

    # Load panoptic meta first so we only include images that have annotations
    print("Loading panoptic.json...")
    seg_map = load_panoptic_meta(panoptic_json)
    # key in seg_map is file_name e.g. "us0035_100000.jpg"

    # Restrict to images that have both image, panoptic PNG, and panoptic annotation
    stems_with_both = []
    for stem in stems:
        fn = stem + ".jpg"
        if (
            os.path.isfile(os.path.join(images_dir, fn))
            and os.path.isfile(os.path.join(panoptic_dir, stem + ".png"))
            and fn in seg_map
        ):
            stems_with_both.append(stem)
    print(f"Using {len(stems_with_both)} images (with image, panoptic, and annotation) from split '{args.split}'.")

    # Write image list
    image_paths = [os.path.join(images_dir, stem + ".jpg") for stem in stems_with_both]
    out_list_path = args.out_list
    if not os.path.isdir(os.path.dirname(out_list_path)) and os.path.dirname(out_list_path):
        os.makedirs(os.path.dirname(out_list_path), exist_ok=True)
    with open(out_list_path, "w") as f:
        for p in image_paths:
            f.write(p + "\n")
    print(f"Wrote image list: {out_list_path} ({len(image_paths)} paths)")

    if args.no_masks:
        return

    if args.legacy_cityscapes_19:
        mapping = None
    else:
        mj = args.mapping_json or DEFAULT_MAPPING_JSON
        if not os.path.isfile(mj):
            raise FileNotFoundError(
                f"Mapping JSON not found: {mj}. Pass --mapping-json or use --legacy-cityscapes-19."
            )
        mapping = load_id_ood_mapping(mj, args.mapping_key)
        print(f"Using mapping from {mj} key={args.mapping_key} ({len(mapping)} entries)")

    os.makedirs(out_mask_dir, exist_ok=True)
    n_ood_masks = 0
    for stem in tqdm(stems_with_both, desc="OOD masks"):
        file_name = stem + ".jpg"
        seg_info = seg_map.get(file_name)
        if not seg_info:
            continue
        png_path = os.path.join(panoptic_dir, stem + ".png")
        img = np.array(Image.open(png_path).convert("RGB"))
        seg_ids = decode_panoptic_segment_id(img)
        cat_ids = np.vectorize(lambda s: seg_info.get(s))(seg_ids)
        if mapping is not None:
            binary = category_flags_from_mapping(cat_ids, mapping)
        else:
            id_mask = np.zeros(seg_ids.shape, dtype=bool)
            for cid in CITYSCAPES_19_LABEL_IDS:
                id_mask |= cat_ids == cid
            binary = np.where(id_mask, 0, 255).astype(np.uint8)
        out_path = os.path.join(out_mask_dir, stem + ".png")
        Image.fromarray(binary).save(out_path)
        n_ood_masks += 1
    print(f"Wrote {n_ood_masks} OOD masks to {out_mask_dir}")


if __name__ == "__main__":
    main()
