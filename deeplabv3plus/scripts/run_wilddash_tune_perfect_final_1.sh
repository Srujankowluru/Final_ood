#!/bin/bash
# Tune Energy, Mahalanobis, Mahalanobis++, VIM, kNN on WildDash VAL using manual JSON masks
# (id_ood_mapping.json wilddash) + save wilddash_tuned_params_1.json for perfect_final_1.
#
# Prereq: masks in WILDDASH_MASK_DIR (run prepare_wilddash_ood_masks.py with --mapping-json).
#
set -e
cd "$(dirname "$0")/.."
DEEPLAB_ABS="$(pwd)"

WILDDASH_MASK_DIR="${WILDDASH_MASK_DIR:-$DEEPLAB_ABS/wilddash_ood_masks_json}"
MAPPING_JSON="${MAPPING_JSON:-$DEEPLAB_ABS/../mapping/id_ood_mapping.json}"
MODEL="${MODEL:-/path/to/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar}"
GALLERY_DIR="."
PARAMS_JSON="wilddash_tuned_params_1.json"
VAL_LIST="wilddash_val_tune.txt"
TEST_LIST="wilddash_test.txt"
MAX_VAL=191
MAX_TEST=300

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi
if [ ! -f "$MODEL" ]; then
  echo "Error: MODEL checkpoint not found: $MODEL"
  echo "Set MODEL env var before running."
  exit 1
fi

echo "=== Step 0: val/test lists ==="
if [ ! -f "$VAL_LIST" ] || [ ! -f "$TEST_LIST" ]; then
  $PYTHON create_wilddash_val_test_split.py --list wilddash_val_list.txt --val-out "$VAL_LIST" --test-out "$TEST_LIST"
else
  echo "Using existing $VAL_LIST and $TEST_LIST"
fi

echo ""
echo "=== Step 0b: WildDash OOD masks (JSON mapping) ==="
if [ ! -d "$WILDDASH_MASK_DIR" ] || [ -z "$(find "$WILDDASH_MASK_DIR" -maxdepth 1 -name '*.png' -print -quit 2>/dev/null)" ]; then
  echo "Writing masks to $WILDDASH_MASK_DIR"
  mkdir -p "$WILDDASH_MASK_DIR"
  $PYTHON prepare_wilddash_ood_masks.py \
    --wilddash-root /fastdata/groupL/datasets/wilddash \
    --split validation \
    --out-list wilddash_val_list.txt \
    --out-mask-dir "$WILDDASH_MASK_DIR" \
    --mapping-json "$MAPPING_JSON" \
    --mapping-key wilddash
else
  echo "Using existing masks in $WILDDASH_MASK_DIR"
fi

echo ""
echo "=== Step 0c: Mahalanobis++ galleries ==="
MAHA_L3="mahalanobis_gallery_layer3.pkl"
MAHA_L4="mahalanobis_gallery_layer4.pkl"
if [ ! -f "$MAHA_L3" ] && [ ! -f "$MAHA_L4" ]; then
  $PYTHON build_feature_gallery_layer3.py
else
  echo "Galleries present."
fi

echo ""
echo "=== Step 1: Tune on WildDash val -> $PARAMS_JSON ==="
rm -f "$PARAMS_JSON"
for METHOD in Energy Mahalanobis "Mahalanobis++" VIM kNN; do
  echo "--- Tuning $METHOD ---"
  $PYTHON run_validation_tuning.py \
    --method "$METHOD" \
    --val-list "$VAL_LIST" \
    --test-list "$TEST_LIST" \
    --mask-dir "$WILDDASH_MASK_DIR" \
    --model "$MODEL" \
    --gallery-dir "$GALLERY_DIR" \
    --metric AUROC \
    --max-val $MAX_VAL \
    --max-test $MAX_TEST \
    --save-params "$PARAMS_JSON"
done

echo ""
echo "Done. Tuned params: $DEEPLAB_ABS/$PARAMS_JSON"
echo "Next: bash run_perfect_final_1.sh"
