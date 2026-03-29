#!/bin/bash
# Same as run_perfect_final.sh but:
# - WildDash masks: manual JSON mapping (default: ./wilddash_ood_masks_json)
# - Output: perfect_final_1/
# - Tuned params: wilddash_tuned_params_1.json (from run_wilddash_tune_perfect_final_1.sh)
#
# Mapillary unchanged. Run tuning first so WildDash uses tuned hyperparameters.
#
set -e
cd "$(dirname "$0")/.."
DEEPLAB_ABS="$(pwd)"
SEGFORMER_ROOT="../segformer"

MODEL="${MODEL:-/path/to/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar}"
GALLERY_DIR="."
PARAMS_JSON="wilddash_tuned_params_1.json"

WILDDASH_MASK="${WILDDASH_MASK:-$DEEPLAB_ABS/wilddash_ood_masks_json}"
WILDDASH_LIST="wilddash_test.txt"
MAX_WILDDASH=300

MAPILLARY_MASK="${MAPILLARY_MASK:-/fastdata/groupL/datasets/mapillary/v1.2/validation/ood_masks}"
MAPILLARY_LIST="val_list.txt"

OUTPUT_BASE="perfect_final_1"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "Error: PyTorch not found. conda activate ood_seg then re-run."
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "Error: MODEL checkpoint not found: $MODEL"
  echo "Set MODEL env var before running."
  exit 1
fi

if [ ! -d "$WILDDASH_MASK" ] || [ -z "$(find "$WILDDASH_MASK" -maxdepth 1 -name '*.png' -print -quit 2>/dev/null)" ]; then
  echo "Error: WildDash masks not found under $WILDDASH_MASK"
  echo "Run: bash run_wilddash_tune_perfect_final_1.sh   (generates masks + tuning)"
  echo "Or:  python prepare_wilddash_ood_masks.py --out-mask-dir $WILDDASH_MASK --mapping-json .../id_ood_mapping.json"
  exit 1
fi

METHOD_PARAMS_ARG=""
if [ -f "$PARAMS_JSON" ]; then
  METHOD_PARAMS_ARG="--method-params $PARAMS_JSON"
  echo "Using tuned params: $PARAMS_JSON"
else
  echo "Warning: no $PARAMS_JSON — WildDash runs use defaults. Run run_wilddash_tune_perfect_final_1.sh first."
fi

mkdir -p "$OUTPUT_BASE"

echo ""
echo "========== 1) DeepLab WildDash -> $OUTPUT_BASE/deeplab_wilddash =========="
if [ -f "$WILDDASH_LIST" ] && [ -d "$WILDDASH_MASK" ]; then
  $PYTHON run_all_methods.py \
    --model "$MODEL" \
    --image-list "$WILDDASH_LIST" \
    --mask-dir "$WILDDASH_MASK" \
    --output-dir "$OUTPUT_BASE/deeplab_wilddash" \
    --gallery-dir "$GALLERY_DIR" \
    --max-images $MAX_WILDDASH \
    $METHOD_PARAMS_ARG \
    --plot
else
  echo "Skip: $WILDDASH_LIST or $WILDDASH_MASK"
fi

echo ""
echo "========== 2) DeepLab Mapillary -> $OUTPUT_BASE/deeplab_mapillary =========="
if [ -f "$MAPILLARY_LIST" ] && [ -d "$MAPILLARY_MASK" ]; then
  $PYTHON run_all_methods.py \
    --model "$MODEL" \
    --image-list "$MAPILLARY_LIST" \
    --mask-dir "$MAPILLARY_MASK" \
    --output-dir "$OUTPUT_BASE/deeplab_mapillary" \
    --gallery-dir "$GALLERY_DIR" \
    $METHOD_PARAMS_ARG \
    --plot
else
  echo "Skip: $MAPILLARY_LIST or $MAPILLARY_MASK"
fi

echo ""
echo "========== 3) SegFormer WildDash -> $OUTPUT_BASE/segformer_wilddash =========="
if [ -d "$SEGFORMER_ROOT" ] && [ -f "$DEEPLAB_ABS/$WILDDASH_LIST" ] && [ -d "$WILDDASH_MASK" ]; then
  cd "$SEGFORMER_ROOT"
  PARAMS_ABS="$DEEPLAB_ABS/$PARAMS_JSON"
  LIST_ABS="$DEEPLAB_ABS/$WILDDASH_LIST"
  SEG_P=""
  [ -f "$PARAMS_ABS" ] && SEG_P="--method-params $PARAMS_ABS"
  $PYTHON run_all_methods_segformer.py \
    --image-list "$LIST_ABS" \
    --mask-dir "$WILDDASH_MASK" \
    --output-dir "$DEEPLAB_ABS/$OUTPUT_BASE/segformer_wilddash" \
    --gallery-dir "./galleries" \
    --max-images $MAX_WILDDASH \
    --low-mem \
    $SEG_P \
    --plot
  cd "$DEEPLAB_ABS"
else
  echo "Skip: SegFormer or WildDash data"
fi

echo ""
echo "========== 4) SegFormer Mapillary -> $OUTPUT_BASE/segformer_mapillary =========="
if [ -d "$SEGFORMER_ROOT" ] && [ -f "$DEEPLAB_ABS/$MAPILLARY_LIST" ] && [ -d "$MAPILLARY_MASK" ]; then
  cd "$SEGFORMER_ROOT"
  PARAMS_ABS="$DEEPLAB_ABS/$PARAMS_JSON"
  LIST_ABS="$DEEPLAB_ABS/$MAPILLARY_LIST"
  SEG_METHOD_PARAMS=""
  [ -f "$PARAMS_ABS" ] && SEG_METHOD_PARAMS="--method-params $PARAMS_ABS"
  $PYTHON run_all_methods_segformer.py \
    --image-list "$LIST_ABS" \
    --mask-dir "$MAPILLARY_MASK" \
    --output-dir "$DEEPLAB_ABS/$OUTPUT_BASE/segformer_mapillary" \
    --gallery-dir "./galleries" \
    --low-mem \
    $SEG_METHOD_PARAMS \
    --plot
  cd "$DEEPLAB_ABS"
else
  echo "Skip: SegFormer or Mapillary data"
fi

echo ""
echo "Done. Results in $OUTPUT_BASE/"
