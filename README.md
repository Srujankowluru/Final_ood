# Final OOD Segmentation Benchmark Pipeline

Organized, reproducible project layout for final benchmarking on:

- `deeplabv3plus/`
- `segformer/`

Each model folder contains code and final results split by dataset:

- `results/wilddash/`
- `results/mapillary/`

## Repository structure

- `deeplabv3plus/`
  - Core method code and DeepLab runner scripts
  - WildDash split files and tuned params
  - `scripts/run_wilddash_tune_perfect_final_1.sh`
  - `scripts/run_perfect_final_1.sh`
- `segformer/`
  - SegFormer benchmark runner + wrapper
  - Dataset-wise final results
- `mapping/`
  - `id_ood_mapping.json` (Mapillary + WildDash ID/OOD/ignore mapping)
  - Class CSVs for traceability

## Quick start

1. Create/activate environment with `torch`, `torchvision`, `transformers`, `numpy`, `Pillow`, `scikit-learn`, `matplotlib`, `tqdm`.
2. Set model checkpoint path:

```bash
export MODEL="/absolute/path/to/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
```

3. Run from `deeplabv3plus/`:

```bash
cd deeplabv3plus
bash scripts/run_wilddash_tune_perfect_final_1.sh
bash scripts/run_perfect_final_1.sh
```

## Run modes

- **Full pipeline run (recommended):** use `deeplabv3plus/scripts/run_perfect_final_1.sh`.
  - This executes both models (DeepLabV3+ and SegFormer) on both datasets (WildDash and Mapillary).
- **Model-specific run:** run the model runner directly:
  - DeepLab only: `deeplabv3plus/run_all_methods.py`
  - SegFormer only: `segformer/run_all_methods_segformer.py`
  - You can pass dataset-specific lists/masks via `--image-list` and `--mask-dir`.

## Data and model sources

- **Cityscapes dataset** (training domain / label taxonomy): [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
- **Mapillary Vistas dataset** (validation images and labels): [https://www.mapillary.com/dataset/vistas](https://www.mapillary.com/dataset/vistas)
- **WildDash 2 dataset** (panoptic annotations / OOD benchmark scenes): [https://wilddash.cc/](https://wilddash.cc/)
- **DeepLabV3+ architecture reference**: [https://github.com/jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
- **SegFormer model family (Hugging Face model card/docs)**: [https://huggingface.co/docs/transformers/model_doc/segformer](https://huggingface.co/docs/transformers/model_doc/segformer)

For this project, the exact checkpoint files are loaded from user-provided local paths in the run scripts (set via `MODEL` or script config).

## Notes

- WildDash masks are generated from `mapping/id_ood_mapping.json` using key `wilddash`.
- Mapillary evaluation uses existing mask directory path configured in scripts (can be overridden with `MAPILLARY_MASK` env var).
- Ignore pixels are excluded from metrics in both DeepLab and SegFormer evaluation code.
