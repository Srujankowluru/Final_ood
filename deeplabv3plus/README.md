# DeepLabV3+ Pipeline

Contains the DeepLabV3+ benchmark implementation, WildDash tuning scripts, and final result summaries.

## Main files

- `run_all_methods.py` - run all OOD methods with DeepLab
- `run_validation_tuning.py` - validation tuning workflow
- `ood_methods.py` - method implementations
- `prepare_wilddash_ood_masks.py` - WildDash mask generation from mapping
- `scripts/run_wilddash_tune_perfect_final_1.sh` - tune WildDash params
- `scripts/run_perfect_final_1.sh` - run full benchmark (DeepLab + SegFormer)

## Results

- `results/wilddash/`
- `results/mapillary/`
