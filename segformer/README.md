# SegFormer Pipeline

Contains the SegFormer benchmark runner and final result summaries.

## Main files

- `run_all_methods_segformer.py` - run all OOD methods with SegFormer
- `segformer_wrapper.py` - model wrapper utilities

`run_all_methods_segformer.py` imports shared OOD logic from sibling folder:

- `../deeplabv3plus/run_all_methods.py`
- `../deeplabv3plus/ood_methods.py`

## Results

- `results/wilddash/`
- `results/mapillary/`
