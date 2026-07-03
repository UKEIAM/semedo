# Project template: GPU
This repository represents a blueprint for Python projects using pyTorch optimized with NVIDIA GPUs. The image pre-install the following software components:

- Python 3.10
- Anaconda with a default environment
    - pyTorch
    - Linter for appropiate code standards (and config files for reasonable defaults)
        - flake8
        - pyLint
- NVIDIA toolchain
    - CUDA
    - cuBLAS
    - NVIDIA cuDNN
    - NVIDIA NCCL (optimized for NVLink)
    - RAPIDS
    - NVIDIA Data Loading Library (DALI)
    - TensorRT
    - Torch-TensorRT

## Usage
Specify the packages you require in the *requirements.txt*. More complex environment customization goes into *Dockerfile*.

While using Visual Studio Code for development is encouraged, the image does not depend on this IDE in any way. As a side effect, its required server components are not even installed by default if the Dockerfile in root is built manually. Opening the project in VS Code will set the proper default and configure everything appropriately. Alternatively, build the container with `docker build -t <YOUR PROJECT NAME>:0.1 .` and run the container with `docker run -p <YOUR LOCAL PORT>:22 --rm --gpus all <YOUR PROJECT NAME>:0.1`.

## Annotation tool

The annotation notebook is now config-driven.

1. Edit [src/annotate/annotation_config.json](src/annotate/annotation_config.json)
2. Set the image folders in `image_folders`
3. Set the GT export path in `output_csv`
4. Open [src/annotate/annotation.ipynb](src/annotate/annotation.ipynb) and run the first cell

By default the config points at the UKE image folders under [data/uke](data/uke) (`train`/`val`/`test`) and writes the ground-truth CSV to `data/uke/train_uke_model_coordinates.csv` — the same paths `training_config.json` expects for `uke_train_dir`/`uke_val_dir`/`uke_test_dir`/`uke_metadata_path`, so you can annotate and then fine-tune without changing any paths.

## Training pipeline

Training uses [src/finetune/training_config.json](src/finetune/training_config.json). Neither the SmartDoc15 dataset nor the pretrained `u2net.pth` weights are stored in this repository.

1. Edit [src/finetune/training_config.json](src/finetune/training_config.json):
   - Check `model_path`, `root_folder`, `opensource_metadata_path`, `opensource_image_folder_path` point where you want the dataset/weights to live.
   - Pick a scenario to run by setting `selected_ratio_name` to one of the `ratio_scenarios` entries (or set `run_all_ratio_scenarios: true` to sweep all of them).
   - Set `wandb_entity`/`wandb_project` to your own Weights & Biases account, or set `wandb_mode: "disabled"` if you don't want experiment tracking.
2. Download the assets referenced by the config:
   ```
   cd src/finetune
   python download_assets.py
   ```
   This fetches `u2net.pth` (from the URL in `u2net_download_url`) and the SmartDoc15 `frames.tar.gz` release asset, and skips anything that's already present. Use `--force` to redownload, or `--skip-model`/`--skip-data` to fetch only one of the two.
3. Start training:
   ```
   python finetuning.py
   ```
   or open [src/finetune/finetuning.ipynb](src/finetune/finetuning.ipynb), run the config/download/validation cells, then set `RUN_TRAINING = True` in the last cell.

If you also have your own annotated (UKE-style) data, point `uke_metadata_path`/`uke_train_dir`/`uke_val_dir`/`uke_test_dir` at it and keep `include_uke_in_training: true`; otherwise set that flag to `false` to train on SmartDoc15 alone.