# R2R

Official training and evaluation code for `R2R`, a prompt-based image restoration framework that supports:

- `1D`: single-task restoration
- `3D`: denoise + derain + dehaze
- `5D`: denoise + derain + dehaze + deblur + low-light
- `compos`: compositional degradations on CDD11

This repository is organized for reproducible training and future public release. The codebase now uses a unified two-stage training pipeline:

1. `pretrain`: normal training from scratch
2. `finetune`: load the final checkpoint and final prompt bank from stage 1, then continue training

For each run, only the final checkpoint and final prompt bank are kept:

- checkpoint: `last.ckpt`
- prompt bank: `last`

## Highlights

- Unified training entry for each setting: `train_1D.py`, `train_3D.py`, `train_5D.py`, `train_compos.py`
- Unified model definitions under `net/`
- Unified configuration under `options/`
- Explicit dataset structure and command examples
- Final-only saving to keep the repository clean

## Repository Structure

```text
R2R/
├── net/
│   ├── feature_bank_1D.py
│   ├── feature_bank_3D.py
│   ├── feature_bank_5D.py
│   ├── feature_bank_compos.py
│   ├── model_1D.py
│   ├── model_3D.py
│   ├── model_5D.py
│   └── model_compos.py
├── options/
│   ├── options_1D.py
│   ├── options_3D.py
│   ├── options_5D.py
│   └── options_compos.py
├── utils/
├── data_dir/
│   ├── noisy/denoise.txt
│   ├── rainy/rainTrain.txt
│   ├── hazy/hazy_outside.txt
│   ├── gopro/train_gopro.txt
│   └── lol/train_lol.txt
├── train_1D.py
├── train_3D.py
├── train_5D.py
├── train_compos.py
├── test_1D.py
├── test_3D.py
├── test_5D.py
└── test_cdd11.py
```

## Environment

The code was cleaned for Python + PyTorch training with Lightning. A minimal environment is:

```bash
conda create -n r2r python=3.10 -y
conda activate r2r

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lightning torchmetrics pillow numpy matplotlib scikit-learn tqdm wandb
```

Notes:

- `wandb` is imported by the training scripts, although the current default logger is TensorBoard.
- `torchmetrics` is used by some evaluation scripts.
- `scikit-learn` and `matplotlib` are required by the prompt bank utilities.

## Data Preparation

All training scripts assume the following top-level layout:

```text
data/
├── Train/
│   ├── Denoise/
│   ├── Derain/
│   ├── Dehaze/
│   ├── Deblur/
│   ├── Lowlight/
│   └── CDD11/
└── Test/
    ├── Denoise/
    ├── Derain/
    ├── Dehaze/
    ├── Deblur/
    ├── Lowlight/
    └── CDD11/
```

The file lists under [`data_dir/`](/Users/wcx/code/read_paper/R2R/data_dir) are used to select training samples for denoising, deraining, dehazing, deblurring, and low-light enhancement.

### 1. Denoising

Training:

```text
data/Train/Denoise/
├── 00012.bmp
├── 00030.bmp
├── ...
```

The selected file names are listed in:

- [`data_dir/noisy/denoise.txt`](/Users/wcx/code/read_paper/R2R/data_dir/noisy/denoise.txt)

Testing:

```text
data/Test/Denoise/
└── cbsd68/
    ├── image1.png
    ├── image2.png
    └── ...
```

### 2. Deraining

Training:

```text
data/Train/Derain/
├── rainy/
│   ├── rain-1.png
│   ├── rain-2.png
│   └── ...
└── gt/
    ├── norain-1.png
    ├── norain-2.png
    └── ...
```

The rainy image list is read from:

- [`data_dir/rainy/rainTrain.txt`](/Users/wcx/code/read_paper/R2R/data_dir/rainy/rainTrain.txt)

Testing:

```text
data/Test/Derain/
└── Rain100L/
    ├── input/
    └── target/
```

### 3. Dehazing

Training:

```text
data/Train/Dehaze/
├── synthetic/
│   ├── part1/
│   ├── part2/
│   └── ...
└── original/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

The hazy file list is read from:

- [`data_dir/hazy/hazy_outside.txt`](/Users/wcx/code/read_paper/R2R/data_dir/hazy/hazy_outside.txt)

Testing:

```text
data/Test/Dehaze/
├── input/
└── target/
```

The code matches `input/xxx_*.png` to `target/xxx.png`.

### 4. Deblurring

Training:

```text
data/Train/Deblur/
├── blur/
└── sharp/
```

The blurred file list is read from:

- [`data_dir/gopro/train_gopro.txt`](/Users/wcx/code/read_paper/R2R/data_dir/gopro/train_gopro.txt)

Testing:

```text
data/Test/Deblur/
├── blur/
└── sharp/
```

### 5. Low-Light Enhancement

Training:

```text
data/Train/Lowlight/
├── low/
└── high/
```

The low-light training list is read from:

- [`data_dir/lol/train_lol.txt`](/Users/wcx/code/read_paper/R2R/data_dir/lol/train_lol.txt)

Testing:

```text
data/Test/Lowlight/
├── low/
└── high/
```

### 6. CDD11 for Compositional Degradations

Training:

```text
data/Train/CDD11/
└── cdd11/
    └── train/
        ├── clear/
        ├── haze/
        ├── rain/
        ├── snow/
        ├── low/
        ├── haze_rain/
        ├── haze_snow/
        ├── low_haze/
        ├── low_rain/
        ├── low_snow/
        ├── low_haze_rain/
        └── low_haze_snow/
```

Testing:

```text
data/Test/CDD11/
└── cdd11/
    └── test/
        ├── clear/
        ├── haze/
        ├── rain/
        ├── snow/
        ├── low/
        ├── haze_rain/
        ├── haze_snow/
        ├── low_haze/
        ├── low_rain/
        ├── low_snow/
        ├── low_haze_rain/
        └── low_haze_snow/
```

Every degraded subfolder should use the same file names as `clear/`.

## Training

### General Rule

Each training script supports:

- `--train_mode pretrain`
- `--train_mode finetune`

Pretraining saves:

- checkpoints under `train_ckpt_*`
- prompts under `save_prompts_*`

Finetuning loads:

- `last.ckpt`
- `last` prompts

and writes new outputs to:

- `train_ckpt_*_finetune`

Finetuning does not update the prompt bank. It reuses the prompt bank saved in pretraining as fixed memory.

### 1D

Single-task restoration. Change `--de_type` to select the task.

Pretrain:

```bash
python train_1D.py --train_mode pretrain --de_type deblur
```

Finetune:

```bash
python train_1D.py \
  --train_mode finetune \
  --de_type deblur \
  --init_ckpt_dir train_ckpt_1D \
  --init_prompt_dir save_prompts_1D \
  --init_prompt_name last
```

### 3D

Multi-task restoration on denoising, deraining, and dehazing.

Pretrain:

```bash
python train_3D.py --train_mode pretrain
```

Finetune:

```bash
python train_3D.py \
  --train_mode finetune \
  --init_ckpt_dir train_ckpt_3D \
  --init_prompt_dir save_prompts_3D \
  --init_prompt_name last
```

### 5D

Multi-task restoration on denoising, deraining, dehazing, deblurring, and low-light enhancement.

Pretrain:

```bash
python train_5D.py --train_mode pretrain
```

Finetune:

```bash
python train_5D.py \
  --train_mode finetune \
  --init_ckpt_dir train_ckpt_5D \
  --init_prompt_dir save_prompts_5D \
  --init_prompt_name last
```

### compos

Compositional restoration on CDD11.

Pretrain:

```bash
python train_compos.py --train_mode pretrain
```

Finetune:

```bash
python train_compos.py \
  --train_mode finetune \
  --init_ckpt_dir train_ckpt_compos \
  --init_prompt_dir save_prompts_compos \
  --init_prompt_name last
```

## Evaluation

### 1D

Modes:

- `0`: denoise
- `1`: derain
- `2`: dehaze
- `3`: deblur
- `4`: lowlight

Example:

```bash
python test_1D.py --mode 3 \
  --ckpt_name train_ckpt_1D_finetune/ \
  --prompt_dir save_prompts_1D/
```

### 3D

Modes:

- `0`: denoise
- `1`: derain
- `2`: dehaze
- `3`: all-in-one

Example:

```bash
python test_3D.py --mode 3 \
  --ckpt_name train_ckpt_3D_finetune/ \
  --prompt_dir save_prompts_3D/
```

### 5D

Modes:

- `0`: denoise
- `1`: derain
- `2`: dehaze
- `3`: deblur
- `4`: lowlight
- `5`: all-in-one

Example:

```bash
python test_5D.py --mode 5 \
  --ckpt_name train_ckpt_5D_finetune/ \
  --prompt_dir save_prompts_5D/
```

### compos / CDD11

Modes:

- `1`: single degradations
- `2`: double degradations
- `3`: triple degradations
- `4`: all

Example:

```bash
python test_cdd11.py --mode 4 \
  --ckpt_name train_ckpt_compos_finetune/ \
  --prompt_dir save_prompts_compos/
```

## Saved Files

Each run keeps only the final artifacts.

Pretraining saves:

- `last.ckpt`
- `last` prompt bank

Finetuning saves:

- `last.ckpt`

and reuses the pretrained prompt bank during evaluation.

Example outputs:

```text
train_ckpt_3D/
└── last.ckpt

save_prompts_3D/
└── last*
```

For `1D`, checkpoints and prompts are grouped by task:

```text
train_ckpt_1D/
└── deblur/
    └── last.ckpt

save_prompts_1D/
└── deblur/
    └── last*
```

## Practical Notes

- `1D` uses task-specific subdirectories for checkpoint/prompt loading and saving.
- `3D`, `5D`, and `compos` save directly under their own checkpoint/prompt directories.
- The current training scripts use TensorBoard by default.
- Some auxiliary debugging utilities still exist in the codebase, but the main training and testing paths are now consolidated and ready for public release.

## Citation

If you use this repository in your research, please cite the corresponding paper once the public paper information is released.
