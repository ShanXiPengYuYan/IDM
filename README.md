# IDM: Intra-modal Discriminability Modulation

Official implementation of *"Boosting Multimodal Remote Sensing Classification via Intra-modal Discriminability Modulation"*, Accepted at ACM MM 2026.

## Overview

IDM is a training-time regularization method for multimodal remote sensing classification. It estimates the per-sample discriminability state of each modality through an ambiguity score and employs a Gaussian modulation function to coordinate intra-modal discriminative enhancement and cross-modal guidance, with a self-calibrated scale strategy that adapts to different modality combinations.

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- NumPy, scikit-learn


## Datasets

We evaluate on the following benchmarks:

| Dataset | Modalities | Classes | Source |
|---------|-----------|---------|--------|
| Houston 2013 | HS + DSM | 15 | [IEEE GRSS](https://hyperspectral.ee.uh.edu/?page_id=459) |
| Berlin | HS + SAR | 8 | [EnMAP Contest](https://www.enmap.org/) |
| Augsburg | HS + SAR / HS + DSM | 7 | [TU Munich](https://mediatum.ub.tum.de/1474000) |

Important: After downloading, open hs_dsm_dataset.py and modify the data root path to your local dataset directory.

## Usage

### Training

```bash
# ResNet18 baseline
python train.py --dataset Houston --model resnet --seed 5

# ResNet18 + IDM
python train.py --dataset Houston --model resnet --seed 5 --lam 0.005 --sigma_scale 0.5 --warmup_epoch 10
```

### Key Arguments

| Argument | Description | Default |
|----------|------------|---------|
| `--dataset` | Dataset name: Houston, Berlin, Augsburg | Houston |
| `--model` | Backbone: resnet, cocnn, mft, exvit, dsymfuser, s2mamba, dahgm | resnet |
| `--lamg` | Regularization weight $\lambda$ | 0.5 |
| `--sigma_scale` | Global scaling factor $c$ | 1.0 |
| `--warmup_epoch` | Warmup epochs before enabling IDM | 10 |
| `--seed` | Random seed | 5 |

### Supported Architectures

- **CNN**: ResNet18, CO-CNN
- **Transformer**: MFT, ExViT, DSymFuser
- **Mamba**: S2Mamba, DAHGM

### Noise Robustness Evaluation

```bash
python eval_noise.py --dataset Houston --model resnet \
    --ckpt_baseline path/to/baseline.pt \
    --ckpt_idm path/to/idm.pt \
    --noise_levels 0.0 0.2 0.4 0.6 0.8 1.0
```

## Project Structure
    ├── train.py                 # Main training script
    ├── hs_dsm_dataset.py        # Dataset loading (modify data path here)
    ├── utils.py                 # Utility functions
    ├── test_noise.py            # Noise performance testing
    ├── losses/                  # Loss functions including IDM regularization
    ├── models/                  # Network architectures
    │   ├── backbone.py
    │   ├── resnet18.py
    │   ├── resnet18_adr.py
    │   ├── co_cnn.py
    │   ├── mft.py
    │   ├── mvit.py
    │   ├── dsymfuser.py
    │   ├── mamba.py
    │   ├── dahgmn.py
    │   └── SaCaCrossMamba.py
    └── requirements.txt


