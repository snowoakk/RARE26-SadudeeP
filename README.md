# RARE26-Oak — Team SadudeeP

> **RARE 2026 Grand Challenge** · Recognition of Abnormalities in Low-Prevalence Cancer  
> Early Barrett's Neoplasia Detection from Endoscopy Images

---

## What is this?

This repo contains **INWZA-007** (Integrated Neural Weighted Zoom Architecture v7), the inference pipeline submitted to the [RARE 2026 Grand Challenge](https://rare26.grand-challenge.org/) by **Team SadudeeP**.

The task: detect early neoplastic lesions in Barrett's esophagus endoscopy images — a severely imbalanced problem (~1% positive prevalence) with cross-center domain shift.

**Result on the hidden test set:** AUROC **0.9098** · PPV@90% recall **0.0335**

---

## How It Works

### 1. Three-Model Ensemble
Three backbone architectures are trained independently, each with a custom multi-layer classification head:

| Model | Backbone | Weights Init | Input Size |
|---|---|---|---|
| ConvNeXt Base | DINOv3-ConvNeXt | DINOv3 foundation | 512×512 |
| ViT-Base | DINOv2-ViT-B/14 | DINOv2 self-supervised | 518×518 |
| ResNet-50 | GastroNet-ResNet50 | GastroNet-5M domain | 512×512 |

### 2. Cross-Center Training
Each model is trained with **2-fold cross-validation split by hospital center** (leave-one-center-out), forcing the network to learn center-invariant features rather than facility-specific artifacts.

### 3. Test-Time Augmentation (TTA)
At inference, every frame is processed through **4 augmented views** (original, H-flip, V-flip, color jitter) and the logits are averaged to reduce frame-level noise.

### 4. Hierarchical Soft-Voting Fusion
Predictions are merged in two tiers:
- **Tier 1 — Fold blending** (within each model family):
  - ConvNeXt: Fold 0 × 40% + Fold 1 × 60%
  - ResNet-50: Fold 0 × 45% + Fold 1 × 55%
  - ViT-Base: Fold 0 × 35% + Fold 1 × 65%
- **Tier 2 — Cross-architecture ensemble:**
  ```
  Final = 0.41 × ConvNeXt + 0.37 × ViT + 0.22 × ResNet
  ```

### 5. Memory-Safe Sequential Inference
To stay within GPU memory limits, models are loaded **one at a time** — each checkpoint is loaded, scored across all frames, then immediately deleted and flushed from CUDA memory before the next model is loaded. Average speed: **~0.04 s/image**.

---

## Repository Structure

```
RARE26-Oak/
├── inference.py          # Main inference pipeline (INWZA-007)
├── Dockerfile            # Container definition (pytorch + CUDA)
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── do_build.sh           # Build Docker image
├── do_test_run.sh        # Local end-to-end test run
├── do_save.sh            # Save image for upload
├── run_pipeline.bat      # Windows pipeline runner
├── custom_weights/       # Trained model checkpoints (.pth) for inference container
├── local_hf_models/      # Offline HuggingFace model files
│   ├── local_dinov3_convnext/
│   └── dinov2_repo/
├── src/                  # Source package (rare26_oak)
├── data/                 # Dataset utilities
├── test/                 # Test inputs & expected outputs
├── model/                # From RARE26 template
```

---

## Model Weights

Place pre-trained checkpoint files in `custom_weights/`:

| File | Model |
|---|---|
| `best_oak3_convnext_fold_0.pth` | ConvNeXt Fold 0 |
| `best_oak3_convnext_fold_1.pth` | ConvNeXt Fold 1 |
| `best_gastronet_rn50_fold_0.pth` | ResNet-50 Fold 0 |
| `best_gastronet_rn50_fold_1.pth` | ResNet-50 Fold 1 |
| `best_gastronet_vit_fold_0.pth` | ViT-Base Fold 0 |
| `best_gastronet_vit_fold_1.pth` | ViT-Base Fold 1 |

---

## What You Need to Download

Everything in the lists below is **gitignored** and not included in this repo. Here's exactly what to get and where to put it.

---

### 🟢 For Inference Only

You need **all three** of these to run `inference.py` / Docker.

#### 1. Fine-tuned champion weights → `custom_weights/`
Our 6 trained `.pth` checkpoints (shared separately by the team):

```
custom_weights/
├── best_oak3_convnext_fold_0.pth
├── best_oak3_convnext_fold_1.pth
├── best_gastronet_rn50_fold_0.pth
├── best_gastronet_rn50_fold_1.pth
├── best_gastronet_vit_fold_0.pth
└── best_gastronet_vit_fold_1.pth
```

#### 2. DINOv3-ConvNeXt HuggingFace model → `local_hf_models/local_dinov3_convnext/`
Download from HuggingFace: [`Simeoni/dinov3-convnext-base-pretrain-lvd1689m`](https://huggingface.co/Simeoni/dinov3-convnext-base-pretrain-lvd1689m)

```
local_hf_models/local_dinov3_convnext/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

#### 3. DINOv2 repository → `local_hf_models/dinov2_repo/`
Clone the facebookresearch DINOv2 repo here so the ViT backbone can be loaded from `source='local'`:

```bash
git clone https://github.com/facebookresearch/dinov2 local_hf_models/dinov2_repo
```

---

### 🟡 For Training (Mandatory Base Weights — Stage 3)

All **three** weights below are mandatory for reproducing champion model performance. Items 4 & 5 come from the same HuggingFace repo.

#### 4. GastroNet-5M ResNet-50 SSL weights → `training/base_model/`
File: `RN50_Billion-Scale-SWSL2BGastroNet-5M_DINOv1.pth`  
Download: [`tgwboers/GastroNet-5M_Pretrained_Weights`](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights) on HuggingFace

#### 5. DINOv2 ViT-Base pre-trained weights → `training/base_model/`
File: `dinov2.pth`  
Download: [`tgwboers/GastroNet-5M_Pretrained_Weights`](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights) on HuggingFace (same repo, different file)

#### 6. DINOv3-ConvNeXt Base model → `local_hf_models/local_dinov3_convnext/`
Used as the backbone initialisation for the ConvNeXt champion model (also reused for inference).  
Download: [`facebook/dinov3-convnext-small-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m) on HuggingFace

```
training/base_model/
├── RN50_Billion-Scale-SWSL2BGastroNet-5M_DINOv1.pth
└── dinov2.pth

local_hf_models/local_dinov3_convnext/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

Then pass them in when running Stage 3 — uncomment these lines in [`run_pipeline.bat`](run_pipeline.bat):
```bat
SET GASTRONET_WEIGHTS=training\base_model\RN50_Billion-Scale-SWSL2BGastroNet-5M_DINOv1.pth
SET DINOV2_WEIGHTS=training\base_model\dinov2.pth
```

---

### 🔴 For Training (All stages — requires dataset access)

These datasets must be requested from the challenge organisers.

#### 7. RARE26 challenge dataset → `data/rare26/`
Request access from the [RARE 2026 Grand Challenge](https://grand-challenge.org).  
Expected folder structure:

```
data/rare26/
├── center_1/
│   ├── ndbe/   ← non-dysplastic Barrett's images (.png / .jpg)
│   └── neo/    ← neoplastic lesion images (.png / .jpg)
└── center_2/
    ├── ndbe/
    └── neo/
```

#### 8. GastroNet-5M unlabeled images → `data/gastronet/`
Used in Stage 2 (pseudo-label generation only). Any `.png`/`.jpg` images placed here will be used.  
Request dataset access via the [GastroNet-5M paper](https://doi.org/10.1053/j.gastro.2025.07.030).

```
data/gastronet/
└── (any flat or nested structure of .png / .jpg images)
```

---

### Summary

| # | What | Where | Needed for |
|---|---|---|---|
| 1 | Champion `.pth` weights (×6) | `custom_weights/` | Inference ✅ |
| 2 | DINOv3-ConvNeXt HF model | `local_hf_models/local_dinov3_convnext/` | Inference ✅ + Training ⚠️ |
| 3 | DINOv2 repo (cloned) | `local_hf_models/dinov2_repo/` | Inference ✅ |
| 4 | [GastroNet ResNet SSL weights](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights) | `training/base_model/` | Training Stage 3 ⚠️ mandatory |
| 5 | [DINOv2 ViT-B/14 weights](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights) | `training/base_model/` | Training Stage 3 ⚠️ mandatory |
| 6 | [DINOv3-ConvNeXt base model](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m) | `local_hf_models/local_dinov3_convnext/` | Training Stage 3 ⚠️ mandatory |
| 7 | RARE26 dataset images | `data/rare26/` | Training Stages 1 & 3 |
| 8 | GastroNet-5M images | `data/gastronet/` | Training Stage 2 |

---

## Training Locally on Windows

> The preferred way to train is via [`run_pipeline.bat`](run_pipeline.bat) — a single script that runs all 3 stages in sequence.

### Prerequisites

- Python 3.11+ installed and on PATH
- GPU with CUDA support strongly recommended (CPU will be very slow)
- All datasets and mandatory base weights downloaded (items 4–8 above)

### Step 1 — Configure `run_pipeline.bat`

Open [`run_pipeline.bat`](run_pipeline.bat) and edit the top section:

```bat
REM Point to your dataset directories
SET RARE26_DATA_DIR=data\rare26
SET GASTRONET_DATA_DIR=data\gastronet

REM Uncomment and set your base weight paths (mandatory for Stage 3)
SET GASTRONET_WEIGHTS=training\base_model\RN50_Billion-Scale-SWSL2BGastroNet-5M_DINOv1.pth
SET DINOV2_WEIGHTS=training\base_model\dinov2.pth
```

### Step 2 — Test the pipeline first (debug mode)

Run with `--debug` to verify everything is wired up correctly using just 1 epoch and 16 samples:

```bat
run_pipeline.bat --debug
```

This runs all 3 stages quickly without real training — use it to catch any path or import errors before committing to a full run.

### Step 3 — Full training run

```bat
run_pipeline.bat
```

This runs the full 3-stage pipeline:

| Stage | What it does | Output |
|---|---|---|
| **Stage 1** | Train mother models (ConvNeXt-Tiny, ConvNeXt-Base, EfficientNetV2) on RARE26 | `outputs/mother_weights/` |
| **Stage 2** | Run mother ensemble on GastroNet-5M to generate pseudo-labels | `outputs/pseudo_labels/pseudo_labels.csv` |
| **Stage 3** | Train champion models (ConvNeXt, ResNet-50, ViT-Base) with real + pseudo data | `outputs/champ_weights/` |

The final champion `.pth` files in `outputs/champ_weights/` are the weights used for inference.

---

## Authors

**Team SadudeeP** — Sadudee Poliwat · Ngo Anh Huyen  
Submitted to RARE 2026 · August 2026
