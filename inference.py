#!/usr/bin/env python3
# ==============================================================================
# 🏆 RARE26 GRAND CHALLENGE - EXODIA-007 (V6 - CHAMPION WEIGHTS + RESCUE MODE)
# 🛡️ FIX: NO RANK ENSEMBLING (Using Raw Probs to fix per-video scaling bug)
# 🎯 FUSION: Original Champion Weights (Conv 41% / ViT 37% / Res 22%)
# ==============================================================================
import os
import json
import gc
from glob import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel
import timm
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. SETTINGS & PATHS (OFFLINE DOCKER)
# ==========================================
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

BASE_DIR = Path(__file__).parent
HF_CONVNEXT_PATH = BASE_DIR / "local_hf_models" / "local_dinov3_convnext"
DINOV2_REPO_PATH = BASE_DIR / "local_hf_models" / "dinov2_repo"  
WEIGHTS_DIR = BASE_DIR / "custom_weights"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE_CNN = 512
IMG_SIZE_VIT = 518

# ==========================================
# 2. MODEL ARCHITECTURES (OFFLINE LOCAL)
# ==========================================
class DINOv3_ConvNeXt_Local(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            str(HF_CONVNEXT_PATH), 
            trust_remote_code=True,
            local_files_only=True
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(128, 2)
        )
    def forward(self, x): 
        out = self.backbone(pixel_values=x)
        features = out.pooler_output if hasattr(out, 'pooler_output') else out.last_hidden_state[:, 0, :]
        return self.head(features)

class GastroNetResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.head(self.backbone(x))

class GastroNetViTNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            str(DINOV2_REPO_PATH), 
            'dinov2_vitb14', 
            source='local', 
            pretrained=False
        )
        self.head = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.head(self.backbone(x))

# ==========================================
# 3. UTILS & SMART LOAD
# ==========================================
def smart_load(model, weight_path, device):
    state_dict = torch.load(weight_path, map_location=device)
    model_state = model.state_dict()
    new_state = {}
    
    for loaded_k, v in state_dict.items():
        parts = loaded_k.split('.')
        matched_key = None
        for i in range(len(parts)):
            suffix = '.'.join(parts[i:])
            possible_keys = [k for k in model_state.keys() if k.endswith(suffix)]
            if len(possible_keys) == 1 and model_state[possible_keys[0]].shape == v.shape:
                matched_key = possible_keys[0]
                break
        if matched_key:
            new_state[matched_key] = v
            
    model.load_state_dict(new_state, strict=False)

def get_tta_mean_logits(model, img_pil, size):
    t_base = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t_hflip = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t_vflip = transforms.Compose([transforms.Resize((size, size)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t_color = transforms.Compose([transforms.Resize((size, size)), transforms.ColorJitter(brightness=0.3, contrast=0.3), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    with torch.amp.autocast('cuda'):
        l_orig = model(t_base(img_pil).unsqueeze(0).to(DEVICE))
        l_h = model(t_hflip(img_pil).unsqueeze(0).to(DEVICE))
        l_v = model(t_vflip(img_pil).unsqueeze(0).to(DEVICE))
        l_c = model(t_color(img_pil).unsqueeze(0).to(DEVICE))
        
    return (l_orig + l_h + l_v + l_c) / 4.0

def predict_raw_prob(model, img_pil, size):
    """🔥 ดึงคะแนนดิบ Softmax ตรงๆ ไม่ผ่าน Rank หรือ Calibration"""
    logits = get_tta_mean_logits(model, img_pil, size)
    return torch.softmax(logits, dim=1)[0, 1].item()

# ==========================================
# 4. GRAND CHALLENGE PIPELINE
# ==========================================
def run():
    interface_key = get_interface_key()
    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]
    return handler()

def interface_0_handler():
    print("=+=" * 10)
    print(f"🚀 THE GOLDEN EXODIA (RESCUE EDITION - NO RANK) Initialized on {DEVICE}")
    print("=+=" * 10)
    
    img_location = INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy"
    input_array = load_image_file_as_array(location=img_location)
    
    all_frames = []
    is_junk = [] 
    
    def process_and_append(arr):
        if arr.std() < 10: 
            is_junk.append(True)
            all_frames.append(None)
        else:
            is_junk.append(False)
            all_frames.append(Image.fromarray(arr.astype('uint8')).convert('RGB'))

    try:
        if input_array.ndim == 4: 
            for i in range(input_array.shape[0]): process_and_append(input_array[i])
        elif input_array.ndim == 3: 
            if input_array.shape[-1] in [3, 4]: process_and_append(input_array)
            else: 
                for i in range(input_array.shape[0]): process_and_append(input_array[i])
        elif input_array.ndim == 2: process_and_append(input_array)
    except Exception as e:
        print(f"🚨 Error processing frames: {e}")
        fallback_length = input_array.shape[0] if input_array.ndim > 2 else 1
        all_frames = [None] * fallback_length
        is_junk = [True] * fallback_length

    num_frames = len(all_frames)
    
    raw_preds_conv = np.zeros(num_frames)
    raw_preds_res = np.zeros(num_frames)
    raw_preds_vit = np.zeros(num_frames)

    # ----------------------------------------------------
    # 🧠 1. ConvNeXt (Fold 0=40% / Fold 1=60%)
    # ----------------------------------------------------
    for fold, w_fold in zip([0, 1], [0.40, 0.60]):
        print(f"🟢 Loading ConvNeXt Fold {fold}...")
        m = DINOv3_ConvNeXt_Local().to(DEVICE).eval()
        smart_load(m, WEIGHTS_DIR / f'best_oak3_convnext_fold_{fold}.pth', DEVICE)
        with torch.no_grad():
            for idx, (img, junk) in enumerate(zip(all_frames, is_junk)):
                if not junk: raw_preds_conv[idx] += predict_raw_prob(m, img, IMG_SIZE_CNN) * w_fold
        del m; gc.collect(); torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 🛡️ 2. ResNet50 (Fold 0=45% / Fold 1=55%)
    # ----------------------------------------------------
    for fold, w_fold in zip([0, 1], [0.45, 0.55]):
        print(f"🟢 Loading ResNet50 Fold {fold}...")
        m = GastroNetResNet50().to(DEVICE).eval()
        smart_load(m, WEIGHTS_DIR / f'best_gastronet_rn50_fold_{fold}.pth', DEVICE)
        with torch.no_grad():
            for idx, (img, junk) in enumerate(zip(all_frames, is_junk)):
                if not junk: raw_preds_res[idx] += predict_raw_prob(m, img, IMG_SIZE_CNN) * w_fold
        del m; gc.collect(); torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 👁️ 3. ViT-Base (Fold 0=35% / Fold 1=65%)
    # ----------------------------------------------------
    for fold, w_fold in zip([0, 1], [0.35, 0.65]):
        print(f"🟢 Loading ViT-Base Fold {fold}...")
        m = GastroNetViTNative().to(DEVICE).eval()
        smart_load(m, WEIGHTS_DIR / f'best_gastronet_vit_fold_{fold}.pth', DEVICE)
        with torch.no_grad():
            for idx, (img, junk) in enumerate(zip(all_frames, is_junk)):
                if not junk: raw_preds_vit[idx] += predict_raw_prob(m, img, IMG_SIZE_VIT) * w_fold
        del m; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # 👑 EXODIA FINAL FUSION (CHAMPION RAW PROBS)
    # ==========================================
    final_preds = np.zeros(num_frames)
    valid_indices = [i for i, junk in enumerate(is_junk) if not junk]
    
    if valid_indices:
        v_conv = raw_preds_conv[valid_indices]
        v_res = raw_preds_res[valid_indices]
        v_vit = raw_preds_vit[valid_indices]
        
        # 🌟 รวมร่างด้วยน้ำหนักดิบ แชมป์เก่า (Conv=41%, ViT=37%, Res=22%)
        valid_final = (v_conv * 0.41) + (v_vit * 0.37) + (v_res * 0.22)
        
        for i, v_idx in enumerate(valid_indices):
            final_preds[v_idx] = valid_final[i]
            
        # Smoothing (ลดจุดรบกวนข้ามเฟรม)
        if len(final_preds) > 3:
            final_preds = gaussian_filter1d(final_preds, sigma=1.0)

    # ⚠️ ป้องกันภาพขยะหลุดเข้ากระบวนการ ด้วยการบังคับให้คะแนนต่ำสุด
    output_scores = []
    for idx, junk in enumerate(is_junk):
        if junk:
            output_scores.append(0.0001)
        else:
            output_scores.append(float(max(0.0001, min(0.9999, final_preds[idx]))))

    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=output_scores,
    )
    print("✅ THE GOLDEN EXODIA (CHAMPION EDITION) Completed Successfully!")
    return 0

# ==========================================
# 5. GRAND CHALLENGE UTILS
# ==========================================
def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))

def load_json_file(*, location):
    with open(location, "r") as f: return json.loads(f.read())

def write_json_file(*, location, content):
    with open(location, "w") as f: f.write(json.dumps(content, indent=4))

def load_image_file_as_array(*, location):
    input_files = glob(str(location / "*.tif")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = sitk.ReadImage(input_files[0])
    return sitk.GetArrayFromImage(result)

if __name__ == "__main__":
    raise SystemExit(run())
