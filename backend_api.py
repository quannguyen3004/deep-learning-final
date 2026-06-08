"""
FastAPI Backend for Animal/Plant Classification
Loads pre-trained ResNet50 and ViT models with Stacking Ensemble
Serves predictions for uploaded images
"""

import os
import io
import json
import torch
import numpy as np
import timm
from pathlib import Path
from typing import List, Dict
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torchvision.models as models
from torchvision import transforms
from torch import nn
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime

# ==================== CONFIG ====================
MODELS_DIR = Path(__file__).parent
RESNET_CHECKPOINT = MODELS_DIR / "best_model_resnet50_finetune.pth"
VIT_CHECKPOINT = MODELS_DIR / "best_model_vit.pth"
STACKING_META_LEARNER = MODELS_DIR / "stacking_meta_learner.pkl"

CLASS_NAMES = [
    "Amphibia", "Animalia", "Arachnida", "Aves", "Fungi",
    "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")

# ==================== INITIALIZE FASTAPI ====================
app = FastAPI(
    title="Animal & Plant Classification API",
    description="API for classifying animals and plants using Deep Learning Ensemble",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== INITIALIZE IMAGE TRANSFORMS ====================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================== LOAD MODELS ====================
def load_resnet50():
    """Load ResNet50 model with fine-tuned weights"""
    print("📦 Loading ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    if RESNET_CHECKPOINT.exists():
        checkpoint = torch.load(RESNET_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ ResNet50 loaded successfully")
    else:
        print(f"⚠️  ResNet50 checkpoint not found at {RESNET_CHECKPOINT}")
    
    return model.to(DEVICE).eval()


def load_vit():
    """Load Vision Transformer model with class expansion handling"""
    print("📦 Loading Vision Transformer model...")
    
    if not VIT_CHECKPOINT.exists():
        print(f"⚠️  ViT checkpoint not found at {VIT_CHECKPOINT}")
        return None
    
    vit_checkpoint = torch.load(VIT_CHECKPOINT, map_location=DEVICE)
    head_weight_shape = vit_checkpoint.get('head.weight', None)
    
    if head_weight_shape is not None:
        checkpoint_num_classes = head_weight_shape.shape[0]
        print(f"   Checkpoint has {checkpoint_num_classes} classes, model needs {NUM_CLASSES} classes")
        
        if checkpoint_num_classes != NUM_CLASSES:
            print(f"   Creating ViT with {checkpoint_num_classes} classes from checkpoint...")
            vit_model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=checkpoint_num_classes,
                in_chans=3
            )
            vit_model.load_state_dict(vit_checkpoint, strict=True)
            
            # Expand to 10 classes
            print(f"   Expanding model to {NUM_CLASSES} classes...")
            old_head = vit_model.head
            vit_model.head = nn.Linear(old_head.in_features, NUM_CLASSES)
            
            with torch.no_grad():
                vit_model.head.weight[:checkpoint_num_classes] = old_head.weight
                vit_model.head.bias[:checkpoint_num_classes] = old_head.bias
                vit_model.head.weight[checkpoint_num_classes:].normal_(0, 0.01)
                vit_model.head.bias[checkpoint_num_classes:].zero_()
        else:
            vit_model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=NUM_CLASSES,
                in_chans=3
            )
            vit_model.load_state_dict(vit_checkpoint, strict=True)
    else:
        print("   Using pretrained ImageNet weights")
        vit_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=NUM_CLASSES,
            in_chans=3
        )
    
    print("✅ Vision Transformer loaded successfully")
    return vit_model.to(DEVICE).eval()


def load_stacking_meta_learner():
    """Load pre-trained stacking meta-learner (Logistic Regression)"""
    print("📦 Loading Stacking Meta-Learner...")
    
    if STACKING_META_LEARNER.exists():
        with open(STACKING_META_LEARNER, 'rb') as f:
            meta_learner = pickle.load(f)
        print("✅ Meta-learner loaded successfully")
        return meta_learner
    else:
        print(f"⚠️  Meta-learner not found at {STACKING_META_LEARNER}")
        print("   Will use simple averaging instead")
        return None


# Load models on startup
print("\n" + "="*60)
print("INITIALIZING MODELS")
print("="*60)
resnet_model = load_resnet50()
vit_model = load_vit()
meta_learner = load_stacking_meta_learner()
print("="*60 + "\n")

# ==================== PREDICTION FUNCTIONS ====================
@torch.no_grad()
def get_predictions(image_tensor: torch.Tensor) -> tuple:
    """
    Get predictions from both ResNet50 and ViT models
    Returns: (resnet_logits, vit_logits, resnet_probs, vit_probs)
    """
    image_tensor = image_tensor.to(DEVICE)
    
    resnet_out = resnet_model(image_tensor)
    resnet_logits = resnet_out.cpu().numpy()
    resnet_probs = torch.softmax(torch.from_numpy(resnet_logits), dim=-1).numpy()
    
    vit_out = vit_model(image_tensor)
    vit_logits = vit_out.cpu().numpy()
    vit_probs = torch.softmax(torch.from_numpy(vit_logits), dim=-1).numpy()
    
    return resnet_logits, vit_logits, resnet_probs, vit_probs


def get_ensemble_prediction(
    resnet_logits: np.ndarray,
    vit_logits: np.ndarray,
    resnet_probs: np.ndarray,
    vit_probs: np.ndarray
) -> Dict:
    """
    Combine predictions using Stacking Ensemble with Logistic Regression
    or weighted voting if meta-learner is not available
    """
    if meta_learner is not None:
        # Stacking Ensemble
        meta_features = np.hstack([resnet_logits, vit_logits])
        ensemble_probs = meta_learner.predict_proba(meta_features)
        ensemble_method = "Stacking Ensemble"
    else:
        # Weighted Soft Voting (optimal weights from tuning: 0.75, 0.25)
        ensemble_probs = (0.75 * resnet_probs + 0.25 * vit_probs)
        ensemble_probs = ensemble_probs / ensemble_probs.sum()
        ensemble_method = "Weighted Soft Voting"
    
    ensemble_pred = np.argmax(ensemble_probs, axis=-1)[0]
    ensemble_confidence = ensemble_probs[0, ensemble_pred]
    
    return {
        "method": ensemble_method,
        "prediction": CLASS_NAMES[ensemble_pred],
        "confidence": float(ensemble_confidence),
        "probabilities": {
            CLASS_NAMES[i]: float(ensemble_probs[0, i])
            for i in range(NUM_CLASSES)
        }
    }


# ==================== API ENDPOINTS ====================
@app.get("/", tags=["Health Check"])
async def root():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "api": "Animal & Plant Classification API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Health Check"])
async def health():
    """Detailed health check with model status"""
    return {
        "status": "healthy",
        "resnet50": "loaded" if resnet_model is not None else "failed",
        "vit": "loaded" if vit_model is not None else "failed",
        "meta_learner": "loaded" if meta_learner is not None else "not available",
        "ensemble_method": "Stacking" if meta_learner is not None else "Weighted Voting",
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES
    }


@app.get("/classes", tags=["Information"])
async def get_classes():
    """Get list of available classes"""
    return {
        "total_classes": NUM_CLASSES,
        "classes": CLASS_NAMES
    }


@app.post("/predict", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Predict animal/plant class from uploaded image
    
    Expected input: PNG, JPG, or JPEG image file
    Returns: Classification result with confidence scores
    """
    try:
        # Validate file type - be more flexible with content type
        allowed_types = ["image/png", "image/jpeg", "image/jpg"]
        if file.content_type and file.content_type not in allowed_types:
            # Also check file extension
            valid_extensions = [".png", ".jpg", ".jpeg"]
            if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Expected PNG or JPEG, got {file.content_type}"
                )
        
        # Read and process image
        image_data = await file.read()
        
        # Validate image data
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        image_tensor = test_transform(image).unsqueeze(0)
        
        # Get predictions from both models
        resnet_logits, vit_logits, resnet_probs, vit_probs = get_predictions(image_tensor)
        
        # Get ensemble prediction
        ensemble_result = get_ensemble_prediction(
            resnet_logits, vit_logits, resnet_probs, vit_probs
        )
        
        # Also return individual model predictions for reference
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "file_name": file.filename,
            "ensemble": ensemble_result,
            "individual_models": {
                "resnet50": {
                    "prediction": CLASS_NAMES[np.argmax(resnet_probs[0])],
                    "confidence": float(np.max(resnet_probs[0])),
                    "probabilities": {
                        CLASS_NAMES[i]: float(resnet_probs[0, i])
                        for i in range(NUM_CLASSES)
                    }
                },
                "vit": {
                    "prediction": CLASS_NAMES[np.argmax(vit_probs[0])],
                    "confidence": float(np.max(vit_probs[0])),
                    "probabilities": {
                        CLASS_NAMES[i]: float(vit_probs[0, i])
                        for i in range(NUM_CLASSES)
                    }
                }
            }
        }
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing image: {str(e)}"
        print(f"🔴 {error_msg}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict animal/plant classes from multiple images
    
    Expected input: List of PNG or JPEG image files
    Returns: List of classification results
    """
    try:
        results = []
        
        for file in files:
            if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "error": f"Invalid file type: {file.content_type}"
                })
                continue
            
            try:
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_tensor = test_transform(image).unsqueeze(0)
                
                resnet_logits, vit_logits, resnet_probs, vit_probs = get_predictions(image_tensor)
                ensemble_result = get_ensemble_prediction(
                    resnet_logits, vit_logits, resnet_probs, vit_probs
                )
                
                results.append({
                    "file_name": file.filename,
                    "status": "success",
                    "prediction": ensemble_result["prediction"],
                    "confidence": ensemble_result["confidence"],
                    "method": ensemble_result["method"]
                })
            
            except Exception as e:
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )


# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("STARTING FASTAPI SERVER")
    print("="*60)
    print("📍 Server running at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
