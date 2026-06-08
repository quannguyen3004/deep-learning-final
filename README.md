# 🐾 Animal & Plant Classification System

Ứng dụng Deep Learning để phân loại động vật và thực vật sử dụng Ensemble Learning (ResNet50 + Vision Transformer).

## 📋 Mục Lục

1. [Tính Năng](#tính-năng)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cài Đặt](#cài-đặt)
4. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
5. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
6. [API Documentation](#api-documentation)
7. [Kết Quả & Hiệu Năng](#kết-quả--hiệu-năng)
8. [Troubleshooting](#troubleshooting)

## 🎯 Tính Năng

✅ **Dự đoán chính xác** với độ tin cậy cao (86.4%)  
✅ **Ensemble Learning** kết hợp 2 mô hình mạnh  
✅ **API RESTful** với FastAPI  
✅ **Giao diện Web** thân thiện với Streamlit  
✅ **Dự đoán hàng loạt** (Batch Prediction)  
✅ **So sánh mô hình** chi tiết  
✅ **Hỗ trợ 10 lớp** động vật và thực vật  

## 📦 Yêu Cầu Hệ Thống

- **Python**: 3.9+
- **RAM**: 8GB+ (để load các mô hình)
- **GPU**: Optional (CUDA 11.8+ nếu có)
- **Disk**: 2GB+ (cho mô hình checkpoint)
- **OS**: macOS, Linux, Windows

## 🚀 Cài Đặt

### 1. Clone/Download dự án
```bash
cd /Users/mac/Desktop/Deep-Learning-final-main
```

### 2. Tạo Virtual Environment
```bash
# Sử dụng venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 4. Kiểm tra các file cần thiết
Đảm bảo bạn có các file sau trong thư mục project:
```
best_model_resnet50_finetune.pth    # ResNet50 weights
best_model_vit.pth                  # Vision Transformer weights
stacking_meta_learner.pkl           # Meta-learner cho Stacking
backend_api.py                      # FastAPI backend
frontend_app.py                     # Streamlit frontend
requirements.txt                    # Dependencies
```

## 📖 Hướng Dẫn Sử Dụng

### Bước 1: Khởi động Backend API

```bash
python3 backend_api.py
```

Output:
```
============================================================
INITIALIZING MODELS
============================================================
📦 Loading ResNet50 model...
✅ ResNet50 loaded successfully
📦 Loading Vision Transformer model...
✅ Vision Transformer loaded successfully
...
============================================================
STARTING FASTAPI SERVER
============================================================
📍 Server running at: http://localhost:8000
📖 API Documentation: http://localhost:8000/docs
============================================================
```

✅ Backend sẽ chạy trên: **http://localhost:8000**

### Bước 2: Khởi động Frontend Streamlit

**Mở terminal mới**, không tắt backend:
```bash
python3 -m streamlit run frontend_app.py
```

Output:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Bước 3: Sử dụng Ứng Dụng

1. Truy cập: **http://localhost:8501**
2. **Sidebar**: Kiểm tra trạng thái API
3. **Upload Image**: Chọn file PNG/JPG/JPEG
4. **Dự đoán**: Nhấn nút "🚀 Dự Đoán"
5. **Xem kết quả**: Hiển thị lớp dự đoán, độ tin cậy, xác suất cho từng lớp

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser (http://localhost:8501)   │
│                    Streamlit Frontend                    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP Request
                         │ (multipart/form-data with image)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│                 http://localhost:8000                    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Image Processing & Normalization                 │   │
│  └──────────────┬──────────────────────────────────┘   │
│                 │                                       │
│  ┌──────────────▼──────────────┐                       │
│  │   ResNet50 Model (83.85%)    │                       │
│  └──────────────┬───────────────┘                       │
│                 │                                       │
│  ┌──────────────▼──────────────┐                       │
│  │   Vision Transformer (55.2%)│                        │
│  └──────────────┬───────────────┘                       │
│                 │                                       │
│  ┌──────────────▼──────────────┐                       │
│  │  Stacking Meta-Learner       │                       │
│  │  (Logistic Regression)       │                       │
│  │  ▶ Combined (86.4%)          │                       │
│  └──────────────┬───────────────┘                       │
│                 │                                       │
│  ┌──────────────▼──────────────┐                       │
│  │    Return JSON Response      │                       │
│  │  - Prediction + Confidence   │                       │
│  │  - Per-class probabilities   │                       │
│  └──────────────┬───────────────┘                       │
└─────────────────┼────────────────────────────────────┘
                  │ HTTP Response (JSON)
                  │
┌─────────────────▼────────────────────────────────────┐
│                  Web Browser                          │
│  Display: Prediction + Visualization                 │
└──────────────────────────────────────────────────────┘
```

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "resnet50": "loaded",
  "vit": "loaded",
  "meta_learner": "loaded",
  "ensemble_method": "Stacking",
  "device": "cpu",
  "num_classes": 10,
  "classes": ["Amphibia", "Animalia", ...]
}
```

#### 2. Get Classes
```bash
GET /classes
```
**Response:**
```json
{
  "total_classes": 10,
  "classes": ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
}
```

#### 3. Single Image Prediction
```bash
POST /predict
Content-Type: multipart/form-data

Body:
- file: <image_file> (PNG, JPG, JPEG)
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2024-06-02T10:30:00.123456",
  "file_name": "image.jpg",
  "ensemble": {
    "method": "Stacking Ensemble",
    "prediction": "Aves",
    "confidence": 0.9234,
    "probabilities": {
      "Amphibia": 0.01,
      "Animalia": 0.02,
      "Arachnida": 0.01,
      "Aves": 0.9234,
      ...
    }
  },
  "individual_models": {
    "resnet50": {
      "prediction": "Aves",
      "confidence": 0.92,
      "probabilities": {...}
    },
    "vit": {
      "prediction": "Aves",
      "confidence": 0.85,
      "probabilities": {...}
    }
  }
}
```

#### 4. Batch Prediction
```bash
POST /predict-batch
Content-Type: multipart/form-data

Body:
- files: <multiple_image_files>
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2024-06-02T10:30:00.123456",
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "file_name": "image1.jpg",
      "status": "success",
      "prediction": "Aves",
      "confidence": 0.9234,
      "method": "Stacking Ensemble"
    },
    ...
  ]
}
```

### Interactive API Docs
Truy cập: **http://localhost:8000/docs**
- Swagger UI với tất cả endpoints
- Try it out trực tiếp từ browser

## 📊 Kết Quả & Hiệu Năng

### Model Performance on Test Set

| Model/Method | Accuracy | Precision | Recall | F1-Score |
|:---|---:|---:|---:|---:|
| **ResNet50** (Individual) | 83.85% | 83.97% | 83.85% | 83.87% |
| **Vision Transformer** (Individual) | 55.20% | 51.13% | 55.20% | 52.38% |
| **Soft Voting** | 72.00% | 73.48% | 72.00% | 71.79% |
| **Weighted Soft Voting** | 84.30% | 84.72% | 84.30% | 84.32% |
| **🏆 Stacking Ensemble** | **86.40%** | **86.43%** | **86.40%** | **86.40%** |

### Per-Class Performance (Stacking Ensemble)

| Class | Precision | Recall | F1-Score |
|:---|---:|---:|---:|
| Amphibia | 86.87% | 86.00% | 86.43% |
| Animalia | 81.00% | 81.00% | 81.00% |
| Arachnida | 91.05% | 86.50% | 88.72% |
| **Aves** | 93.14% | **95.00%** | **94.06%** |
| Fungi | 90.05% | 90.50% | 90.27% |
| Insecta | 89.34% | 88.00% | 88.67% |
| Mammalia | 85.99% | 89.00% | 87.47% |
| Mollusca | 81.03% | 79.00% | 80.00% |
| Plantae | 81.13% | 86.00% | 83.50% |
| Reptilia | 84.69% | 83.00% | 83.84% |

### Inference Speed

- **ResNet50**: ~50ms/image (GPU), ~200ms (CPU)
- **Vision Transformer**: ~80ms/image (GPU), ~300ms (CPU)
- **Stacking Ensemble**: ~150ms/image (GPU), ~500ms (CPU)

## 🐛 Troubleshooting

### ❌ "API không khả dụng"
**Giải pháp:**
1. Kiểm tra backend đang chạy: `python backend_api.py`
2. Kiểm tra port 8000 có bị chiếm không: `lsof -i :8000`
3. Nếu bị chiếm, kill process: `kill -9 <PID>`

### ❌ "ModuleNotFoundError: No module named 'torch'"
**Giải pháp:**
```bash
pip install -r requirements.txt
```

### ❌ "File not found: best_model_resnet50_finetune.pth"
**Giải pháp:**
1. Chắc chắn rằng bạn đã chạy `DL_ResNet.ipynb` đến hết
2. Model checkpoint phải lưu trong thư mục project
3. Kiểm tra đường dẫn trong `backend_api.py`

### ❌ "CUDA out of memory"
**Giải pháp:**
1. Sử dụng CPU: Backend sẽ tự động fallback
2. Giảm batch size trong code
3. Close các ứng dụng khác

### ⚠️ "Kết nối chậm"
**Giải pháp:**
1. Kiểm tra công suất CPU: `top` hoặc `Activity Monitor`
2. Kiểm tra RAM có đủ không
3. Restart backend nếu cần

## 📝 Cấu Trúc File

```
Deep-Learning-final-main/
├── backend_api.py                    # FastAPI Server
├── frontend_app.py                   # Streamlit Frontend
├── requirements.txt                  # Dependencies
├── README.md                         # Hướng dẫn này
├── helper_export.py                  # Helper để export model
├── stacking_meta_learner.pkl         # Meta-learner (auto-generated)
├── best_model_resnet50_finetune.pth # ResNet50 weights
├── best_model_vit.pth                # ViT weights
├── DL_caption.ipynb                  # ViT training notebook
├── DL_ResNet.ipynb                   # ResNet50 training notebook
├── DL_Optimization_Ensemble.ipynb    # Ensemble notebook
├── images/                           # Dataset images
│   ├── train/
│   └── test/
└── tabular/                          # CSV labels
    ├── merged_data.csv
    └── Test Dataset Labels.csv
```

## 🔐 Các File Quan Trọng

| File | Mô Tả | Kích Thước | Bắt Buộc |
|:---|:---|---:|:---:|
| `backend_api.py` | FastAPI server | ~8KB | ✅ |
| `frontend_app.py` | Streamlit app | ~12KB | ✅ |
| `best_model_resnet50_finetune.pth` | ResNet50 weights | ~100MB | ✅ |
| `best_model_vit.pth` | ViT weights | ~330MB | ✅ |
| `stacking_meta_learner.pkl` | Meta-learner | ~2.4KB | ✅ |

## 📞 Support & Contact

Nếu gặp vấn đề:
1. Check logs trong terminal
2. Đọc Troubleshooting section
3. Xem API docs tại http://localhost:8000/docs

## 📄 License

MIT License - Tự do sử dụng trong project của bạn

---

**Phiên bản**: 1.0.0  
**Cập nhật lần cuối**: Tháng 6, 2026  
**Trạng thái**: Production Ready ✅
