"""
Streamlit Frontend for Animal & Plant Classification
Provides interactive UI for image upload and classification
"""

import streamlit as st
import requests
import io
from PIL import Image
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🐾 Animal & Plant Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 5px;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 15px;
        border-radius: 5px;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONFIG ====================
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def check_api_health():
    """Check if API backend is running"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=TIMEOUT
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.warning(f"⚠️ API không khả dụng: {str(e)}")
        return None


def get_api_info():
    """Get API information"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/",
            timeout=TIMEOUT
        )
        return response.json()
    except:
        return None


def predict_image(image_file):
    """Send image to API for prediction"""
    try:
        # Send file directly to API - Streamlit UploadedFile works with requests
        files = {"file": (image_file.name, image_file.getbuffer(), image_file.type)}
        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Lỗi API: {response.status_code}")
            st.error(response.json().get("detail", "Unknown error"))
            return None
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout: API không phản hồi trong thời gian quy định")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Không thể kết nối tới API. Hãy chắc chắn backend đang chạy!")
        return None
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return None


def create_probability_chart(probabilities: dict, title: str):
    """Create interactive probability chart"""
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Sort by probability
    sorted_items = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    classes_sorted, probs_sorted = zip(*sorted_items)
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs_sorted,
            y=classes_sorted,
            orientation='h',
            marker=dict(
                color=probs_sorted,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"{p:.2%}" for p in probs_sorted],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Xác suất",
        yaxis_title="Lớp",
        height=400,
        showlegend=False,
        margin=dict(l=150)
    )
    
    return fig


# ==================== MAIN APP ====================
st.title("🐾 Ứng Dụng Phân Loại Động Vật & Thực Vật")
st.markdown("Sử dụng Deep Learning Ensemble (ResNet50 + Vision Transformer)")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Cài Đặt")
    
    # API Health Check
    st.subheader("📊 Trạng Thái API")
    api_health = check_api_health()
    
    if api_health:
        st.success("✅ API đang chạy")
        st.info(f"""
        **Thông tin:**
        - ResNet50: {api_health.get('resnet50', 'N/A')}
        - Vision Transformer: {api_health.get('vit', 'N/A')}
        - Meta-Learner: {api_health.get('meta_learner', 'N/A')}
        - Phương pháp: {api_health.get('ensemble_method', 'N/A')}
        - Device: {api_health.get('device', 'N/A')}
        """)
    else:
        st.error("❌ API không khả dụng!")
        st.warning("""
        Hãy chắc chắn rằng:
        1. Backend API đang chạy: `python backend_api.py`
        2. Chạy trên host: 0.0.0.0, port: 8000
        """)
    
    st.divider()
    
    # Information
    st.subheader("ℹ️ Thông Tin")
    if api_health and "classes" in api_health:
        num_cls = api_health.get("num_classes", len(api_health["classes"]))
        # Đưa toàn bộ 45 lớp vào khối Expander thu gọn để giữ thiết kế sạch sẽ
        with st.expander(f"📋 Danh sách {num_cls} lớp mục tiêu"):
            classes_list = api_health["classes"]
            # Tạo các đầu điểm Bullet point động cho 45 lớp
            st.markdown("\n".join([f"- **{cls}**" for cls in classes_list]))
    else:
        st.markdown("""
        **Danh sách lớp:**
        *(Chờ kết nối API để tải danh sách 45 lớp...)*
        """)
    
    st.divider()
    
    # Mode Selection
    mode = st.radio(
        "🎯 Chế độ",
        ["Dự đoán Đơn Lẻ", "Dự đoán Hàng Loạt"],
        help="Chọn dự đoán một ảnh hay nhiều ảnh cùng lúc"
    )

# ==================== MAIN CONTENT ====================
if not api_health:
    st.error("""
    ### ⚠️ Lỗi Kết Nối API
    
    Vui lòng khởi động backend API:
    ```bash
    python backend_api.py
    ```
    """)
else:
    # ==================== SINGLE PREDICTION MODE ====================
    if mode == "Dự đoán Đơn Lẻ":
        st.header("📤 Tải Ảnh Để Phân Loại")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Tải Ảnh")
            uploaded_file = st.file_uploader(
                "Chọn ảnh (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                help="Tải lên ảnh động vật hoặc thực vật để phân loại"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh được tải lên", use_container_width=True)
                
                # Get image info
                st.info(f"""
                **Thông tin ảnh:**
                - Tên: {uploaded_file.name}
                - Kích thước: {image.size}
                - Loại: {image.format}
                """)
        
        with col2:
            st.subheader("Kết Quả")
            
            if uploaded_file is not None:
                if st.button("🚀 Dự Đoán", use_container_width=True):
                    with st.spinner("⏳ Đang xử lý..."):
                        result = predict_image(uploaded_file)
                    
                    if result and result.get("status") == "success":
                        ensemble = result["ensemble"]
                        
                        # Main prediction
                        st.markdown("### 🎯 Kết Quả Dự Đoán")
                        col_pred, col_conf = st.columns([2, 1])
                        
                        with col_pred:
                            st.metric(
                                "Phân Loại",
                                ensemble["prediction"],
                                help="Dự đoán chính từ Stacking Ensemble"
                            )
                        
                        with col_conf:
                            confidence = ensemble["confidence"]
                            st.metric(
                                "Độ Tin Cậy",
                                f"{confidence:.2%}",
                                delta=f"{(confidence - 0.5) * 100:.1f}%" if confidence > 0.5 else None
                            )
                        
                        st.info(f"**Phương pháp:** {ensemble['method']}")
                        
                        # Ensemble probabilities
                        st.markdown("### 📊 Xác Suất Ensemble")
                        ensemble_fig = create_probability_chart(
                            ensemble["probabilities"],
                            "Ensemble Probabilities (Stacking)"
                        )
                        st.plotly_chart(ensemble_fig, use_container_width=True)
                        
                        # Individual models comparison
                        st.markdown("### 🔍 So Sánh Các Mô Hình")
                        
                        individual = result["individual_models"]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ResNet50")
                            st.metric(
                                "Dự Đoán",
                                individual["resnet50"]["prediction"],
                                individual["resnet50"]["confidence"]
                            )
                            resnet_fig = create_probability_chart(
                                individual["resnet50"]["probabilities"],
                                "ResNet50 Probabilities"
                            )
                            st.plotly_chart(resnet_fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Vision Transformer")
                            st.metric(
                                "Dự Đoán",
                                individual["vit"]["prediction"],
                                individual["vit"]["confidence"]
                            )
                            vit_fig = create_probability_chart(
                                individual["vit"]["probabilities"],
                                "Vision Transformer Probabilities"
                            )
                            st.plotly_chart(vit_fig, use_container_width=True)
                        
                        # Summary table
                        st.markdown("### 📋 Bảng So Sánh")
                        comparison_df = pd.DataFrame({
                            "Mô Hình": ["ResNet50", "ViT", "Ensemble (Stacking)"],
                            "Dự Đoán": [
                                individual["resnet50"]["prediction"],
                                individual["vit"]["prediction"],
                                ensemble["prediction"]
                            ],
                            "Độ Tin Cậy": [
                                f"{individual['resnet50']['confidence']:.2%}",
                                f"{individual['vit']['confidence']:.2%}",
                                f"{ensemble['confidence']:.2%}"
                            ]
                        })
                        st.dataframe(comparison_df, use_container_width=True)
            
            else:
                st.info("👆 Vui lòng tải lên một ảnh để bắt đầu")
    
    # ==================== BATCH PREDICTION MODE ====================
    else:
        st.header("📤 Dự Đoán Hàng Loạt")
        
        st.markdown("""
        Tải lên nhiều ảnh cùng lúc để phân loại hàng loạt.
        Kết quả sẽ hiển thị trong một bảng.
        """)
        
        uploaded_files = st.file_uploader(
            "Chọn nhiều ảnh (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Chọn từ 1 đến nhiều ảnh"
        )
        
        if uploaded_files:
            st.info(f"📁 Đã chọn {len(uploaded_files)} ảnh")
            
            if st.button("🚀 Dự Đoán Hàng Loạt", use_container_width=True):
                # Show progress
                progress_bar = st.progress(0)
                results_list = []
                
                with st.spinner("⏳ Đang xử lý..."):
                    for idx, file in enumerate(uploaded_files):
                        result = predict_image(file)
                        
                        if result and result.get("status") == "success":
                            ensemble = result["ensemble"]
                            results_list.append({
                                "Ảnh": file.name,
                                "Dự Đoán": ensemble["prediction"],
                                "Độ Tin Cậy": f"{ensemble['confidence']:.2%}",
                                "Phương Pháp": ensemble["method"]
                            })
                        else:
                            results_list.append({
                                "Ảnh": file.name,
                                "Dự Đoán": "Lỗi",
                                "Độ Tin Cậy": "-",
                                "Phương Pháp": "-"
                            })
                        
                        # Update progress
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                
                # Display results
                st.markdown("### 📊 Kết Quả Dự Đoán")
                results_df = pd.DataFrame(results_list)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.markdown("### 📈 Thống Kê")
                successful = sum(1 for r in results_list if r["Dự Đoán"] != "Lỗi")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tổng Ảnh", len(uploaded_files))
                col2.metric("Thành Công", successful)
                col3.metric("Lỗi", len(uploaded_files) - successful)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Tải Kết Quả (CSV)",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ==================== FOOTER ====================
st.divider()
st.markdown("""
---
**🐾 Animal & Plant Classification System**
- Backend: FastAPI + PyTorch
- Frontend: Streamlit
- Models: ResNet50 + Vision Transformer (Stacking Ensemble)
- Dataset: Animal & Plant Classification
""")
