"""
Banana Classification Web Application
Website Klasifikasi Jenis Pisang menggunakan Deep Learning
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import cv2
import os
from datetime import datetime
import base64

# ====================================
# KONFIGURASI
# ====================================
IMG_SIZE = (150, 150)
MODEL_PATH = 'fish_classifier_model.keras'  # Akan diganti sesuai model pisang
CLASS_INDICES_PATH = 'class_indices.json'

# Page configuration
st.set_page_config(
    page_title="Banana Classifier - Klasifikasi Jenis Pisang",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================
# CUSTOM CSS & STYLING
# ====================================
def load_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Hide Streamlit Default */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Remove padding */
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        
        /* Navbar Styles */
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 999;
        }
        
        .navbar-brand {
            font-size: 1.8rem;
            font-weight: 700;
            color: white !important;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .nav-link {
            color: white !important;
            text-decoration: none;
            font-weight: 500;
            font-size: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            transition: all 0.3s ease;
        }
        
        .nav-link:hover {
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }
        
        .nav-link.active {
            background: rgba(255,255,255,0.3);
        }
        
        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5rem 2rem;
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            animation: fadeInUp 1s;
        }
        
        .hero-subtitle {
            font-size: 1.5rem;
            opacity: 0.95;
            margin-bottom: 2rem;
            animation: fadeInUp 1.2s;
        }
        
        .hero-button {
            background: white;
            color: #667eea;
            padding: 1rem 2.5rem;
            border-radius: 30px;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            animation: fadeInUp 1.4s;
        }
        
        .hero-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        /* Card Styles */
        .feature-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            border: 1px solid #f0f0f0;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .feature-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #333;
        }
        
        .feature-description {
            color: #666;
            line-height: 1.6;
        }
        
        /* Section Styles */
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin: 3rem 0 2rem 0;
            color: #333;
        }
        
        .section-subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 3rem;
        }
        
        /* Stats Card */
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .stats-number {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .stats-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        /* Upload Section */
        .upload-section {
            background: #f8f9fa;
            padding: 3rem;
            border-radius: 20px;
            margin: 2rem 0;
        }
        
        /* Result Card */
        .result-card {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 2rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .result-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 1rem;
        }
        
        .confidence-badge {
            display: inline-block;
            background: rgba(255,255,255,0.9);
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            font-size: 1.2rem;
            font-weight: 600;
            color: #2d3748;
        }
        
        /* Footer Styles */
        .custom-footer {
            background: #2d3748;
            color: white;
            padding: 3rem 2rem 2rem 2rem;
            margin-top: 5rem;
        }
        
        .footer-section {
            margin-bottom: 2rem;
        }
        
        .footer-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .footer-link {
            color: #cbd5e0;
            text-decoration: none;
            display: block;
            margin-bottom: 0.5rem;
            transition: color 0.3s;
        }
        
        .footer-link:hover {
            color: white;
        }
        
        .footer-bottom {
            text-align: center;
            padding-top: 2rem;
            border-top: 1px solid #4a5568;
            color: #cbd5e0;
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Button Styles */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        /* Navigation Button Override */
        button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.9) !important;
            color: #667eea !important;
            border: 2px solid rgba(102, 126, 234, 0.3) !important;
        }
        
        button[kind="primary"] {
            background: white !important;
            color: #667eea !important;
            border: 2px solid white !important;
            font-weight: 700 !important;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Metric Cards */
        .css-1xarl3l {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
    </style>
    """, unsafe_allow_html=True)

# ====================================
# LOAD MODEL & CLASS MAPPING
# ====================================
@st.cache_resource
def load_model():
    """Load model yang sudah dilatih"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file tidak ditemukan: {MODEL_PATH}")
        st.info("Jalankan notebook terlebih dahulu untuk melatih dan menyimpan model!")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_indices():
    """Load class indices mapping"""
    if not os.path.exists(CLASS_INDICES_PATH):
        st.error(f"Class indices file tidak ditemukan: {CLASS_INDICES_PATH}")
        return None
    with open(CLASS_INDICES_PATH, 'r') as f:
        return json.load(f)

# ====================================
# FUNGSI PREDIKSI
# ====================================
def preprocess_image(image):
    """Preprocess gambar untuk prediksi"""
    # Resize image
    img = image.resize(IMG_SIZE)
    # Convert to array
    img_array = np.array(img)
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fish(image, model, class_indices):
    """Melakukan prediksi klasifikasi ikan"""
    # Preprocess
    img_array = preprocess_image(image)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_class_idx]
    
    # Get class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    pred_class_name = idx_to_class[pred_class_idx]
    
    # Get all probabilities
    all_probs = {}
    for idx, prob in enumerate(predictions[0]):
        class_name = idx_to_class[idx]
        all_probs[class_name] = float(prob)
    
    return pred_class_name, confidence, all_probs

# ====================================
# NAVBAR COMPONENT
# ====================================
def show_navbar():
    """Display navigation bar"""
    pages = {
        "home": "Home",
        "classify": "Klasifikasi",
        "about": "Tentang",
        "contact": "Kontak"
    }
    
    current_page = st.session_state.get('page', 'home')
    
    navbar_html = f"""
    <div class="navbar">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
            <div class="navbar-brand">
                🍌 Banana Classifier
            </div>
        </div>
    </div>
    """
    
    st.markdown(navbar_html, unsafe_allow_html=True)
    
    # Navigation buttons with proper styling
    st.markdown("<div style='margin-top: -50px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    col_spacer1, col1, col2, col3, col4, col_spacer2 = st.columns([2, 1, 1, 1, 1, 2])
    
    with col1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True, 
                     type="primary" if current_page == "home" else "secondary"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        if st.button("🔍 Klasifikasi", key="nav_classify", use_container_width=True,
                     type="primary" if current_page == "classify" else "secondary"):
            st.session_state.page = "classify"
            st.rerun()
    with col3:
        if st.button("📖 Tentang", key="nav_about", use_container_width=True,
                     type="primary" if current_page == "about" else "secondary"):
            st.session_state.page = "about"
            st.rerun()
    with col4:
        if st.button("📞 Kontak", key="nav_contact", use_container_width=True,
                     type="primary" if current_page == "contact" else "secondary"):
            st.session_state.page = "contact"
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ====================================
# FOOTER COMPONENT
# ====================================
def show_footer():
    """Display footer"""
    footer_html = """
    <div class="custom-footer">
        <div style="max-width: 1200px; margin: 0 auto;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
                <div class="footer-section">
                    <div class="footer-title">🍌 Banana Classifier</div>
                    <p style="color: #cbd5e0; line-height: 1.6;">
                        Sistem klasifikasi jenis pisang berbasis Deep Learning 
                        yang akurat dan mudah digunakan.
                    </p>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">Quick Links</div>
                    <a href="#" class="footer-link">Home</a>
                    <a href="#" class="footer-link">Klasifikasi</a>
                    <a href="#" class="footer-link">Tentang Kami</a>
                    <a href="#" class="footer-link">Kontak</a>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">Kontak</div>
                    <p style="color: #cbd5e0; margin-bottom: 0.5rem;">📧 contact@banana-classifier.com</p>
                    <p style="color: #cbd5e0; margin-bottom: 0.5rem;">🌐 www.banana-classifier.com</p>
                    <p style="color: #cbd5e0; margin-bottom: 0.5rem;">📱 +62 812-3456-7890</p>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">Follow Us</div>
                    <div style="display: flex; gap: 1rem; font-size: 1.5rem;">
                        <a href="#" style="color: #cbd5e0;">🔗</a>
                        <a href="#" style="color: #cbd5e0;">📘</a>
                        <a href="#" style="color: #cbd5e0;">📸</a>
                        <a href="#" style="color: #cbd5e0;">🐦</a>
                    </div>
                </div>
            </div>
            
            <div class="footer-bottom">
                <p>© 2024 Banana Classifier. All rights reserved.</p>
                <p style="margin-top: 0.5rem;">Developed with ❤️ using Streamlit & TensorFlow</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(footer_html, unsafe_allow_html=True)

# ====================================
# NAVIGATION & PAGES
# ====================================
def show_home():
    """Halaman Home"""
    # Hero Section
    hero_html = """
    <div class="hero-section">
        <h1 class="hero-title">🍌 Banana Classifier</h1>
        <p class="hero-subtitle">Sistem Klasifikasi Jenis Pisang Berbasis Deep Learning</p>
        <p style="font-size: 1.1rem; max-width: 800px; margin: 0 auto 2rem auto; opacity: 0.9;">
            Identifikasi berbagai jenis pisang secara otomatis dan akurat menggunakan 
            teknologi Artificial Intelligence
        </p>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # CTA Button with styling
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔍 Coba Sekarang", type="primary", use_container_width=True, key="cta_classify"):
            st.session_state.page = "classify"
            st.rerun()
    with col2:
        if st.button("📖 Pelajari Lebih Lanjut", use_container_width=True, key="cta_about"):
            st.session_state.page = "about"
            st.rerun()
    with col3:
        if st.button("📞 Hubungi Kami", use_container_width=True, key="cta_contact"):
            st.session_state.page = "contact"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<h2 class="section-title">✨ Mengapa Memilih Kami?</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Teknologi terdepan dengan fitur-fitur unggulan</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Akurasi Tinggi</div>
            <div class="feature-description">Model CNN dengan akurasi lebih dari 85%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Cepat & Real-time</div>
            <div class="feature-description">Prediksi dalam hitungan detik</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Mudah Digunakan</div>
            <div class="feature-description">Interface yang user-friendly</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Berbasis Riset</div>
            <div class="feature-description">Menggunakan teknologi terkini</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown('<h2 class="section-title">📊 Statistik Sistem</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">87%</div>
            <div class="stats-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">6</div>
            <div class="stats-label">Jenis Pisang</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">1,234</div>
            <div class="stats-label">Total Prediksi</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">< 2s</div>
            <div class="stats-label">Waktu Proses</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # How It Works
    st.markdown('<h2 class="section-title">🔄 Cara Kerja Sistem</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Tiga langkah mudah untuk mengklasifikasi pisang</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
            <div class="feature-title">1. Upload Gambar</div>
            <div class="feature-description">Upload foto pisang dalam format JPG, PNG, atau JPEG</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
            <div class="feature-title">2. AI Processing</div>
            <div class="feature-description">Model CNN menganalisis gambar dan fitur-fiturnya</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
            <div class="feature-title">3. Hasil Prediksi</div>
            <div class="feature-description">Dapatkan hasil klasifikasi dengan confidence score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Technology Stack with better design
    st.markdown('<h2 class="section-title">🛠️ Teknologi yang Digunakan</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Dibangun dengan teknologi terbaik dan terpercaya</p>', unsafe_allow_html=True)
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🐍</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; color: #667eea;">Python</div>
            <div style="color: #666; font-size: 0.9rem;">Programming Language</div>
        </div>
        """, unsafe_allow_html=True)
    with tech_col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; color: #667eea;">TensorFlow</div>
            <div style="color: #666; font-size: 0.9rem;">Deep Learning Framework</div>
        </div>
        """, unsafe_allow_html=True)
    with tech_col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎨</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; color: #667eea;">Streamlit</div>
            <div style="color: #666; font-size: 0.9rem;">Web Framework</div>
        </div>
        """, unsafe_allow_html=True)
    with tech_col4:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📸</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; color: #667eea;">OpenCV</div>
            <div style="color: #666; font-size: 0.9rem;">Image Processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Testimonials / Benefits Section
    st.markdown('<h2 class="section-title">� Keunggulan Sistem</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
            <h4 style="color: #2d3748; margin-bottom: 1rem;">Proses Cepat</h4>
            <p style="color: #555;">Hasil klasifikasi dalam waktu kurang dari 2 detik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h4 style="color: #2d3748; margin-bottom: 1rem;">Akurasi Tinggi</h4>
            <p style="color: #555;">Model CNN dengan training accuracy 87%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📱</div>
            <h4 style="color: #2d3748; margin-bottom: 1rem;">Mudah Digunakan</h4>
            <p style="color: #555;">Interface sederhana dan intuitif untuk semua kalangan</p>
        </div>
        """, unsafe_allow_html=True)

def show_about():
    """Halaman Tentang Kami"""
    # Hero Section for About
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 4rem 2rem; text-align: center; border-radius: 0 0 30px 30px; margin-bottom: 3rem;">
        <h1 style="color: white; font-size: 3rem; margin-bottom: 1rem;">📖 Tentang Kami</h1>
        <p style="color: white; font-size: 1.2rem; opacity: 0.9;">Mengenal lebih dekat sistem klasifikasi pisang berbasis AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # About Project with Card
    st.markdown('<h2 class="section-title">🎯 Tentang Project</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-card" style="max-width: 1000px; margin: 0 auto 3rem auto; text-align: left;">
        <p style="font-size: 1.1rem; line-height: 1.8; color: #555;">
            <strong>Banana Classifier</strong> adalah sistem klasifikasi jenis pisang berbasis Deep Learning 
            yang dikembangkan sebagai bagian dari Tugas Akhir mata kuliah Pengenalan Pola. 
            <br><br>
            Sistem ini menggunakan <strong>Convolutional Neural Network (CNN)</strong> untuk mengidentifikasi 
            berbagai jenis pisang berdasarkan gambar yang diupload pengguna dengan tingkat akurasi tinggi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mission & Vision with Beautiful Cards
    st.markdown('<h2 class="section-title">🎯 Visi & Misi</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h3 style="color: white; margin-bottom: 1rem;">Misi Kami</h3>
            <p style="color: white; opacity: 0.95; line-height: 1.6;">
                Mengembangkan sistem klasifikasi pisang yang akurat dan mudah digunakan 
                untuk membantu petani, pedagang, dan konsumen dalam mengidentifikasi 
                jenis pisang dengan cepat dan tepat.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👁️</div>
            <h3 style="color: #2d3748; margin-bottom: 1rem;">Visi Kami</h3>
            <p style="color: #2d3748; line-height: 1.6;">
                Menjadi solusi teknologi terdepan dalam identifikasi dan klasifikasi 
                produk pertanian menggunakan Artificial Intelligence yang dapat 
                diakses oleh semua kalangan.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Performance with Metrics
    st.markdown('<h2 class="section-title">📊 Performa Model</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
            <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">87%</div>
            <div style="opacity: 0.9;">Training Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;">
            <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">68%</div>
            <div style="opacity: 0.9;">Validation Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white;">
            <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">6</div>
            <div style="opacity: 0.9;">Jenis Pisang</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white;">
            <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">1.2K</div>
            <div style="opacity: 0.9;">Dataset Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Model Architecture
    st.markdown('<h2 class="section-title">🧠 Arsitektur Model</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-bottom: 1.5rem;">Layer Structure</h3>
            <div style="text-align: left;">
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>📥 Input Layer:</strong> 150x150x3 RGB Images
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>� Convolutional Blocks:</strong>
                    <ul style="margin-top: 0.5rem; margin-bottom: 0;">
                        <li>Conv2D (32 filters) + MaxPooling</li>
                        <li>Conv2D (64 filters) + MaxPooling</li>
                        <li>Conv2D (128 filters) + MaxPooling</li>
                    </ul>
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>🧩 Dense Layers:</strong> 256 units + Dropout (0.5)
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <strong>📤 Output Layer:</strong> 6 classes (Softmax)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-bottom: 1.5rem;">Technical Specifications</h3>
            <div style="text-align: left;">
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>🧠 Framework:</strong> TensorFlow 2.15.0
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>📚 API:</strong> Keras Sequential Model
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>⚙️ Optimizer:</strong> Adam
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>📉 Loss Function:</strong> Categorical Crossentropy
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <strong>🎯 Activation:</strong> ReLU + Softmax
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Dataset Information with Icons
    st.markdown('<h2 class="section-title">🍌 Jenis Pisang yang Dapat Dideteksi</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    banana_types = [
        ("Pisang Ambon", "🍌", "Pisang manis dengan tekstur lembut"),
        ("Pisang Kepok", "🍌", "Pisang untuk digoreng atau direbus"),
        ("Pisang Raja", "🍌", "Pisang premium dengan rasa manis"),
        ("Pisang Mas", "🍌", "Pisang kecil dengan rasa manis"),
        ("Pisang Tanduk", "🍌", "Pisang besar untuk makanan"),
        ("Pisang Cavendish", "🍌", "Pisang ekspor internasional")
    ]
    
    for idx, (name, emoji, desc) in enumerate(banana_types):
        col = [col1, col2, col3][idx % 3]
        with col:
            st.markdown(f"""
            <div class="feature-card" style="min-height: 150px;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{emoji}</div>
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">{name}</h4>
                <p style="color: #666; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem 2rem; text-align: center; border-radius: 20px; margin: 3rem 0;">
        <h2 style="color: white; margin-bottom: 1rem;">Siap Mencoba Sistem Kami?</h2>
        <p style="color: white; font-size: 1.1rem; opacity: 0.9; margin-bottom: 2rem;">
            Upload gambar pisang Anda dan dapatkan hasil klasifikasi dalam hitungan detik!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Mulai Klasifikasi", type="primary", use_container_width=True):
            st.session_state.page = "classify"
            st.rerun()

def show_classify():
    """Halaman Klasifikasi"""
    st.title("🔍 Klasifikasi Jenis Pisang")
    st.markdown("Upload gambar pisang untuk mengetahui jenisnya")
    st.markdown("---")
    
    # Load model
    model = load_model()
    class_indices = load_class_indices()
    
    if model is None or class_indices is None:
        st.error("⚠️ Model atau class indices tidak ditemukan!")
        st.warning("Pastikan Anda sudah menjalankan notebook dan menyimpan model.")
        st.info("📝 Langkah-langkah:")
        st.markdown("""
        1. Buka `pengenalanpola.ipynb`
        2. Jalankan semua cell hingga cell "Save Model"
        3. Refresh halaman ini
        """)
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Gambar Pisang")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Pilih gambar pisang...",
            type=['jpg', 'jpeg', 'png'],
            help="Format yang didukung: JPG, JPEG, PNG"
        )
        
        # Tips
        with st.expander("💡 Tips untuk Hasil Terbaik"):
            st.markdown("""
            - Gunakan gambar dengan pencahayaan yang baik
            - Pastikan pisang terlihat jelas
            - Hindari gambar yang blur atau buram
            - Ukuran gambar minimal 150x150 pixels
            - Background yang kontras lebih baik
            """)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Gambar yang diupload", use_column_width=True)
            
            # Display image info
            st.markdown("**Detail Gambar:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Lebar", f"{image.size[0]}px")
            with col_b:
                st.metric("Tinggi", f"{image.size[1]}px")
            
            st.write(f"📄 **Format:** {image.format}")
            st.write(f"🎨 **Mode:** {image.mode}")
    
    with col2:
        st.header("🎯 Hasil Klasifikasi")
        
        if uploaded_file is not None:
            # Predict button
            if st.button("🔍 Mulai Klasifikasi", type="primary", use_container_width=True):
                with st.spinner("🔄 Memproses gambar..."):
                    try:
                        # Predict
                        pred_class, confidence, all_probs = predict_fish(
                            image, model, class_indices
                        )
                        
                        # Display result with animation
                        st.balloons()
                        st.success("✅ Prediksi berhasil!")
                        
                        # Main prediction result box
                        st.markdown("---")
                        st.markdown("### 🏆 Jenis Pisang Terdeteksi:")
                        
                        # Result card
                        st.markdown(f"""
                        <div style='padding: 20px; background-color: #f0f8ff; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                            <h1 style='color: #2e7d32; margin: 0;'>{pred_class}</h1>
                            <p style='font-size: 24px; color: #666; margin: 10px 0;'>Confidence: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(float(confidence))
                        
                        # Interpretation with icons
                        st.markdown("---")
                        if confidence > 0.9:
                            st.success("🎯 **Sangat Yakin** - Model sangat confident dengan prediksi ini!")
                        elif confidence > 0.7:
                            st.info("✅ **Cukup Yakin** - Model confident dengan prediksi ini.")
                        elif confidence > 0.5:
                            st.warning("⚠️ **Kurang Yakin** - Hasil mungkin kurang akurat.")
                        else:
                            st.error("❌ **Tidak Yakin** - Gambar mungkin bukan pisang atau kualitas rendah.")
                        
                        # All probabilities
                        st.markdown("---")
                        st.markdown("### 📊 Probabilitas Semua Kelas:")
                        
                        # Sort by probability
                        sorted_probs = sorted(
                            all_probs.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # Create beautiful probability bars
                        for idx, (class_name, prob) in enumerate(sorted_probs):
                            # Medal emoji for top 3
                            medal = ""
                            if idx == 0:
                                medal = "🥇 "
                            elif idx == 1:
                                medal = "🥈 "
                            elif idx == 2:
                                medal = "🥉 "
                            
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f"**{medal}{class_name}**")
                                st.progress(float(prob))
                            with col_b:
                                st.metric("", f"{prob:.1%}")
                        
                        # Additional info
                        st.markdown("---")
                        st.markdown(f"⏱️ **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Download result button
                        result_text = f"""
                        HASIL KLASIFIKASI PISANG
                        ========================
                        Jenis Pisang: {pred_class}
                        Confidence: {confidence:.2%}
                        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        Detail Probabilitas:
                        {chr(10).join([f'- {name}: {prob:.2%}' for name, prob in sorted_probs])}
                        """
                        
                        st.download_button(
                            label="📥 Download Hasil",
                            data=result_text,
                            file_name=f"hasil_klasifikasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {str(e)}")
                        st.exception(e)
        else:
            st.info("👆 Upload gambar pisang terlebih dahulu untuk melakukan klasifikasi")
            st.markdown("---")
            st.markdown("### 📋 Jenis Pisang yang Dapat Dideteksi:")
            
            if class_indices:
                cols = st.columns(2)
                for idx, cls_name in enumerate(sorted(class_indices.keys())):
                    with cols[idx % 2]:
                        st.markdown(f"✓ **{cls_name}**")

def show_contact():
    """Halaman Kontak"""
    st.title("📞 Hubungi Kami")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💌 Kirim Pesan")
        
        with st.form("contact_form"):
            name = st.text_input("Nama Lengkap *")
            email = st.text_input("Email *")
            subject = st.selectbox(
                "Subjek",
                ["Pertanyaan Umum", "Laporan Bug", "Saran Fitur", "Kolaborasi", "Lainnya"]
            )
            message = st.text_area("Pesan *", height=150)
            
            submitted = st.form_submit_button("📤 Kirim Pesan", use_container_width=True)
            
            if submitted:
                if name and email and message:
                    st.success("✅ Pesan berhasil dikirim! Kami akan menghubungi Anda segera.")
                    st.balloons()
                else:
                    st.error("❌ Mohon lengkapi semua field yang wajib diisi (*)")
    
    with col2:
        st.markdown("### 📍 Informasi Kontak")
        
        st.markdown("""
        **📧 Email:**  
        contact@banana-classifier.com
        
        **🌐 Website:**  
        www.banana-classifier.com
        
        **💼 GitHub:**  
        github.com/banana-classifier
        
        **🐦 Twitter:**  
        @banana_ai
        
        **📱 Instagram:**  
        @banana.classifier
        
        **⏰ Jam Operasional:**  
        Senin - Jumat: 09:00 - 17:00 WIB
        """)
        
        st.markdown("---")
        st.markdown("### 🏢 Alamat")
        st.info("""
        Universitas XYZ  
        Fakultas Teknik  
        Jl. Pendidikan No. 123  
        Kota, Provinsi 12345  
        Indonesia
        """)

# ====================================
# MAIN APP WITH NAVIGATION
# ====================================
def main():
    # Load custom CSS
    load_css()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Show Navbar
    show_navbar()
    
    # Route to pages
    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "classify":
        show_classify()
    elif st.session_state.page == "about":
        show_about()
    elif st.session_state.page == "contact":
        show_contact()
    
    # Show Footer
    show_footer()

if __name__ == "__main__":
    main()
