# Fish Classification Web Application

## 📋 Persiapan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Notebook
Pastikan Anda sudah menjalankan `pengenalanpola.ipynb` hingga selesai untuk:
- Melatih model
- Menyimpan model (`fish_classifier_model.keras`)
- Menyimpan class indices (`class_indices.json`)

## 🚀 Cara Menjalankan Web App

### Metode 1: Streamlit (Rekomendasi - Paling Mudah)
```bash
streamlit run app_streamlit.py
```

Web app akan terbuka otomatis di browser pada `http://localhost:8501`

## 📱 Cara Menggunakan

1. Buka web app di browser
2. Klik tombol "Browse files" untuk upload gambar ikan
3. Klik tombol "Klasifikasi Gambar"
4. Lihat hasil prediksi dengan confidence score
5. Lihat probabilitas untuk semua kelas ikan

## 📂 Struktur File

```
TugasPengenalanPola_Final/
├── pengenalanpola.ipynb          # Notebook training
├── app_streamlit.py               # Web app Streamlit
├── requirements.txt               # Dependencies
├── fish_classifier_model.keras    # Model terlatih
├── class_indices.json             # Mapping kelas
├── train/                         # Dataset training
└── test/                          # Dataset testing
```

## 🎯 Fitur Web App

- ✅ Upload gambar (JPG, PNG, JPEG)
- ✅ Preview gambar yang diupload
- ✅ Prediksi klasifikasi dengan confidence score
- ✅ Tampilkan probabilitas semua kelas
- ✅ Progress bar untuk visualisasi confidence
- ✅ Interface yang clean dan user-friendly
- ✅ Responsive design

## 🔧 Troubleshooting

### Error: Model file tidak ditemukan
**Solusi:** Jalankan cell terakhir di notebook untuk menyimpan model

### Error: ModuleNotFoundError
**Solusi:** Install dependencies dengan `pip install -r requirements.txt`

### Error: Port already in use
**Solusi:** Gunakan port berbeda:
```bash
streamlit run app_streamlit.py --server.port 8502
```

## 📊 Model Information

- **Architecture:** Custom CNN (3 Conv blocks)
- **Input Size:** 150x150 pixels
- **Classes:** 6 jenis ikan (Anaji, Bichi, Champa, Deshi, Shagor, Shobri)
- **Framework:** TensorFlow/Keras

## 🌐 Deployment

Untuk deploy ke cloud (opsional):

### Streamlit Cloud (Gratis)
1. Push code ke GitHub
2. Daftar di https://share.streamlit.io
3. Connect repository
4. Deploy!

### Heroku
```bash
heroku create fish-classifier-app
git push heroku main
```

### Google Cloud Run
```bash
gcloud run deploy --source .
```

## 📝 Notes

- Pastikan model sudah dilatih sebelum menjalankan web app
- Upload gambar dengan resolusi yang baik untuk hasil optimal
- Model akan resize gambar ke 150x150 pixels secara otomatis
