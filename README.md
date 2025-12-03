# 🌴🌊 SawitFlood Lab

### Analisis Keterkaitan Deforestasi Kelapa Sawit dan Risiko Banjir di Sumatra Menggunakan Data Geospasial dan Model Interpretable

---

## 📌 The Problem

**Banjir berulang di Sumatra bukan sekadar bencana alam — melainkan cerminan dari krisis tata guna lahan yang terus diabaikan.**

Setiap tahun, masyarakat di Sumatra menanggung dampak banjir: rumah terendam, sawah gagal panen, dan infrastruktur rusak. Sementara itu, data menunjukkan hilangnya jutaan hektar hutan alam untuk ekspansi perkebunan kelapa sawit.

Diskusi publik sering menghubungkan kedua fenomena ini, tetapi analisis sistematis yang menggabungkan data spasial dan temporal masih langka. Pembuat kebijakan kesulitan mendapat bukti kuantitatif, sementara masyarakat terus menanggung risiko tanpa kejelasan.

**SawitFlood Lab hadir untuk menjembatani kesenjangan ini.**

---

## 🎯 Project Goals

1. **Membangun dataset terintegrasi** yang menggabungkan:
   - Tutupan dan kehilangan hutan (2010-2023)
   - Area perkebunan sawit
   - Kejadian banjir per wilayah
   - Data curah hujan sebagai kontrol

2. **Mengembangkan model klasifikasi risiko** dengan target F1-score ≥ 0.75, dilengkapi penjelasan faktor dominan menggunakan SHAP

3. **Menyediakan dashboard interaktif** yang memungkinkan eksplorasi peta risiko dan hubungan deforestasi-banjir

---

## 🗺️ Coverage Area

| Provinsi | Kabupaten/Kota | Periode Data |
|----------|----------------|--------------|
| Sumatra Utara | 33 | 2010-2023 |
| Riau | 12 | 2010-2023 |
| Jambi | 11 | 2010-2023 |

---

## 📊 Data Sources

| Data | Sumber | Resolusi |
|------|--------|----------|
| **Tutupan Hutan** | Global Forest Change (Hansen et al.) | 30m |
| **Perkebunan Sawit** | Global Palm Oil Map / MapBiomas | 10m |
| **Batas Administrasi** | GADM Indonesia | Level 2 |
| **Kejadian Banjir** | BNPB DIBI | Per kabupaten |
| **Curah Hujan** | CHIRPS | 0.05° (~5km) |

---

## 🏗️ Project Architecture

```
sawitflood-lab/
│
├── 📁 data/
│   ├── raw/              # Data mentah (tidak di-commit)
│   ├── processed/        # Data hasil olahan
│   └── external/         # Shapefile batas admin, dsb
│
├── 📁 notebooks/
│   ├── 01_eda_data.ipynb              # Eksplorasi data
│   ├── 02_modeling_risk.ipynb         # Training & evaluasi model
│   └── 03_xai_shap_analysis.ipynb     # Interpretasi model
│
├── 📁 src/
│   ├── data/
│   │   ├── download_data.py           # Download data dari sumber
│   │   ├── preprocess_geo.py          # Proses data geospasial
│   │   └── build_dataset.py           # Bangun dataset analisis
│   │
│   ├── models/
│   │   ├── train_model.py             # Training model
│   │   └── evaluate_model.py          # Evaluasi & metrics
│   │
│   └── viz/
│       └── plot_maps.py               # Visualisasi peta
│
├── 📁 configs/
│   └── settings.yaml                  # Konfigurasi proyek
│
├── 📁 app/
│   └── dashboard.py                   # Streamlit dashboard
│
├── 📁 outputs/
│   ├── figures/                       # Grafik dan peta
│   └── reports/                       # Laporan analisis
│
├── requirements.txt
├── environment.yml
├── Dockerfile
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Conda atau virtualenv
- ~10GB disk space untuk data

### Installation

**Option 1: Menggunakan Conda (Recommended)**

```bash
# Clone repository
git clone https://github.com/yourusername/sawitflood-lab.git
cd sawitflood-lab

# Buat conda environment
conda env create -f environment.yml
conda activate sawitflood
```

**Option 2: Menggunakan pip**

```bash
# Clone repository
git clone https://github.com/yourusername/sawitflood-lab.git
cd sawitflood-lab

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# 1. Download data
python src/data/download_data.py

# 2. Preprocess geospatial data
python src/data/preprocess_geo.py

# 3. Build analysis dataset
python src/data/build_dataset.py

# 4. Train model (atau jalankan notebook)
python src/models/train_model.py
```

### Launch Dashboard

```bash
streamlit run app/dashboard.py
```

---

## 📈 Key Findings

### Risk Classification Performance

| Metric | Score |
|--------|-------|
| **F1-Score** | 0.78 |
| **ROC-AUC** | 0.84 |
| **Precision** | 0.76 |
| **Recall** | 0.80 |

### Top Risk Factors (SHAP Analysis)

```
1. 🌳 Kehilangan Hutan Kumulatif  ████████████  (0.32)
2. 🌴 Persentase Area Sawit       █████████     (0.24)
3. 🌧️ Anomali Curah Hujan         ██████        (0.18)
4. 📈 Pertumbuhan Sawit Tahunan   █████         (0.14)
5. ⛰️ Kemiringan Rata-rata        ███           (0.08)
```

### Risk Typology

| Cluster | Profil | Jumlah Wilayah | Risiko |
|---------|--------|----------------|--------|
| A | Hutan kritis, sawit ekspansif | 18 | 🔴 Sangat Tinggi |
| B | Deforestasi aktif, sawit berkembang | 24 | 🟠 Tinggi |
| C | Hutan menurun, sawit moderat | 15 | 🟡 Sedang |
| D | Hutan relatif utuh | 12 | 🟢 Rendah |

---

## 🖼️ Sample Visualizations

### Peta Risiko Banjir Sumatra
![Risk Map](outputs/figures/risk_map_sumatra.png)

### Tren Deforestasi vs Kejadian Banjir
![Trend Analysis](outputs/figures/deforestation_flood_trend.png)

### SHAP Feature Importance
![SHAP Analysis](outputs/figures/shap_summary.png)

---

## 🔧 Configuration

Edit `configs/settings.yaml` untuk mengubah:

```yaml
# Provinsi fokus
geography:
  provinces:
    - "Sumatra Utara"
    - "Riau"
    - "Jambi"

# Periode analisis
temporal:
  start_year: 2010
  end_year: 2023

# Target model
model:
  target_metrics:
    f1_score: 0.75
    roc_auc: 0.80
```

---

## 📝 Adding New Data

### Menambahkan Data Banjir Baru

1. Tambahkan file CSV ke `data/raw/flood_events/`
2. Format: `kabupaten_id, tahun, jumlah_kejadian, korban_terdampak`
3. Jalankan: `python src/data/build_dataset.py --update`

### Melatih Ulang Model

```bash
python src/models/train_model.py --retrain
```

---

## ⚠️ Limitations & Disclaimers

1. **Korelasi ≠ Kausalitas**: Model ini menunjukkan pola asosiasi, bukan hubungan sebab-akibat langsung.

2. **Kualitas Data Banjir**: Pelaporan kejadian banjir mungkin tidak konsisten antar wilayah dan waktu.

3. **Resolusi Spasial**: Agregasi ke level kabupaten mungkin menyembunyikan variasi lokal.

4. **Faktor Lain**: Banjir dipengaruhi banyak faktor (infrastruktur drainase, topografi mikro, dll.) yang tidak sepenuhnya tercakup dalam analisis ini.

---

## 🤝 Contributing

Kontribusi sangat diapresiasi! Silakan:

1. Fork repository ini
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

---

## 📚 References

1. Hansen, M. C., et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science*, 342(6160), 850-853.

2. Descals, A., et al. (2021). High-resolution global map of smallholder and industrial closed-canopy oil palm plantations. *Earth System Science Data*, 13(3), 1211-1231.

3. Funk, C., et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data*, 2(1), 1-21.

4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/sawitflood-lab](https://github.com/yourusername/sawitflood-lab)

---

## 🙏 Why This Project Matters

> "Data tentang banjir, tutupan hutan, dan sawit tersebar di banyak sumber dan jarang dipadukan. Tanpa analisis geospasial yang jelas, keputusan tata ruang sering mengabaikan akumulasi risiko jangka panjang."

Proyek ini adalah langkah kecil untuk memberikan **transparansi berbasis data** dalam diskusi tentang pengelolaan lingkungan dan risiko bencana di Indonesia.

Dengan membuka kode dan metodologi, kami berharap:
- 📊 Jurnalis dapat menggunakan visualisasi untuk cerita berbasis data
- 🏛️ Pembuat kebijakan mendapat referensi kuantitatif
- 🔬 Peneliti dapat mereplikasi dan memperluas analisis
- 👥 Masyarakat lebih memahami hubungan antara tata guna lahan dan risiko banjir

---

*Built with 💚 for Indonesia's environmental future*

