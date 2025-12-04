# Indonesia Flood Risk Analytics (IFRA)

**Ketika banjir datang setiap tahun, kita sering menyalahkan cuaca. Tapi bagaimana kalau ada faktor lain yang selama ini luput dari perhatian?**

---

## Sekilas Proyek Ini

Proyek ini lahir dari satu pertanyaan sederhana: *"Kenapa banjir di Indonesia kok kayak jadwal rutin tahunan?"*

Setiap musim hujan, berita banjir muncul silih berganti. Rumah terendam, jalan lumpuh, dan korban berjatuhan. Kita cenderung pasrah menyebutnya "bencana alam" — seolah-olah memang sudah takdirnya begitu.

Padahal, data bercerita lain.

IFRA (Indonesia Flood Risk Analytics) mencoba mengulik hubungan antara perubahan tutupan lahan (baca: deforestasi dan ekspansi pertanian) dengan peningkatan risiko banjir. Bukan untuk menyalahkan satu pihak, tapi untuk memberikan bukti berbasis data yang selama ini langka tersedia.

**Yang proyek ini tawarkan:**

- Dataset terintegrasi dari BNPB, Global Forest Change, dan CHIRPS Rainfall
- Dashboard interaktif untuk eksplorasi pola banjir per wilayah
- Analisis korelasi antara deforestasi dan kejadian banjir
- Model machine learning untuk identifikasi faktor risiko dominan

Intinya: kita perlu berhenti nebak-nebak dan mulai pakai data.

---

## Mengapa Ini Penting

Diskusi soal lingkungan dan bencana di Indonesia seringkali mandek di level opini. Ada yang bilang deforestasi penyebab banjir, ada yang bilang tidak ada hubungannya. Debatnya seru, tapi datanya mana?

Proyek ini tidak mengklaim punya semua jawaban. Tapi setidaknya menyediakan:

- **Transparansi metodologi** — semua kode terbuka, bisa diaudit siapa saja
- **Data yang dapat direplikasi** — siapapun bisa jalankan ulang analisisnya
- **Visualisasi yang mudah dipahami** — tidak perlu jadi data scientist untuk mengerti hasilnya

Harapannya sederhana: keputusan tata ruang dan mitigasi bencana bisa lebih berbasis bukti, bukan sekadar asumsi.

---

## Sumber Data

| Data | Sumber | Keterangan |
|------|--------|------------|
| Kejadian Banjir | BNPB DIBI | Data bencana 2020-2025 |
| Tutupan Hutan | Global Forest Change (Hansen et al.) | Resolusi 30m |
| Curah Hujan | CHIRPS | Data klimatologi satelit |
| Batas Administrasi | GADM Indonesia | Level kabupaten/kota |

---

## Cara Instalasi

### Prasyarat

- Python 3.11 atau lebih baru
- `uv` package manager (rekomendasi) atau pip biasa
- Sekitar 5GB ruang disk untuk data

### Langkah Instalasi

**Opsi 1: Pakai uv (Lebih Cepat)**

```bash
# Clone repo
git clone https://github.com/bulgogipedas/sumatra-flood-risk-analysis.git
cd sumatra-flood-risk-analysis

# Install dependencies
uv sync

# Aktifkan environment
source .venv/bin/activate
```

**Opsi 2: Pakai pip Biasa**

```bash
# Clone repo
git clone https://github.com/bulgogipedas/sumatra-flood-risk-analysis.git
cd sumatra-flood-risk-analysis

# Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install
pip install -e .
```

---

## Penggunaan

### Jalankan Data Pipeline

Data mentah perlu diproses dulu sebelum bisa dianalisis:

```bash
# Siapkan data dasar
uv run python scripts/01_prepare_data.py

# Generate data forest cover (simulasi)
uv run python scripts/02_download_gfc.py

# Generate data rainfall (simulasi)
uv run python scripts/03_download_chirps.py

# Gabungkan semua data
uv run python scripts/04_merge_all_data.py
```

### Buka Dashboard

Ini bagian serunya — eksplorasi data lewat visualisasi interaktif:

```bash
uv run streamlit run app/dashboard.py
```

Buka browser ke `http://localhost:8501` dan mulai eksplorasi.

Dashboard terdiri dari beberapa halaman:
- **Home** — ringkasan eksekutif dan metrik utama
- **Flood Analysis** — pola dan distribusi kejadian banjir
- **Land Impact** — hubungan deforestasi dengan banjir
- **Regional** — perbandingan antar pulau dan provinsi
- **Data Explorer** — akses langsung ke data mentah

### Jalankan Notebook

Untuk analisis lebih mendalam:

```bash
jupyter notebook notebooks/
```

Tersedia tiga notebook:
1. `01_eda_data.ipynb` — eksplorasi data awal
2. `02_modeling_risk.ipynb` — training model klasifikasi
3. `03_xai_shap_analysis.ipynb` — interpretasi model dengan SHAP

---

## Struktur Proyek

```
sumatra-flood-risk-analysis/
├── app/
│   ├── dashboard.py          # Halaman utama
│   └── pages/                 # Multi-page Streamlit
├── data/
│   ├── raw/                   # Data mentah
│   ├── processed/             # Data hasil olahan
│   └── external/              # Shapefile, dsb
├── notebooks/                 # Jupyter notebooks
├── scripts/                   # Pipeline scripts
├── src/
│   ├── data/                  # Fungsi preprocessing
│   ├── models/                # Training & evaluasi
│   └── viz/                   # Visualisasi
├── configs/
│   └── settings.yaml          # Konfigurasi
└── pyproject.toml             # Dependencies
```

---

## Batasan dan Disclaimer

Sebelum terlalu excited, perlu diingat beberapa hal:

1. **Korelasi bukan kausalitas** — Model ini menunjukkan pola asosiasi, bukan membuktikan bahwa A menyebabkan B secara langsung.

2. **Kualitas data bervariasi** — Pelaporan banjir antar daerah tidak selalu konsisten. Ada yang rajin lapor, ada yang tidak.

3. **Agregasi ke level kabupaten** — Detail lokal mungkin hilang karena data diagregasi ke level administratif.

4. **Faktor lain tidak tercakup** — Infrastruktur drainase, topografi mikro, dan faktor sosial tidak dimasukkan dalam model.

Treat hasil analisis ini sebagai titik awal diskusi, bukan kebenaran final.

---

## Kontribusi

Proyek ini open source dan terbuka untuk kontribusi. Kalau kamu:

- Menemukan bug atau error
- Punya ide improvement
- Ingin menambahkan data dari sumber lain
- Atau sekadar mau ngobrol soal metodologi

Silakan:

1. Fork repository ini
2. Buat branch baru (`git checkout -b fitur-keren`)
3. Commit perubahan (`git commit -m 'Tambah fitur keren'`)
4. Push ke branch (`git push origin fitur-keren`)
5. Buka Pull Request

Atau kalau mau lebih santai, buka Issue aja untuk diskusi.

---

## Referensi

- Hansen, M. C., et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science*, 342(6160), 850-853.
- Funk, C., et al. (2015). The climate hazards infrared precipitation with stations. *Scientific Data*, 2(1), 1-21.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

---

## Lisensi

MIT License — bebas dipakai, dimodifikasi, dan didistribusikan. Lihat file `LICENSE` untuk detail.

---

## Penutup

Data tentang banjir, hutan, dan tata guna lahan di Indonesia sebetulnya ada. Masalahnya, tersebar di mana-mana dan jarang ada yang menggabungkannya jadi satu cerita yang utuh.

Proyek ini adalah upaya kecil untuk mengubah itu. Bukan untuk menyalahkan siapa-siapa, tapi untuk membuka ruang diskusi yang lebih berbasis fakta.

Kalau kamu jurnalis yang butuh visualisasi untuk liputan, peneliti yang ingin mereplikasi analisis, pembuat kebijakan yang perlu referensi data, atau warga biasa yang penasaran kenapa banjir terus terjadi — semoga proyek ini berguna.

**Clone repo-nya, jalankan dashboard-nya, dan lihat sendiri apa yang data katakan. Siapa tahu, insight-nya bisa mengubah cara kita memandang masalah banjir di Indonesia.**

---

*Proyek ini dibuat untuk masa depan lingkungan Indonesia yang lebih baik.*
