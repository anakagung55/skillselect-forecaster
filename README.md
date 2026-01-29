# SkillSelect Forecaster 🚀

**SkillSelect Forecaster** adalah aplikasi analitik dan forecasting berbasis Streamlit untuk menganalisis data SkillSelect (EOI) dan memproyeksikan tren masa depan menggunakan model Prophet.

---

## 📌 Ringkasan (Overview)

Aplikasi ini ditujukan untuk:
- Menggabungkan dan membersihkan data SkillSelect menjadi satu sumber kebenaran.
- Menyediakan dashboard interaktif untuk analitik historis (leaderboard, breakdown poin, dll).
- Melakukan pelatihan model time-series (Prophet) secara on-the-fly untuk memproyeksikan permintaan EOI per pekerjaan (occupation).

Fitur utama:
- Halaman **Project Overview**, **Top Market Leaderboard**, dan **Specific Forecast & Trends**
- Filter dinamis (visa type, EOI status, points, period)
- Visualisasi interaktif dengan Plotly
- Prediksi 6 bulan menggunakan Prophet (dapat dikonfigurasi)

---

## 🧭 Table of Contents

1. [Persyaratan](#-persyaratan)
2. [Instalasi](#-instalasi)
3. [Menjalankan Aplikasi](#-menjalankan-aplikasi)
4. [Struktur Proyek & Penjelasan Kode](#-struktur-proyek--penjelasan-kode)
5. [Format Data yang Diharapkan](#-format-data-yang-diharapkan)
6. [Penjelasan Halaman Dashboard](#-penjelasan-halaman-dashboard)
7. [Deployment & Tips Produksi](#-deployment--tips-produksi)
8. [Troubleshooting & FAQ](#-troubleshooting--faq)
9. [Kontribusi](#-kontribusi)
10. [Lisensi](#-lisensi)

---

## ✅ Persyaratan

- Python 3.9+ (direkomendasikan 3.10/3.11)
- Sistem operasi: Windows/Mac/Linux
- Dependensi utama ada di: `requirements.txt`
  - `streamlit`, `pandas`, `prophet`, `plotly`, `pyarrow`, dll

Catatan: Instalasi `prophet` pada beberapa sistem (terutama Windows) mungkin memerlukan build toolchain atau conda. Lihat bagian Troubleshooting untuk tips.

---

## 🔧 Instalasi

1. Clone repositori:

```bash
git clone https://github.com/<your-org>/skillselect-forecaster.git
cd skillselect-forecaster
```

2. Buat virtual environment dan aktifkan:

- Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependensi:

```bash
pip install -r requirements.txt
```

Jika `prophet` gagal terinstal via pip, coba:

```bash
pip install prophet --pre
# atau gunakan conda (direkomendasikan untuk Windows):
conda install -c conda-forge prophet
```

---

## ▶ Menjalankan Aplikasi (Local)

1. Pastikan file data master tersedia dalam salah satu lokasi berikut (prioritas diurutkan):
   - `df_master.parquet`
   - `df_master.zip` (csv dalam zip)
   - `data/df_master.csv`

2. Jalankan Streamlit:

```bash
streamlit run app.py
```

Akses UI di `http://localhost:8501`.

---

## 🗂 Struktur Proyek & Penjelasan Kode

- `app.py` — Aplikasi Streamlit utama. Memuat data (dengan cache), menyediakan navigasi sidebar dan 3 halaman utama.
  - Bagian konfigurasi halaman dan custom CSS
  - Fungsi `get_master_data()` untuk membaca `parquet/csv` dan pra-proses ringan
  - Halaman: `Project Overview`, `Top Market Leaderboard`, `Specific Forecast & Trends`
  - Modul plotting menggunakan Plotly dan fungsi forecasting menggunakan `Prophet` (on-the-fly)

- `requirements.txt` — daftar dependensi Python.

Catatan kunci di `app.py`:
- Data diharapkan memiliki kolom kunci seperti: `ds` (datetime), `occupation`, `visa_type`, `eoi_status`, `points`, `count_eois`.
- Forecasting menggunakan `Prophet` dengan `seasonality_mode='multiplicative'` dan prediksi 6 bulan ke depan.

---

## 🧾 Format Data yang Diharapkan

Minimal kolom yang dibutuhkan:

- `ds` — tanggal (format ISO, mis. `YYYY-MM-DD`) — akan di-convert ke `datetime`
- `occupation` — nama pekerjaan / ANZSCO
- `count_eois` — jumlah EOI pada tanggal tersebut
- `visa_type` — tipe visa (mis. 189/190/491)
- `eoi_status` — status (mis. SUBMITTED)
- `points` — nilai poin (string/number)

Contoh baris:

```csv
ds,occupation,count_eois,visa_type,eoi_status,points
2024-01-01,Software Engineer,123,189,SUBMITTED,90
```

---

## 📊 Penjelasan Halaman Dashboard

1. Project Overview
   - Menampilkan ringkasan metrik, trend global EOI, dan tujuan proyek.
2. Top Market Leaderboard
   - Menyediakan filter (visa type, status, month) dan menampilkan top 15 ocupations berdasarkan volume EOI.
   - Menampilkan breakdown poin: dominant point, min, max.
3. Specific Forecast & Trends
   - Pilih occupation, lakukan filter, lihat breakdown historis.
   - Tombol **Generate AI Forecast** melatih Prophet on-the-fly dan menampilkan proyeksi + interval kepercayaan.

---

## 🚀 Deployment & Produksi

Beberapa opsi deployment:
- Streamlit Cloud (cepat & mudah untuk demo)
- Docker container
- VPS / cloud server (Gunakan gunicorn/uvicorn + reverse proxy jika ingin non-streamlit server)

Contoh Dockerfile minimal (opsional):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## ❗ Troubleshooting & FAQ

Q: Data tidak muncul / DataFrame kosong?
- Pastikan file `df_master.parquet` atau `data/df_master.csv` berada di directory project.
- Periksa header dan tipe kolom: `ds` harus parseable ke `datetime`.

Q: `prophet` gagal install di Windows?
- Coba install via `conda install -c conda-forge prophet` atau gunakan `pip install --upgrade build` lalu `pip install prophet`.

Q: Model Prophet error saat fit?
- Pastikan dataset untuk training memiliki >= 2 titik waktu (untuk prediksi sederhana setidaknya 2 bulan data).

---

## 🤝 Kontribusi

Sangat diterima! Beberapa pedoman singkat:
1. Fork repository dan buat branch feature/bugfix
2. Sertakan deskripsi perubahan dan contoh (jika perlu)
3. Buka Pull Request dan jelaskan testing yang sudah dilakukan

---

## 📜 Lisensi

Silakan tambahkan file `LICENSE` (mis. MIT) jika ingin membuka kode ini untuk penggunaan bebas. Saat ini belum ada file lisensi spesifik di repo ini.

---

## 📫 Kontak

Jika ada pertanyaan, fitur yang ingin ditambahkan, atau masalah instalasi, silakan buka Issue di repository atau hubungi maintainer repository.

---

Terima kasih sudah menggunakan **SkillSelect Forecaster**! ⭐
