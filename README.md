# Laporan Proyek Machine Learning - Aura Rachmawaty

## Domain Proyek

Latar Belakang:

Industri keuangan, khususnya lembaga pemberi pinjaman seperti bank dan fintech, menghadapi risiko utama berupa gagal bayar kredit (credit default). Risiko ini dapat menyebabkan kerugian finansial signifikan. Oleh karena itu, penting bagi lembaga keuangan untuk mengidentifikasi calon debitur yang berisiko gagal bayar sebelum kredit disetujui. Pemanfaatan machine learning untuk prediksi risiko kredit dapat menjadi solusi yang efektif karena algoritma mampu belajar dari data historis untuk membuat prediksi yang akurat.

Menurut laporan Biswas, etc. [1], lembaga keuangan yang mengadopsi pendekatan analitik prediktif mengalami peningkatan akurasi keputusan kredit hingga 35%. Selain itu, penelitian oleh Brown dan Mues [2] menunjukkan bahwa model machine learning seperti Random Forest, XGBoost, dan neural networks unggul dalam memprediksi risiko gagal bayar dibandingkan metode tradisional.

Referensi:

[1]    Biswas, S., Carson, B., Chung, V., Singh, S., & Thomas, R. (2020). AI in banking: Can banks meet the challenge? McKinsey, 01, 1–10.

[2]    Brown, I., & Mues, C. (2012). An experimental comparison of classification algorithms for imbalanced credit scoring data sets. Expert systems with applications, 39(3), 3446-3453.

## Business Understanding

### Problem Statements

- Bagaimana mengidentifikasi nasabah yang berisiko gagal membayar kredit menggunakan data historis?
- Bagaimana meminimalkan jumlah nasabah berisiko gagal bayar yang tidak terdeteksi oleh sistem?

### Goals

- Membangun model machine learning yang dapat memprediksi status pembayaran pinjaman (lancar atau gagal bayar).
- Mengoptimalkan model untuk meningkatkan recall pada kelas gagal bayar (agar lebih banyak kasus gagal bayar dapat terdeteksi).

### Solution statements
- Menggunakan algoritma Random Forest Classifier sebagai baseline model karena keunggulannya dalam menangani data tabular dan campuran numerik/kategorikal.
- Melakukan improvement dengan:
    - Imputasi nilai hilang (daripada drop data)
    - Penyeimbangan data menggunakan SMOTE
    - Tuning hyperparameter
    - Evaluasi model menggunakan metrik: accuracy, precision, recall, f1-score, dan ROC-AUC

## Data Understanding
Dataset yang digunakan berasal dari Kaggle dengan judul: Credit Risk Dataset. https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download

Dataset ini berisi informasi tentang nasabah dan pinjaman mereka, yang mencakup variabel demografis, keuangan, dan status pembayaran.

Ukuran dataset awal:

Jumlah baris (observasi): 32.581

Jumlah kolom (fitur): 12

Dataset ini berisi informasi mengenai peminjam, status keuangan, dan karakteristik pinjaman yang diajukan. Target dari dataset ini adalah loan_status, yaitu apakah peminjam mengalami gagal bayar (1) atau tidak (0).

Variabel-variabel pada dataset:
- person_age: Usia peminjam
- person_income: Pendapatan tahunan
- person_home_ownership: Status kepemilikan rumah
- person_emp_length: Lama bekerja (tahun)
- loan_intent: Tujuan pinjaman
- loan_grade: Peringkat pinjaman
- loan_amnt: Jumlah pinjaman
- loan_int_rate: Tingkat bunga
- loan_percent_income: Persentase pendapatan yang digunakan untuk membayar pinjaman
- cb_person_default_on_file: Riwayat gagal bayar sebelumnya
- cb_person_cred_hist_length: Lama riwayat kredit
- loan_status: Target — 1 (gagal bayar), 0 (tidak gagal bayar)

Kondisi Awal Dataset:
- Terdapat missing values pada kolom person_emp_length dan loan_int_rate
- Kelas target tidak seimbang: hanya sekitar 7–8% data berlabel gagal bayar
- Beberapa kolom bertipe kategorikal, perlu diubah ke format numerik

## Data Preparation

Beberapa tahap data preparation yang dilakukan:
1. Mengatasi Nilai Hilang
- Menghapus baris yang memiliki missing value (alternatif: imputasi rata-rata/median)
2. Encoding Variabel Kategorikal
- Menggunakan LabelEncoder untuk mengubah kolom kategorikal menjadi numerik
3. Pemisahan Fitur dan Label
- X = df.drop("loan_status"), y = df["loan_status"]
4. Split Data
- Menggunakan train_test_split dengan rasio 80:20 untuk pelatihan dan pengujian
5. Standarisasi
- Menggunakan StandardScaler agar data terdistribusi normal dan mempercepat konvergensi model

## Modeling

Model yang Digunakan
- Random Forest Classifier
    - n_estimators=100, random_state=42
    - Alasan: mampu menangani outlier, fitur kategorikal, dan overfitting lebih rendah dibanding decision tree tunggal

Cara Kerja Random Forest:

Random Forest adalah algoritma ensemble learning berbasis bagging (Bootstrap Aggregating). Algoritma ini membangun banyak decision tree dan menggabungkan hasil prediksinya (voting mayoritas untuk klasifikasi, rata-rata untuk regresi). Setiap decision tree dilatih pada subset data yang dipilih secara acak dengan pengembalian (bootstrap sample). Selain itu, pada setiap node dalam tree, hanya subset acak dari fitur yang dipertimbangkan untuk pemisahan (split). Hal ini memberikan dua keunggulan utama:
1. Mengurangi Overfitting
Karena setiap pohon melihat data dan fitur yang berbeda, kombinasi mereka cenderung lebih stabil dan generalizable dibanding satu decision tree.

2. Meningkatkan Akurasi
Voting dari banyak model yang tidak terlalu berkorelasi cenderung menghasilkan prediksi yang lebih akurat.

Visualisasi Sederhana Cara Kerja:
Data > Banyak bootstrap sample > Banyak decision tree > Hasil akhir = mayoritas voting (klasifikasi)

Alasan Pemilihan:
- Random Forest cocok untuk data tabular dengan campuran numerik dan kategorikal
- Dapat menangani outlier dan missing value dengan baik
- Memberikan insight pentingnya fitur (feature_importances_)


Kelebihan:
- Dapat menangani data besar dan fitur campuran
- Mendeteksi pentingnya fitur
- Tahan terhadap overfitting

Kekurangan:
- Interpretabilitas lebih rendah dari model linear
- Membutuhkan lebih banyak memori dan komputasi

## Evaluation
Metrik Evaluasi yang Digunakan:
1. Accuracy: Persentase prediksi yang benar terhadap total data
2. Precision: Seberapa tepat model dalam memprediksi kelas positif
3. Recall: Seberapa banyak kasus positif yang benar-benar terdeteksi
4. F1-Score: Harmonik dari precision dan recall
5. ROC-AUC: Mengukur kemampuan model membedakan kelas (semakin tinggi semakin baik)

Hasil Evaluasi:
Metrik	    Kelas 0 (Tidak Gagal Bayar)	Kelas 1 (Gagal Bayar)

Precision	0.92	                    0.97

Recall	    0.99	                    0.71

F1-Score	0.96	                    0.82

Accuracy	93%	

ROC-AUC	    0.93	

Interpretasi:
- Model sangat baik dalam mendeteksi nasabah yang membayar tepat waktu (recall 99%)
- Masih ada ruang perbaikan dalam mendeteksi gagal bayar (recall 71%)
- Tingkat precision yang tinggi (97%) pada kelas gagal bayar menunjukkan prediksi yang cukup akurat
- Nilai ROC-AUC 0.93 menunjukkan model cukup andal dalam membedakan kedua kelas
