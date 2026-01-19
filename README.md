# Perbandingan Model Pembelajaran Mesin untuk Peringkasan Ekstraktif

Proyek ini bertujuan untuk melakukan studi komparatif terhadap berbagai model pembelajaran mesin (Machine Learning) dalam konteks peringkasan teks ekstraktif. Tugas peringkasan ini diperlakukan sebagai masalah klasifikasi biner pada tingkat kalimat, di mana setiap kalimat diklasifikasikan sebagai bagian dari ringkasan atau tidak.

## Tujuan

Tujuan utama dari penelitian ini adalah untuk memprediksi apakah sebuah kalimat dalam dokumen sumber layak dimasukkan ke dalam ringkasan akhir. Hal ini dilakukan dengan mengekstraksi fitur-fitur relevan dari setiap kalimat dan melatih model untuk membedakan antara kalimat penting (ringkasan) dan kalimat pendukung lainnya.

## Model yang Dibandingkan

Studi ini membandingkan kinerja dari lima algoritma pembelajaran mesin berikut:

1.  **Support Vector Machine (SVM)** - Menjadi fokus utama penelitian.
2.  **Random Forest (RF)**
3.  **Logistic Regression (LR)**
4.  **K-Nearest Neighbors (KNN)**
5.  **Naive Bayes (NB)**

## Metodologi

Evaluasi model dilakukan dalam dua tahap:

1.  **Evaluasi Baseline:** Menguji model menggunakan hiperparameter default untuk mendapatkan tolok ukur kinerja awal.
2.  **Optimasi Hiperparameter:** Menggunakan **Optuna** untuk mencari kombinasi hiperparameter terbaik yang memaksimalkan metrik evaluasi.

Validasi silang (Cross-Validation) dengan metode *Stratified K-Fold* digunakan untuk memastikan hasil evaluasi yang kuat dan tidak bias.

## Metrik Evaluasi

Kinerja model diukur menggunakan beberapa metrik statistik, dengan penekanan utama pada:
*   **F1-Score:** Untuk menyeimbangkan antara Presisi (Precision) dan Recall, memastikan ringkasan yang dihasilkan relevan dan komprehensif.
*   **Cohen's Kappa:** Untuk mengukur tingkat kesepakatan antara prediksi model dan label sebenarnya, dengan memperhitungkan kemungkinan kesepakatan yang terjadi secara kebetulan.

Hasil dari studi ini diharapkan dapat memberikan wawasan mengenai model mana yang paling efektif dan tangguh untuk tugas peringkasan teks otomatis.
