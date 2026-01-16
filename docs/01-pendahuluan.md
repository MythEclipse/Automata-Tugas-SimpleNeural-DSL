# SimpleNeural-DSL: Domain Specific Language untuk Konfigurasi Eksperimen Machine Learning

## 1. Pendahuluan

### 1.1 Latar Belakang

Dalam era kecerdasan buatan, Machine Learning (ML) telah menjadi komponen krusial dalam berbagai aplikasi, mulai dari prediksi harga, pengenalan gambar, hingga pemrosesan bahasa alami. Namun, untuk membangun model ML, praktisi data science harus menulis kode yang kompleks menggunakan library seperti TensorFlow, PyTorch, atau Scikit-learn.

Kompleksitas ini menciptakan hambatan bagi:
- **Pemula** yang ingin belajar ML tanpa harus menguasai pemrograman mendalam
- **Domain Expert** yang memahami bisnis tetapi tidak familiar dengan coding
- **Peneliti** yang ingin melakukan prototyping cepat tanpa boilerplate code

**SimpleNeural-DSL** hadir sebagai solusi dengan menyediakan bahasa pemrograman sederhana (Domain Specific Language) yang memungkinkan pengguna mendefinisikan arsitektur model ML secara deklaratif.

### 1.2 Tujuan Proyek

1. **Menyederhanakan Konfigurasi ML**: Pengguna dapat mendefinisikan model neural network tanpa menulis kode Python secara manual
2. **Menerapkan Teori Automata & Kompilasi**: Implementasi lexer, parser, semantic analyzer, dan code generator
3. **Menghasilkan Kode Production-Ready**: Output berupa kode Python yang dapat langsung dieksekusi
4. **Validasi Otomatis**: Deteksi error pada tahap kompilasi sebelum eksekusi

### 1.3 Ruang Lingkup

| Aspek | Cakupan |
|-------|---------|
| **Input** | File `.sndsl` (SimpleNeural DSL) |
| **Output** | File `.py` (Python dengan TensorFlow/Keras) |
| **Model Support** | Sequential Neural Networks, Dense Layers |
| **Framework Target** | TensorFlow 2.x / Keras |
| **Platform** | Cross-platform (Python 3.8+) |

### 1.4 Terminologi

| Istilah | Definisi |
|---------|----------|
| **DSL** | Domain Specific Language - bahasa pemrograman yang dirancang untuk domain tertentu |
| **Lexer** | Komponen yang memecah input menjadi token-token |
| **Parser** | Komponen yang menganalisis struktur sintaksis token |
| **AST** | Abstract Syntax Tree - representasi pohon dari kode sumber |
| **Semantic Analyzer** | Komponen yang memeriksa kebenaran makna program |
| **Code Generator** | Komponen yang menghasilkan kode target dari AST |

---

## 2. Analisis Kebutuhan

### 2.1 Kebutuhan Fungsional

| ID | Kebutuhan | Prioritas | Deskripsi |
|----|-----------|-----------|-----------|
| **FR-01** | Load Dataset | High | Sistem dapat membaca konfigurasi dataset dari file CSV |
| **FR-02** | Define Model | High | Pengguna dapat mendefinisikan arsitektur neural network |
| **FR-03** | Configure Layers | High | Pengguna dapat mengatur parameter setiap layer (units, activation) |
| **FR-04** | Set Optimizer | Medium | Pengguna dapat memilih dan mengonfigurasi optimizer |
| **FR-05** | Training Config | High | Pengguna dapat mengatur epochs, batch size, validation split |
| **FR-06** | Error Detection | High | Sistem mendeteksi error sintaks dan semantik |
| **FR-07** | Code Generation | High | Sistem menghasilkan kode Python yang valid |
| **FR-08** | Preprocessing | Medium | Dukungan untuk normalisasi dan train-test split |
| **FR-09** | Model Save | Low | Opsi untuk menyimpan model hasil training |
| **FR-10** | Metrics Display | Medium | Menampilkan metrik evaluasi model |

### 2.2 Kebutuhan Non-Fungsional

| ID | Kebutuhan | Spesifikasi |
|----|-----------|-------------|
| **NFR-01** | Performance | Kompilasi DSL ke Python < 1 detik untuk file < 1000 baris |
| **NFR-02** | Usability | Sintaks mudah dipahami oleh pemula dalam 30 menit |
| **NFR-03** | Reliability | Error message yang informatif dan actionable |
| **NFR-04** | Portability | Berjalan di Windows, macOS, Linux |
| **NFR-05** | Maintainability | Kode modular dengan separation of concerns |
| **NFR-06** | Extensibility | Mudah menambahkan layer type baru |

### 2.3 Business Rules

| ID | Rule | Deskripsi |
|----|------|-----------|
| **BR-01** | Valid Learning Rate | Learning rate harus dalam range 0.0001 - 1.0 |
| **BR-02** | Positive Units | Jumlah neuron (units) harus positif integer |
| **BR-03** | Valid Activation | Activation function harus dari daftar yang didukung |
| **BR-04** | Dataset Required | Model harus memiliki dataset yang terdefinisi sebelumnya |
| **BR-05** | At Least One Layer | Model minimal memiliki satu layer |
| **BR-06** | Output Layer Required | Model harus memiliki output layer yang sesuai dengan task |

---

## 3. User Stories

### 3.1 Daftar User Stories

| ID | Role | Story | Acceptance Criteria |
|----|------|-------|---------------------|
| **US-01** | Data Scientist | Sebagai data scientist, saya ingin mendefinisikan arsitektur NN dengan sintaks sederhana agar bisa prototyping cepat | Kode DSL 10 baris menghasilkan model functional |
| **US-02** | ML Engineer | Sebagai ML engineer, saya ingin mendapat error message yang jelas saat ada kesalahan konfigurasi | Error menunjukkan baris, kolom, dan saran perbaikan |
| **US-03** | Student | Sebagai mahasiswa, saya ingin belajar konsep NN tanpa kompleksitas Python | Sintaks mirip bahasa natural yang mudah dibaca |
| **US-04** | Researcher | Sebagai peneliti, saya ingin hasil kompilasi berupa kode Python yang clean | Output code mengikuti PEP8 dan best practices |
| **US-05** | Developer | Sebagai developer, saya ingin mengintegrasikan DSL ke pipeline CI/CD | CLI interface dengan exit codes yang proper |

---

*Dokumen ini adalah bagian pertama dari rancangan lengkap SimpleNeural-DSL. Lanjut ke dokumen berikutnya untuk Use Case dan Arsitektur Sistem.*
