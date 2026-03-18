# 🚗 Car Damage Detection

A deep learning project that classifies whether a car has visible damage using a custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras — served through a clean **Streamlit web app**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📸 Demo

> Upload a car image → the model analyses it → instant damage assessment with confidence score.

| Damaged | Undamaged |
|---------|-----------|
| ⚠️ Red alert card with confidence % | ✅ Green card with confidence % |

---

## 🧠 Model Architecture

```
Input (128×128×3)
    │
    ├─ Conv2D(32) + ReLU + MaxPool
    ├─ Conv2D(64) + ReLU + MaxPool
    ├─ Conv2D(128) + ReLU + MaxPool
    │
    ├─ Flatten
    ├─ Dense(128) + ReLU + Dropout(0.3)
    ├─ Dense(64)  + ReLU
    ├─ Dense(32)  + ReLU
    │
    └─ Dense(2) + Softmax → [Damaged, Undamaged]
```

**Optimizer:** Adam  
**Loss:** Sparse Categorical Cross-Entropy  
**Input Size:** 128 × 128 pixels

---

## 📂 Project Structure

```
car-damage-detector/
│
├── app.py              ← Streamlit web application
├── train.py            ← Model training script
├── predict.py          ← CLI inference script
├── requirements.txt    ← Python dependencies
│
├── model/
│   └── car_damage_model.h5   ← Saved trained model (generate via train.py)
│
├── notebooks/
│   └── Car_Damage_Prediction.ipynb   ← Original exploration notebook
│
└── data1a/             ← Dataset (download from Kaggle)
    ├── training/
    │   ├── 00-damage/
    │   └── 01-whole/
    └── validation/
        ├── 00-damage/
        └── 01-whole/
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/car-damage-detector.git
cd car-damage-detector
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
The dataset is from Kaggle: [Car Damage Detection](https://www.kaggle.com/datasets/anujms/car-damage-detection)

```bash
kaggle datasets download anujms/car-damage-detection
unzip car-damage-detection.zip -d .
```

### 5. Train the model
```bash
python train.py --data_dir ./data1a --epochs 30
# Model saved to model/car_damage_model.h5
```

### 6. Launch the web app
```bash
streamlit run app.py
```

---

## 🖥️ CLI Inference

Run predictions directly from the terminal:

```bash
python predict.py --image path/to/car.jpg
```

Sample output:
```
========================================
  Image     : path/to/car.jpg
  Prediction: DAMAGED ⚠️
  Damaged   : 98.96%
  Undamaged :  1.04%
========================================
```

---

## 📊 Dataset

| Split | Images |
|-------|--------|
| Training | ~1,400 |
| Validation | ~400 |

**Classes:**
- `00-damage` → label `0` — Cars with visible damage
- `01-whole`  → label `1` — Undamaged cars

Source: [Kaggle — Car Damage Detection by anujms](https://www.kaggle.com/datasets/anujms/car-damage-detection)

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Image size | 128 × 128 |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping | patience = 5 |
| LR reduction | patience = 3, factor = 0.5 |
| Train/test split | 80/20 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building & training |
| OpenCV | Image preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Train/test split |
| Streamlit | Web application UI |
| Matplotlib | Training curves |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Your Name**  
[GitHub](https://github.com/samarthx4216) · [LinkedIn](www.linkedin.com/in/samarth-marathe-338163321)
