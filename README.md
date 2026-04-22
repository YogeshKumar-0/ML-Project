# 🐱 Cat vs Dog Image Classification using CNN

A binary image classification project that predicts whether a given image contains a **cat** or a **dog**, built using a Convolutional Neural Network (CNN) with TensorFlow and Keras on Google Colab.

---

## 📌 Project Overview

| Field | Details |
|---|---|
| **Course** | Machine Learning with Python (CBECPE360) |
| **University** | Pimpri Chinchwad University, School of Engineering and Technology |
| **Semester** | VI (2025–2026) |
| **Platform** | Google Colab |
| **Framework** | TensorFlow / Keras |

---

## 📁 Dataset

- **Source:** [Kaggle — Cats and Dogs Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification)
- **License:** CC0-1.0
- **Size:** ~64.4 MB
- **Training images:** 557 (2 classes)
- **Validation images:** 140 (2 classes)
- **Classes:** `Cat (0)` / `Dog (1)`

---

## 🧠 Model Architecture

```
Model: "sequential"
─────────────────────────────────────────────────────
Layer (type)              Output Shape        Param #
─────────────────────────────────────────────────────
Conv2D (32, 3×3, relu)    (None, 254, 254, 32)    896
MaxPooling2D (2×2)        (None, 127, 127, 32)      0
Conv2D (64, 3×3, relu)    (None, 125, 125, 64)  18,496
MaxPooling2D (2×2)        (None, 62,  62,  64)      0
Conv2D (128, 3×3, relu)   (None, 60,  60, 128)  73,856
MaxPooling2D (2×2)        (None, 30,  30, 128)      0
Flatten                   (None, 115200)             0
Dense (128, relu)         (None, 128)       14,745,728
Dense (64, relu)          (None, 64)             8,256
Dense (1, sigmoid)        (None, 1)                65
─────────────────────────────────────────────────────
Total params: 14,847,297 (56.64 MB)
```

> **Output:** Sigmoid activation → value near `0` = Cat, value near `1` = Dog

---

## ⚙️ Pipeline

```
Kaggle API Download
      ↓
Extract ZIP → /content/train & /content/test
      ↓
Resize to 256×256, Normalize (÷255), Cast to float32
      ↓
Build Sequential CNN (3 Conv blocks + Dense layers)
      ↓
Compile: Adam | binary_crossentropy | accuracy
      ↓
Train: 10 epochs | batch_size=32
      ↓
Plot Accuracy & Loss curves
      ↓
Predict on test image using OpenCV
```

---

## 📊 Results

| Metric | Training | Validation |
|---|---|---|
| **Accuracy** | 98.20% | 66.43% |
| **Loss** | 0.0660 | 1.6396 |

### Accuracy Curve
![Accuracy](plots/accuracy_plot.png)

### Loss Curve
![Loss](plots/loss_plot.png)

> ⚠️ **Overfitting observed** — training accuracy reaches ~98% while validation accuracy plateaus at ~66%. The validation loss starts increasing from epoch 4 onwards, indicating the model memorizes training data rather than generalizing.

---

## 🔍 Sample Prediction

```python
test_image = cv2.imread('/content/dog.jpg.webp')
test_image = cv2.resize(test_image, (256, 256))
test_input = test_image.reshape(1, 256, 256, 3)
model.predict(test_input)
# Output: array([[1.]], dtype=float32)  →  Predicted: DOG ✅
```

---

## 🛠️ Tech Stack

| Tool | Version |
|---|---|
| Python | 3.12 |
| TensorFlow / Keras | 2.20.0 |
| NumPy | 2.0.2 |
| OpenCV | 4.13.0 |
| Matplotlib | 3.x |
| Google Colab | — |
| Kaggle API | — |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cat-vs-dog-cnn.git
cd cat-vs-dog-cnn
```

### 2. Set up Kaggle API
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download the dataset
```bash
kaggle datasets download -d samuelcortinhas/cats-and-dogs-image-classification
unzip cats-and-dogs-image-classification.zip -d /content
```

### 4. Open in Google Colab
Upload `Cat_Dog.ipynb` to [Google Colab](https://colab.research.google.com) and run all cells, **or** click the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MutrXJ1PV4rn7KF11tBOufhTUxFbhGZG)

---

## 📂 Repository Structure

```
cat-vs-dog-cnn/
│
├── Cat_Dog.ipynb          # Main Colab notebook
├── README.md              # This file
│
├── plots/
│   ├── accuracy_plot.png  # Training vs Validation accuracy
│   └── loss_plot.png      # Training vs Validation loss
│
└── sample/
    └── dog.jpg.webp       # Sample test image
```

---

## 🔮 Future Improvements

- [ ] Add **Dropout** and **Batch Normalization** to reduce overfitting
- [ ] Apply **Data Augmentation** (flip, zoom, rotation) for better generalization
- [ ] Use **Transfer Learning** (MobileNetV2 / VGG16) for higher accuracy
- [ ] Expand to **multi-class animal classification**
- [ ] Deploy as a web app using **Streamlit** or **Flask**

---

## 📜 License

This project is for academic purposes under **Pimpri Chinchwad University**.  
Dataset licensed under [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/).

---

> Made with ❤️ for ML Mini Project — Sem VI, PCU SET 2025–26
