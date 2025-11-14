# 🫁 AI Pneumonia Detection System

A deep learning-based medical image classification system that detects pneumonia from chest X-ray images with **84.94% accuracy**.

![Accuracy](https://img.shields.io/badge/Accuracy-84.94%25-brightgreen)
![Model](https://img.shields.io/badge/Model-ResNet50-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train_balanced.py
```
This will create `best_pneumonia_model.h5` (138 MB) with 84.94% accuracy.

**Note:** Pre-trained model not included due to GitHub size limits. Train it yourself or download from [releases](https://github.com/shreyaslarer/pneumonia-detection-ai/releases).

### 3. Run Web Application
```bash
python app.py
```

### 4. Open Browser
Navigate to: `http://127.0.0.1:5000`

---

## 📁 Project Structure

```
eda/
├── app.py                          # Flask web application
├── train_balanced.py               # Model training script
├── best_pneumonia_model.h5         # Trained model (84.94% accuracy)
├── requirements.txt                # Python dependencies
├── PROJECT_EXPLANATION.md          # Detailed technical documentation
├── training_history.png            # Training visualization
├── templates/
│   └── index.html                  # Modern web interface
└── archive/
    └── chest_xray/                 # Dataset
        ├── train/                  # Training images
        └── test/                   # Test images
```

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 84.94% |
| **Normal Precision** | 76% |
| **Normal Recall** | 88% |
| **Pneumonia Precision** | 92% |
| **Pneumonia Recall** | 83% |
| **Inference Time** | ~0.5 seconds |

### Confusion Matrix
```
              Predicted
              Normal  Pneumonia
Actual Normal   206      28
       Pneumonia 66     324
```

---

## 🏗️ Architecture

- **Base Model:** ResNet50 (pre-trained on ImageNet)
- **Transfer Learning:** Fine-tuned last 15 layers
- **Input Size:** 150x150 RGB images
- **Output:** Binary classification (Normal/Pneumonia)

### Model Layers
```
ResNet50 (frozen first 35 layers)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(1) + Sigmoid
```

---

## 🔧 Key Features

### Backend
- ✅ ResNet50 CNN with transfer learning
- ✅ Class-weighted training for imbalanced data
- ✅ Data augmentation (rotation, zoom, flip)
- ✅ Flask REST API for predictions

### Frontend
- ✅ Modern animated UI with gradient background
- ✅ Drag & drop file upload
- ✅ Real-time image preview
- ✅ Animated confidence bar
- ✅ Color-coded results
- ✅ Fully responsive design

---

## 📊 Training Details

- **Dataset:** 5,216 training images (1,341 Normal + 3,875 Pneumonia)
- **Epochs:** 15
- **Batch Size:** 32
- **Optimizer:** Adam (learning rate: 0.0001)
- **Loss Function:** Binary Crossentropy
- **Class Weights:** Normal=1.94, Pneumonia=0.67

---

## 🛠️ Technologies Used

- **Deep Learning:** TensorFlow, Keras
- **Backend:** Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Image Processing:** PIL, NumPy
- **Visualization:** Matplotlib
- **Metrics:** scikit-learn

---

## 📖 Usage

### Upload X-Ray Image
1. Open the web application
2. Drag & drop or click to upload chest X-ray
3. Click "Analyze X-Ray"
4. View results with confidence score

### API Endpoint
```python
POST /predict
Content-Type: multipart/form-data
Body: file (image file)

Response:
{
    "prediction": "PNEUMONIA" | "NORMAL",
    "confidence": 0.92
}
```

---

## 📝 Notes

- This is an AI-assisted diagnostic tool
- Always consult healthcare professionals for medical decisions
- Model trained on chest X-ray dataset
- Best performance with clear, frontal chest X-rays

---

## 📄 License

Educational and research purposes only.

---

## 👨‍💻 Author

Built with ❤️ using AI and Deep Learning

---

For detailed technical explanation, see [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)
