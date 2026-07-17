# 🧬 Non-Invasive Blood Group Prediction Using Fingerprint Analysis

A deep learning-based system that predicts human blood groups using fingerprint images in a non-invasive manner.  
The project uses an EfficientNetB0 CNN model along with a Flask web application for real-time prediction.

---

## 📌 Overview

- Predicts blood group from fingerprint images  
- Uses EfficientNetB0 (transfer learning)  
- Supports 8 classes: A+, A-, B+, B-, AB+, AB-, O+, O-  
- Provides real-time predictions via web interface  
- Achieved ~90% test accuracy  

---

## ✨ Features

### 1. Deep Learning Prediction
- EfficientNetB0-based CNN model  
- Outputs predicted label and confidence score  

### 2. Fingerprint Quality Auto-Correction
- Noise reduction (Gaussian Blur)  
- Contrast enhancement (CLAHE)  
- Improves prediction accuracy on low-quality images  

### 3. Prediction History Tracking
- Stores recent predictions with timestamp  
- Includes predicted label and confidence score  
- Lightweight in-memory implementation  

### 4. Web-Based Interface
- Upload fingerprint image  
- Instant prediction output  
- Displays confidence and results clearly  

---

## 📂 Project Structure

## 🛠 Tech Stack
- **Programming**: Python 3.10+, TensorFlow/Keras, scikit-learn, OpenCV  
- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Flask (API + templates)  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: Local demo (Docker/Render compatible)  

---

## 📂 Project structure (actual files)

Non-Invasive-Blood-Group-Prediction-Using-Fingerprint-Analysis/
├── app.py
├── train.py
├── requirements.txt
├── model/
│   └── final_best_efficientnetb0_model_final.keras
├── utils/
│   ├── predict.py
│   └── history.py
├── templates/
│   └── index.html
├── static/
│   └── input_images/

## 🚀 Quickstart

### 1
```bash
cd Non-Invasive-Blood-Group-Prediction-Using-Fingerprint-Analysis
````

### 2. Create virtual environment & install requirements

```bash
python -m venv venv
# activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Model file
The trained model is **already included** in this repository at: `model/final_best_efficientnetb0_model_final.keras`
### 4. Run the demo

```bash
python app.py
```

* Open browser: `http://127.0.0.1:5000`

---

## 📊 Training (updated)

* Requires the SOCOFing dataset (not included due to size).
* Trains EfficientNetB0 with preprocessing, normalization, and dropout.
* Evaluated with accuracy, precision/recall, confusion matrix.

---

## 🎯 Features

* End-to-end ML pipeline: dataset → training → evaluation → deployment.
* Flask demo app with cyberpunk dark-themed UI.
* Easy to run locally (requirements + model file).
* Extendable to MLOps tools (Docker, MLflow, GitHub Actions).

---

## 📈 Results

* **Test Accuracy**: 90.33%
* **Model**: EfficientNetB0 (transfer learning)
* **Classes**: 8 blood groups → `['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']`

---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

* **Dataset**: [SOCOFing – Sokoto Coventry Fingerprint Dataset](https://www.kaggle.com/datasets/ruizgara/socofing)
* **Supervisor**: Dr. Umair Muneer Butt (UMT Sialkot)
* **Team**: Habiba Fiaz, Saad Jamshaid, Zahra Akhtar, Wabil Nadeem Butt

---
