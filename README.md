# 🎙️ Voice Emotion Recognition using Deep Learning

This project focuses on classifying human emotions from speech using deep learning techniques. Using the **RAVDESS** dataset, we explore advanced audio preprocessing, feature engineering, and neural network architectures to accurately detect emotions from voice signals.

---

## 🔍 Project Overview

- Built a deep learning pipeline to classify 8 emotional states: **Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised**
- Used **log-Mel spectrograms** and **derivative features (delta, delta²)** for rich temporal-spectral representation.
- Applied **data augmentation** strategies (noise addition, pitch shifting, time-stretching) to improve generalization.
- Trained and evaluated multiple models with **Conv1D + BiGRU** based architectures.

---

## 📁 Folder Structure

```

voice\_emotion\_recognition/
│
├── data/                  # All dataset-related files
│   └── features/
│       ├── features.npy               # Processed features
│       ├── labels.npy                 # Corresponding labels
│       ├── features\_v4.npy            # Alt version (e.g., different pipeline)
│       └── labels\_v4.npy
│   └── raw/ravdess/                  # Raw RAVDESS audio data
│       ├── Actor\_01/
│       ├── Actor\_02/
│       └── ... Actor\_24/
│
├── models/                # Saved trained models
│   ├── emotional\_model.h5
│   ├── emotional\_model\_v2.keras
│   ├── emotional\_model\_v3.keras
│   └── emotional\_model\_v4.keras
│
├── src/                   # Main scripts
│   ├── model\_architectures/
│   │   ├── model.ipynb
│   │   ├── model\_v2.ipynb
│   │   └── model\_v4.ipynb
│   └── preprocessing/
│       └── preprocess.py
│
├── utils/                 # Utility functions
│   └── augment.py         # Audio augmentation helpers
│
└── requirements.txt       # Python dependencies

````

---

## 🚀 Features & Highlights

- ✅ **Advanced Spectral Features**: Extracted log-Mel spectrograms along with temporal derivatives (Δ, Δ²) to enhance emotional cues in audio.
- ✅ **Robust Augmentation**: Applied real-world inspired augmentations — noise, pitch shift, and time-stretch — to create a resilient model.
- ✅ **Temporal Modeling**: Used time-aware Conv1D layers followed by BiGRU for effective emotion detection from voice sequences.
- ✅ **Modular Codebase**: Clean separation of preprocessing, model building, and utilities for easy experimentation and extension.

---

## 📊 Model Evaluation

Performance evaluated using standard metrics:

- **Precision, Recall, F1-Score**
- **Class-wise Emotion Analysis**
- **Confusion Matrix Visualization**

> Achieved **~59% accuracy** on an 8-class classification task, with strong performance on *Angry*, *Disgust*, *Calm*, and *Surprised* emotions.

---

## 📦 Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
````

---

## 🗃️ Dataset

* [RAVDESS Dataset](https://zenodo.org/record/1188976)

  * 24 professional actors vocalizing two lexically-matched statements in a neutral North American accent
  * Emotions: calm, happy, sad, angry, fearful, surprise, disgust, neutral

---

## 🧠 Future Enhancements

* Integrate attention layers for improved temporal focus.
* Try transformer-based models for long-sequence encoding.
* Extend to multilingual or real-time emotion recognition.

---

## 📌 Credits

* Audio Data: [RAVDESS - Livingstone & Russo, 2018](https://zenodo.org/record/1188976)
* Libraries used: `Librosa`, `TensorFlow/Keras`, `NumPy`, `Matplotlib`

---

## 📜 License

This project is for educational purposes. Please ensure RAVDESS terms are followed for any public sharing.

---

## 🤝 Contributions

Feel free to fork the repo, suggest improvements, or raise issues!
