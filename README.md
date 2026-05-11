<div align="center">


# LookALike.AI

### AI-Powered Facial Analysis Platform

**Face Detection · Celebrity Recognition · Emotion Analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.x-092E20?style=flat-square&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Overview

**LookALike.AI** is an intelligent, web-based facial analysis application that brings together Computer Vision and Machine Learning into a seamless experience. Upload a photo or use your webcam to instantly receive:

- 🎯 **Face Detection** — Precise facial localization using deep learning
- 🌟 **Celebrity Recognition** — Identity matching with confidence scores
- 😄 **Emotion Analysis** — Real-time classification of facial expressions

Built with Django, OpenCV, and Scikit-learn, LookALike.AI replaces inefficient manual observation with a scalable, automated pipeline ready for real-world deployment.

---

## Demo

> Upload an image or capture directly via webcam — results are generated in seconds.

| Feature | Description |
|---|---|
| 📸 Webcam Capture | Real-time photo taken from the browser |
| 🖼️ Image Upload | Supports standard image formats (JPG, PNG) |
| 📊 Confidence Score | Probability shown for each prediction |

---

## Features

- **Real-Time Face Detection** using a pretrained SSD ResNet DNN (OpenCV)
- **Celebrity Identity Recognition** using facial embeddings + ML classifiers
- **Emotion Classification** — Happy, Sad, Angry, Neutral, Surprise
- **Webcam Integration** built into the browser interface
- **Confidence Scoring** for both identity and emotion predictions
- **Clean Web Interface** powered by Bootstrap

---

## System Architecture

The application follows a structured multi-stage ML pipeline:

```
Image Input
    │
    ▼
Face Detection (SSD ResNet DNN)
    │
    ▼
Feature Extraction (OpenFace Embeddings)
    │
    ├──────────────────────┐
    ▼                      ▼
Identity Recognition   Emotion Recognition
(SVM / LR / RF)        (ML Classifier)
    │                      │
    └──────────┬───────────┘
               ▼
       Result Visualization
       (Bounding Box + Labels + Confidence)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Django (Python) |
| **Computer Vision** | OpenCV |
| **Machine Learning** | Scikit-learn |
| **Numerical Computing** | NumPy, SciPy |
| **Frontend** | HTML, CSS, Bootstrap, JavaScript |
| **Database** | SQLite3 |
| **Dev Tools** | VS Code, Jupyter Notebook |

---

## ML Models

| Model File | Purpose |
|---|---|
| `celebrity_identity_model.pkl` | Celebrity face recognition |
| `emotion_recognition_model.pkl` | Emotion classification |
| `dlib_face_recognition_resnet_model_v1.dat` | Facial feature extraction (dlib) |
| `shape_predictor_68_face_landmarks.dat` | 68-point facial landmark detection |
| `res10_300x300_ssd_iter_140000.caffemodel` | Face detection (SSD ResNet) |
| `deploy.prototxt.txt` | DNN architecture configuration |
| `openface.nn4.small2.v1.t7` | OpenFace embedding model |

> **Note:** Large model files (`.dat`, `.caffemodel`, `.t7`) are not included in the repository. Download links are listed in [`static/models/urls.txt`](static/models/urls.txt).

---

## Project Structure

```
vision-craft-main/
├── facerecognition/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── app/                      # Core Django application
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   ├── machinelearning.py    # ML inference logic
│   └── migrations/
├── static/
│   ├── models/               # Pretrained model files
│   └── logo.png
├── templates/
│   └── index.html
├── training_data/            # Dataset used during training
├── train_model.py            # Model training script
├── manage.py
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python **3.8+**
- pip

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/LookALike.AI.git
cd LookALike.AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Files

Download the required pretrained models and place them in `static/models/`.
Reference links are provided in [`static/models/urls.txt`](static/models/urls.txt).

| File | Size (approx.) |
|---|---|
| `dlib_face_recognition_resnet_model_v1.dat` | ~22 MB |
| `shape_predictor_68_face_landmarks.dat` | ~99 MB |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~10 MB |
| `openface.nn4.small2.v1.t7` | ~31 MB |

### 5. Apply Migrations

```bash
python manage.py migrate
```

### 6. Run the Development Server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` in your browser.

---

## Training Your Own Models

To retrain models on custom data:

```bash
python train_model.py
```

Jupyter notebooks for each pipeline stage are available in the project root:

| Notebook | Description |
|---|---|
| `1_data_preprocessing.ipynb` | Face data preprocessing |
| `2_train_machine_learning_model.ipynb` | Identity model training |
| `3_data_preprocessing-emotion.ipynb` | Emotion data preprocessing |
| `4_train_machine_learning_model-emotion.ipynb` | Emotion model training |
| `5_pipeline.ipynb` | End-to-end pipeline testing |

---

## ML Pipeline Details

### Face Detection
A pretrained **SSD ResNet DNN** (via OpenCV) localizes faces in uploaded or captured images with high accuracy.

### Feature Extraction
**OpenFace (nn4.small2.v1.t7)** and **dlib ResNet** models generate 128-dimensional facial embeddings as numerical representations of each detected face.

### Identity Recognition
Three classifiers were evaluated:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

The best-performing model is saved as `celebrity_identity_model.pkl`.

### Emotion Recognition
A dedicated classifier trained on labeled emotion data (`emotion_recognition_model.pkl`) categorizes expressions into: **Happy**, **Sad**, **Angry**, **Neutral**, **Surprise**.

---

## Requirements

Key dependencies from `requirements.txt`:

```
django
opencv-python
dlib
numpy
scipy
scikit-learn
Pillow
```

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [dlib](http://dlib.net/) — Facial landmark and feature extraction
- [OpenCV](https://opencv.org/) — Computer vision pipeline
- [OpenFace](https://cmusatyalab.github.io/openface/) — Face embedding model
- [Scikit-learn](https://scikit-learn.org/) — ML classifiers
- [Django](https://www.djangoproject.com/) — Web framework

---

<div align="center">

Made with ❤️ · LookALike.AI

</div>
