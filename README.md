## SignaMind: An Adaptive ASL Learning Intelligence System

SignaMind is a behavior-aware, data-driven ASL learning application built with Streamlit, computer vision, and machine learning. It is designed to support American Sign Language fingerspelling practice through real-time sign recognition, confidence-aware feedback, curriculum-guided learning, and Business Intelligence inspired performance analytics.

The system goes beyond static prediction by tracking how users practice, identifying failure patterns, and providing adaptive feedback to improve learning outcomes over time.

### Project Overview

This project combines:
- **Computer Vision** for image preprocessing and hand-sign preparation
- **Deep Learning** for ASL sign recognition using a MobileNetV2-based model
- **Business Intelligence concepts** for confidence tracking, performance monitoring, and user-behavior insights
- **Interactive Learning Design** through practice modes, streaks, XP, badges, and targeted correction

### Key Features

```
- Real-time ASL fingerspelling recognition
- Confidence-based prediction feedback
- Image quality checks for blur, brightness, and contrast
- Target Practice and Free Detection modes
- Hard Gesture Practice for difficult signs
- Confusion Pair Practice for frequently mixed signs
- Word Practice for sequential fingerspelling
- Typewriter Mode for sign-to-text interaction
- XP, streaks, levels, and badges
- Anti-frustration logic with corrective pro-tips
- Session analytics and learning feedback
```

## Practice Modes
### 1. Target Practice
Users practice one target sign at a time from the full class set.

### 2. Free Detection
The model predicts any sign shown, without a predefined target.

### 3. Hard Gesture Practice
Focuses on more difficult or commonly misclassified signs.

### 4. Confusion Pair Practice
Allows users to practice between similar signs that are often confused.

### 5. Word Practice
Users build complete words letter by letter and receive progress-based feedback.

### 6. Typewriter Mode
Users analyze signs one at a time and confirm them into a text buffer, with optional spelling suggestions.


### Machine Learning and Inference

The app uses a **MobileNetV2 model** for ASL sign classification across **36 classes (A-Z and 0-9)**. Prediction logic includes:
- preprocessing with grayscale conversion and CLAHE
- largest foreground crop detection
- square padding and resizing
- test-time augmentation with brightness and rotation variants
- averaged probability output across augmented samples

### Business Intelligence Alignment

This project aligns with Business Intelligence programming goals by:
- using Python for data processing, transformation, and automation
- integrating machine learning into an interpretable learning system
- tracking user behavior through session metrics and persistent progress logs
- presenting dashboard-like feedback for confidence, accuracy, streaks, and progress
- generating actionable insights rather than only raw predictions

### Project Structure

```text
SignaMind/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   │
│   ├── assets/
│   │   ├── letters/
│   │   │   ├── A.jpg
│   │   │   ├── B.jpg
│   │   │   └── ... Z.jpg
│   │   │
│   │   └── numbers/
│   │       ├── 0.png
│   │       ├── 1.png
│   │       └── ... 9.png
│   │
│   └── data/
│       └── progress.json
│
├── artifacts/
│   ├── class_mapping.json
│   ├── hand_landmarker.task
│   │
│   ├── curriculum/
│   │   ├── curriculum.json
│   │   ├── easy_letters.csv
│   │   ├── hard_letters.csv
│   │   ├── confusion_pairs.csv
│   │   └── word_curriculum.csv
│   │
│   ├── evaluation/
│   │   ├── confusion_matrix_baseline.npy
│   │   ├── confusion_matrix_model2_augmented.npy
│   │   ├── confusion_matrix_model3_mobilenetv2.npy
│   │   ├── model_comparison_summary.csv
│   │   ├── per_class_metrics_baseline.csv
│   │   ├── per_class_metrics_model2_augmented.csv
│   │   ├── per_class_metrics_model3_mobilenetv2.csv
│   │   ├── top_confusions_baseline.csv
│   │   ├── top_confusions_model2_augmented.csv
│   │   └── top_confusions_model3_mobilenetv2.csv
│   │
│   ├── asl_landmarks.csv
│   ├── asl_landmarks_failed.csv
│   │
│   ├── model_card_20260205_225054.json
│   ├── model_card_model2_20260206_010227.json
│   ├── model_card_model3_20260206_020710.json
│   │
│   ├── train_history_20260205_225054.csv
│   ├── train_history_model2_20260206_010227.csv
│   ├── train_history_model3_20260206_020710.csv
│
├── models/
│   ├── baseline_cnn.keras
│   ├── model2_augmented.keras
│   └── model3_mobilenetv2.keras
│
├── data/
│   ├── word_curriculum.csv
│   │
│   ├── logs/
│   │   └── predictions_log.csv
│   │
│   └── processed/
│       ├── train_split.csv
│       ├── val_split.csv
│       └── test_split.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model2_augmented_cnn.ipynb
│   ├── 05_model3_transfer_learning.ipynb
│   └── 06_evaluation_metrics.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt

Installation
```
1. Clone the repository

2. Create and activate a virtual environment
- Windows
python -m venv venv
venv\Scripts\activate

- macOS / Linux
python -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
```bash
streamlit run app3.py
```

### Required Files
Before running the project, make sure these files are present:

- artifacts/model3_mobilenetv2.keras
- artifacts/class_mapping.json
- curriculum CSV and JSON files under artifacts/curriculum/
- evaluation files under artifacts/evaluation/

Notes: 
> The spell-check suggestion feature uses pyspellchecker as an optional dependency.
> 
> User progress is stored locally in app/logs/user_progress.json.
> 
> Word attempts are logged to app/logs/word_attempts_log.csv.
> 
> If reference images are available in the assets folders, they are used to guide practice feedback.


Known Improvement Opportunities: 
*Add live webcam streaming instead of single-image capture
Add downloadable analytics reports
Expand from fingerspelling to word-level or sentence-level signing
Improve robustness under varied backgrounds and lighting
Deploy publicly on Streamlit Cloud*

```
Author
Shubhangi Singh
``` 
