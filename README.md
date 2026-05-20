# Early-Stage Diabetes Risk Prediction System

## Team Members
- Bhargav Jakkaraju

---

## Problem Statement

Diabetes affects over 400 million people worldwide, making it one of the most prevalent chronic diseases. Early detection and intervention are critical to preventing serious complications, including heart disease, kidney failure, and vision loss. However, identifying at-risk individuals before symptoms manifest remains a significant challenge in healthcare.

This project aims to develop a machine learning-based predictive system that can identify patients at risk of diabetes or pre-diabetes based on readily available clinical health indicators. By leveraging classification models trained on historical patient data, the system will enable healthcare professionals to:
- Identify high-risk patients earlier in the disease progression
- Enable timely preventive interventions
- Reduce healthcare costs associated with advanced-stage diabetes management
- Support data-driven clinical decision-making with interpretable model predictions

---

## Dataset and Data Source

**Primary Dataset:** Pima Indians Diabetes Dataset

**Source:**
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Diabetes
- Kaggle Mirror: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Dataset Characteristics:**
- **Total Records:** 768 patient samples
- **Features:** 8 clinical health indicators
- **Target Variable:** Outcome (0 = Non-diabetic, 1 = Diabetic)

**Features:**
1. **Pregnancies** – Number of times the patient has been pregnant
2. **Glucose** – Plasma glucose concentration (2-hour oral glucose tolerance test)
3. **BloodPressure** – Diastolic blood pressure (mm Hg)
4. **SkinThickness** – Triceps skin fold thickness (mm)
5. **Insulin** – 2-hour serum insulin (mu U/ml)
6. **BMI** – Body mass index (weight in kg / height in m²)
7. **DiabetesPedigreeFunction** – Diabetes likelihood based on family history
8. **Age** – Age of the patient in years

---

## Planned Model and System Approach

### 1. Data Preprocessing Pipeline
- Handle missing and zero values appropriately
- Feature scaling and normalization
- Address class imbalance in the target variable
- Data exploration and statistical analysis

### 2. Machine Learning Models
- **Logistic Regression** – Baseline linear model
- **Random Forest** – Ensemble method with feature importance
- **XGBoost** – Gradient boosting (best baseline model, used for inference)
- **Support Vector Machine (SVM)** – Non-linear classifier

### 3. Model Evaluation and Comparison
- Accuracy, Precision, Recall, F1-Score, AUC-ROC

### 4. Real-Time Inference Pipeline
- `src/inference.py` loads baseline XGBoost, scaler, and imputation stats
- Batch scoring via synthetic patient CSV

### 5. Interactive Web Application (Streamlit)
- Single-patient risk prediction
- Synthetic batch demo
- Model metrics and plots from `results/`

---

## Current Implementation Progress

**Completed:**
- Exploratory data analysis (`01eda.py`)
- Data preprocessing and artifact export (`02_preprocessing.py`)
- Model training and comparison (`03_model_training.py`)
- XGBoost hyperparameter optimization (`04_optimize_best_model.py`)
- Shared preprocessing utilities (`src/preprocessing_utils.py`)
- Inference module (`src/inference.py`) — **baseline XGBoost**
- Synthetic data generator (`scripts/generate_synthetic.py`)
- Streamlit demo app (`app/streamlit_app.py`)

**Note:** Optimized XGBoost is retained for comparison; inference uses baseline XGBoost because it achieves higher holdout AUC.

---

## Project Structure

```
cmpe188-proj/
├── README.md
├── requirements.txt
├── run_pipeline.sh
├── 01eda.py
├── 02_preprocessing.py
├── 03_model_training.py
├── 04_optimize_best_model.py
├── data/
│   ├── pima-indians-diabetes.csv
│   └── synthetic_data/
├── src/
│   ├── preprocessing_utils.py
│   └── inference.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── generate_synthetic.py
│   └── generate_demo_summary.py
└── results/
    ├── model_metrics.csv
    ├── xgboost.pkl
    ├── scaler.pkl
    ├── imputation_stats.pkl
    └── demo_metrics_summary.md
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

```bash
git clone https://github.com/BhargavJakkaraju/cmpe188-proj.git
cd cmpe188-proj

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run full pipeline

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Or step by step:

```bash
python 01eda.py
python 02_preprocessing.py
python 03_model_training.py
python 04_optimize_best_model.py
python scripts/generate_synthetic.py --n 20
python scripts/generate_demo_summary.py
```

### Inference CLI

```bash
# Single patient (JSON with 8 base features)
python -m src.inference --input patient.json

# Batch CSV
python -m src.inference --csv data/synthetic_data/synthetic_patients.csv
```

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## References

**Dataset:**
- UCI Machine Learning Repository – Pima Indians Diabetes Dataset: https://archive.ics.uci.edu/ml/datasets/Diabetes
- Kaggle Mirror: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Libraries:**
- Scikit-learn, XGBoost, Streamlit, Pandas, NumPy, Matplotlib, Seaborn

---

## License

MIT License. See repository for details.

## Contact

Bhargav Jakkaraju — https://github.com/BhargavJakkaraju
