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

**Additional Data:**
- Synthetic patient records will be generated for real-time pipeline evaluation, simulating live clinical data intake streams using statistical distributions from the training data

---

## Planned Model and System Approach

### 1. **Data Preprocessing Pipeline**
- Handle missing and zero values appropriately
- Feature scaling and normalization
- Address class imbalance in the target variable
- Data exploration and statistical analysis

### 2. **Machine Learning Models**
The project will train and evaluate four classification models:
- **Logistic Regression** – Baseline linear model for interpretability
- **Random Forest** – Ensemble method for robustness and feature importance
- **XGBoost** – Gradient boosting for high predictive performance
- **Support Vector Machine (SVM)** – Non-linear classifier for decision boundary separation

### 3. **Model Evaluation and Comparison**
All models will be evaluated using:
- **Accuracy** – Overall correctness of predictions
- **Precision** – True positive rate among predicted positives
- **Recall** – True positive rate among actual positives
- **F1-Score** – Harmonic mean of precision and recall
- **AUC-ROC Curve** – Trade-off between true positive and false positive rates

### 4. **Real-Time Inference Pipeline**
- Integration of the best-performing model into a live prediction system
- Batch processing of simulated patient records with confidence scores
- Feature importance analysis for model interpretability

### 5. **Interactive Web Application (Streamlit)**
- User-friendly interface for clinicians to input patient vitals
- Real-time diabetes risk predictions with confidence scores
- Feature importance visualization showing which factors most influence predictions
- Visual comparison of model performance metrics

### 6. **Technology Stack**
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Version Control:** Git/GitHub

---

## Current Implementation Progress

**In Progress:**
- 🔄 Environment setup and dependency installation
- 🔄 Data loading and exploratory data analysis (EDA)

**Next Steps:**
- [ ] Data preprocessing and cleaning
- [ ] Handling missing values and class imbalance
- [ ] Feature scaling and engineering
- [ ] Model training (Logistic Regression, Random Forest, XGBoost, SVM)
- [ ] Model evaluation and comparison
- [ ] Best model selection and optimization
- [ ] Real-time inference pipeline development
- [ ] Streamlit web app development
- [ ] Final demo preparation and testing

---

## Project Structure

```
diabetes-prediction-ml/
├── README.md
├── requirements.txt
├── data/
│   ├── pima-indians-diabetes.csv
│   └── synthetic_data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── inference.py
├── app/
│   └── streamlit_app.py
├── results/
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── roc_curves.png
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd diabetes-prediction-ml

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run Streamlit web app
streamlit run app/streamlit_app.py
```

---

## References

**Dataset:**
- UCI Machine Learning Repository – Pima Indians Diabetes Dataset: https://archive.ics.uci.edu/ml/datasets/Diabetes
- Kaggle Mirror: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Open Source Libraries:**
- Scikit-learn (ML models): https://github.com/scikit-learn/scikit-learn
- XGBoost (Gradient boosting): https://github.com/dmlc/xgboost
- Streamlit (Web framework): https://github.com/streamlit/streamlit
- Pandas (Data processing): https://github.com/pandas-dev/pandas
- NumPy (Numerical computing): https://github.com/numpy/numpy
- Matplotlib (Visualization): https://github.com/matplotlib/matplotlib
- Seaborn (Statistical visualization): https://github.com/mwaskom/seaborn

---

## License
[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contact
[Add contact information or team email]
