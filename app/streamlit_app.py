"""
Streamlit web app for diabetes risk prediction demo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import DiabetesRiskPredictor

RESULTS_DIR = PROJECT_ROOT / "results"
SYNTHETIC_CSV = PROJECT_ROOT / "data" / "synthetic_data" / "synthetic_patients.csv"
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_synthetic.py"


@st.cache_resource
def load_predictor():
    return DiabetesRiskPredictor()


def ensure_synthetic_data(n: int = 20):
    if not SYNTHETIC_CSV.exists():
        subprocess.run(
            [sys.executable, str(GENERATE_SCRIPT), "--n", str(n), "--no-predict"],
            cwd=PROJECT_ROOT,
            check=True,
        )


def render_predict_tab(predictor: DiabetesRiskPredictor):
    st.subheader("Single Patient Prediction")
    defaults = DiabetesRiskPredictor.default_input_values(predictor.imputation_stats)

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=int(defaults["Pregnancies"]))
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=float(defaults["Glucose"]))
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=float(defaults["BloodPressure"])
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=float(defaults["SkinThickness"])
        )
    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=float(defaults["Insulin"]))
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=float(defaults["BMI"]))
        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=float(defaults["DiabetesPedigreeFunction"]),
            format="%.3f",
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=int(defaults["Age"]))

    patient = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }

    if st.button("Predict Risk", type="primary"):
        result = predictor.predict(patient).iloc[0]
        st.metric("Risk Label", result["risk_label"])
        st.metric("Diabetes Probability", f"{result['probability_diabetic']:.1%}")
        st.progress(min(max(result["probability_diabetic"], 0.0), 1.0))
        st.caption(f"Model confidence: {result['confidence']:.1%}")


def render_batch_tab(predictor: DiabetesRiskPredictor):
    st.subheader("Batch Demo (Synthetic Patients)")
    n = st.slider("Number of synthetic patients", 5, 50, 20)

    if st.button("Generate & Predict Batch"):
        subprocess.run(
            [sys.executable, str(GENERATE_SCRIPT), "--n", str(n)],
            cwd=PROJECT_ROOT,
            check=True,
        )
        st.success("Synthetic batch generated and scored.")

    if SYNTHETIC_CSV.exists():
        results = predictor.predict_batch(SYNTHETIC_CSV)
        st.dataframe(results, use_container_width=True)
    else:
        st.info("Click 'Generate & Predict Batch' to create synthetic data.")


def render_metrics_tab():
    st.subheader("Model Performance")
    metrics_path = RESULTS_DIR / "model_metrics.csv"
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
    else:
        st.warning("Run 03_model_training.py to generate model_metrics.csv")

    comparison_path = RESULTS_DIR / "model_comparison.png"
    roc_path = RESULTS_DIR / "roc_curves.png"
    fi_path = RESULTS_DIR / "feature_importance.png"

    cols = st.columns(2)
    if comparison_path.exists():
        cols[0].image(str(comparison_path), caption="Model Comparison")
    if roc_path.exists():
        cols[1].image(str(roc_path), caption="ROC Curves")

    if fi_path.exists():
        st.image(str(fi_path), caption="Feature Importance (Best Model)")


def render_about_tab():
    st.markdown(
        """
        ### Early-Stage Diabetes Risk Prediction

        This demo uses the **Pima Indians Diabetes** dataset and a **baseline XGBoost**
        classifier (test AUC ~0.95) trained in the CMPE 188 project pipeline.

        **Pipeline steps:** EDA → preprocessing → model training → optimization → inference.

        **Disclaimer:** This tool is for educational demonstration only and is **not**
        medical advice. Do not use for clinical decisions.
        """
    )
    st.markdown(
        "- [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes)\n"
        "- [Kaggle Mirror](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)"
    )


def main():
    st.set_page_config(
        page_title="Diabetes Risk Predictor",
        page_icon="🩺",
        layout="wide",
    )
    st.title("Early-Stage Diabetes Risk Prediction")
    st.caption("Baseline XGBoost model | CMPE 188 Project")

    predictor = load_predictor()

    tab_predict, tab_batch, tab_metrics, tab_about = st.tabs(
        ["Predict", "Batch Demo", "Model Performance", "About"]
    )

    with tab_predict:
        render_predict_tab(predictor)
    with tab_batch:
        render_batch_tab(predictor)
    with tab_metrics:
        render_metrics_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
