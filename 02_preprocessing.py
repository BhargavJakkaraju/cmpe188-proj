"""
Data Preprocessing and Cleaning
Early-Stage Diabetes Risk Prediction System

Handles data cleaning, zero/missing-value imputation, feature engineering,
train/test split, feature scaling, class imbalance handling, and artifact export.
"""

import ssl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
LOCAL_DATA_PATH = "data/pima-indians-diabetes.csv"

COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

IMPUTE_FEATURES = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def load_data(url=DATA_URL, local_path=LOCAL_DATA_PATH):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        if Path(local_path).exists():
            print(f"Loading data from local file: {local_path}")
            df = pd.read_csv(local_path, names=COLUMN_NAMES)
        else:
            print(f"Downloading data from {url}...")
            ssl._create_default_https_context = ssl._create_unverified_context
            df = pd.read_csv(url, names=COLUMN_NAMES)
            df.to_csv(local_path, index=False)
            print(f"Data saved to {local_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    return df


def clean_dataset(df):
    """Normalize schema artifacts and ensure numeric dtypes."""
    df_clean = df.copy()

    if str(df_clean.iloc[0]['Pregnancies']) == 'Pregnancies':
        df_clean = df_clean.iloc[1:].reset_index(drop=True)

    for col in COLUMN_NAMES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    before_drop = len(df_clean)
    df_clean = df_clean.dropna().drop_duplicates().reset_index(drop=True)
    dropped = before_drop - len(df_clean)

    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)
    print(f"  Removed invalid/duplicate rows: {dropped}")
    print(f"  Remaining rows: {len(df_clean)}")

    return df_clean


def impute_zeros_stratified(df):
    """Replace zeros in medical features with per-class median values."""
    df_clean = df.copy()
    df_clean[IMPUTE_FEATURES] = df_clean[IMPUTE_FEATURES].astype(float)

    print("\n" + "="*70)
    print("ZERO-VALUE IMPUTATION (Stratified Median)")
    print("="*70)

    print("\nZero counts BEFORE imputation:")
    for feature in IMPUTE_FEATURES:
        count = (df_clean[feature] == 0).sum()
        pct = count / len(df_clean) * 100
        print(f"  {feature}: {count} zeros ({pct:.1f}%)")

    for feature in IMPUTE_FEATURES:
        for outcome_class in [0, 1]:
            mask_zero = df_clean[feature] == 0
            mask_class = df_clean['Outcome'] == outcome_class
            median_val = df_clean.loc[~mask_zero & mask_class, feature].median()
            df_clean.loc[mask_zero & mask_class, feature] = median_val

    print("\nZero counts AFTER imputation:")
    for feature in IMPUTE_FEATURES:
        count = (df_clean[feature] == 0).sum()
        print(f"  {feature}: {count} zeros")

    return df_clean


def engineer_features(df):
    """Create clinically-inspired interaction and ratio features."""
    df_feat = df.copy()
    eps = 1e-6

    df_feat['Glucose_BMI_Interaction'] = df_feat['Glucose'] * df_feat['BMI']
    df_feat['Age_BMI_Interaction'] = df_feat['Age'] * df_feat['BMI']
    df_feat['Insulin_Glucose_Ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + eps)
    df_feat['RiskScore_Glucose_Age_BMI'] = (
        0.5 * df_feat['Glucose'] + 0.3 * df_feat['BMI'] + 0.2 * df_feat['Age']
    )

    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    print("  Added features:")
    print("   - Glucose_BMI_Interaction")
    print("   - Age_BMI_Interaction")
    print("   - Insulin_Glucose_Ratio")
    print("   - RiskScore_Glucose_Age_BMI")

    return df_feat


def split_data(df, test_size=0.2, random_state=42):
    """Stratified 80/20 train/test split."""
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)
    print(f"  Training set:  {X_train.shape[0]} samples")
    print(f"  Test set:      {X_test.shape[0]} samples")
    print(f"  Train class distribution: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"  Test class distribution:  {y_test.value_counts(normalize=True).to_dict()}")

    return X_train, X_test, y_train, y_test


def oversample_minority_class(X_train, y_train, random_state=42):
    """Randomly oversample minority class in training split only."""
    train_df = X_train.copy()
    train_df['Outcome'] = y_train.values

    majority = train_df[train_df['Outcome'] == 0]
    minority = train_df[train_df['Outcome'] == 1]

    if len(minority) == 0 or len(majority) == 0:
        print("\nClass imbalance handling skipped: one class missing.")
        return X_train, y_train

    if len(minority) < len(majority):
        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=random_state
        )
        balanced = pd.concat([majority, minority_upsampled], axis=0)
    else:
        majority_upsampled = resample(
            majority,
            replace=True,
            n_samples=len(minority),
            random_state=random_state
        )
        balanced = pd.concat([majority_upsampled, minority], axis=0)

    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    X_train_bal = balanced.drop(columns=['Outcome'])
    y_train_bal = balanced['Outcome']

    print("\n" + "="*70)
    print("CLASS IMBALANCE HANDLING")
    print("="*70)
    print(f"  Original train distribution: {y_train.value_counts().to_dict()}")
    print(f"  Balanced train distribution: {y_train_bal.value_counts().to_dict()}")
    print("  Method: Random over-sampling on training set")

    return X_train_bal, y_train_bal


def scale_features(X_train, X_test, output_dir="results"):
    """Fit StandardScaler on training data, transform both splits, save scaler."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    scaler_path = f"{output_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print("\n" + "="*70)
    print("FEATURE SCALING")
    print("="*70)
    print(f"  StandardScaler fitted on training data and saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def compute_class_weights(y_train, output_dir="results"):
    """Compute balanced class weights and save for model training."""
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    weights_path = f"{output_dir}/class_weights.pkl"
    joblib.dump(class_weight_dict, weights_path)

    print("\n" + "="*70)
    print("CLASS WEIGHTS")
    print("="*70)
    print(f"  Class 0 (Non-Diabetic): {class_weight_dict[0]:.4f}")
    print(f"  Class 1 (Diabetic):     {class_weight_dict[1]:.4f}")
    print(f"  Saved to {weights_path}")

    return class_weight_dict


def save_datasets(X_train, X_test, y_train, y_test, data_dir="data"):
    """Save scaled train/test splits to CSV."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(f"{data_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{data_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{data_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{data_dir}/y_test.csv", index=False)

    print("\n" + "="*70)
    print("SAVED DATASETS")
    print("="*70)
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        print(f"  {data_dir}/{name}.csv")


def save_balanced_training_set(X_train_bal, y_train_bal, data_dir="data"):
    """Save balanced training split for models that do not use class weights."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    X_train_bal.to_csv(f"{data_dir}/X_train_balanced.csv", index=False)
    y_train_bal.to_csv(f"{data_dir}/y_train_balanced.csv", index=False)
    print(f"  {data_dir}/X_train_balanced.csv")
    print(f"  {data_dir}/y_train_balanced.csv")


def visualize_preprocessing(df_raw, df_clean, output_dir="results"):
    """Side-by-side histograms of imputed features before and after cleaning."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Feature Distributions: Before vs. After Zero Imputation", fontsize=14)

    for col_idx, feature in enumerate(IMPUTE_FEATURES):
        ax_before = axes[0, col_idx]
        ax_after = axes[1, col_idx]

        ax_before.hist(df_raw[feature], bins=30, color='salmon', edgecolor='black', alpha=0.8)
        ax_before.set_title(f"{feature}\n(Before)", fontsize=10)
        ax_before.set_xlabel("Value")
        ax_before.set_ylabel("Frequency" if col_idx == 0 else "")

        ax_after.hist(df_clean[feature], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
        ax_after.set_title(f"{feature}\n(After)", fontsize=10)
        ax_after.set_xlabel("Value")
        ax_after.set_ylabel("Frequency" if col_idx == 0 else "")

    plt.tight_layout()
    out_path = f"{output_dir}/05_preprocessing_before_after.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n  Visualization saved to {out_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("EARLY-STAGE DIABETES RISK PREDICTION")
    print("Data Preprocessing and Cleaning")
    print("="*70)

    df_raw = load_data()
    if df_raw is None:
        print("Failed to load data. Exiting.")
        return

    df = clean_dataset(df_raw)

    df_clean = impute_zeros_stratified(df)
    df_featured = engineer_features(df_clean)

    print("\n" + "="*70)
    print("GENERATING BEFORE/AFTER VISUALIZATIONS")
    print("="*70)
    visualize_preprocessing(df, df_clean)

    X_train, X_test, y_train, y_test = split_data(df_featured)

    X_train_bal, y_train_bal = oversample_minority_class(X_train, y_train)

    X_train_bal_scaled, X_test_scaled, _ = scale_features(X_train_bal, X_test)

    compute_class_weights(y_train_bal)

    save_datasets(X_train_bal_scaled, X_test_scaled, y_train_bal, y_test)
    save_balanced_training_set(X_train_bal, y_train_bal)

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print("Next steps:")
    print("1. Review results/05_preprocessing_before_after.png")
    print("2. Train models using data/X_train.csv + data/y_train.csv (balanced + scaled)")
    print("3. Pass results/class_weights.pkl to model constructors")
    print("4. Use results/scaler.pkl for inference pipeline")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
