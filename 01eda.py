"""
Data Loading and Exploratory Data Analysis (EDA)
Early-Stage Diabetes Risk Prediction System

This script loads the Pima Indians Diabetes Dataset and performs initial
exploratory data analysis to understand data characteristics, distributions,
and potential data quality issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/diabetes.data.csv"
LOCAL_DATA_PATH = "data/pima-indians-diabetes.csv"

COLUMN_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]

def load_data(url=DATA_URL, local_path=LOCAL_DATA_PATH):
    """
    Load Pima Indians Diabetes Dataset from UCI or local file.
    
    Args:
        url (str): URL to download dataset from
        local_path (str): Local path to save/load dataset
    
    Returns:
        pd.DataFrame: Loaded dataset with column names
    """
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if Path(local_path).exists():
            print(f"Loading data from local file: {local_path}")
            df = pd.read_csv(local_path, names=COLUMN_NAMES)
        else:
            print(f"Downloading data from {url}...")
            df = pd.read_csv(url, names=COLUMN_NAMES)
            df.to_csv(local_path, index=False)
            print(f"Data saved to {local_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    return df

def basic_statistics(df):
    """
    Display basic statistics about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
    """
    print("\n" + "="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    print(f"Dataset shape: {df.shape}")
    print(f"Total records: {df.shape[0]}")
    print(f"Total features: {df.shape[1]}")
    
    print("\n" + "="*70)
    print("DATA TYPES AND MISSING VALUES")
    print("="*70)
    print(df.info())
    
    print("\n" + "="*70)
    print("MISSING VALUES SUMMARY")
    print("="*70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print(missing[missing > 0])
    
    print("\n" + "="*70)
    print("DESCRIPTIVE STATISTICS")
    print("="*70)
    print(df.describe())

def class_distribution(df):
    """
    Analyze target variable distribution.
    
    Args:
        df (pd.DataFrame): Input dataset
    """
    print("\n" + "="*70)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*70)
    
    outcome_counts = df['Outcome'].value_counts()
    outcome_pcts = df['Outcome'].value_counts(normalize=True) * 100
    
    print(f"\nOutcome Distribution:")
    print(f"Non-Diabetic (0): {outcome_counts[0]} records ({outcome_pcts[0]:.2f}%)")
    print(f"Diabetic (1): {outcome_counts[1]} records ({outcome_pcts[1]:.2f}%)")
    print(f"\nClass Imbalance Ratio: {outcome_counts[0] / outcome_counts[1]:.2f}:1")

def zero_value_analysis(df):
    """
    Analyze zero values in features (common issue in medical data).
    
    Args:
        df (pd.DataFrame): Input dataset
    """
    print("\n" + "="*70)
    print("ZERO VALUE ANALYSIS")
    print("="*70)
    
    features_no_zeros = [
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI'
    ]
    
    print("\nZero counts in features (likely missing values):")
    for feature in features_no_zeros:
        zero_count = (df[feature] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        if zero_count > 0:
            print(f"{feature}: {zero_count} zeros ({zero_pct:.2f}%)")

def correlation_analysis(df):
    """
    Analyze feature correlations with target variable.
    
    Args:
        df (pd.DataFrame): Input dataset
    """
    print("\n" + "="*70)
    print("FEATURE CORRELATION WITH TARGET")
    print("="*70)
    
    correlations = df.corr()['Outcome'].sort_values(ascending=False)
    print("\nCorrelation with Outcome (Diabetes):")
    print(correlations)

def create_visualizations(df, output_dir="results"):
    """
    Create and save exploratory visualizations.
    
    Args:
        df (pd.DataFrame): Input dataset
        output_dir (str): Directory to save visualizations
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Target Distribution
    plt.figure(figsize=(10, 5))
    outcome_counts = df['Outcome'].value_counts()
    plt.subplot(1, 2, 1)
    outcome_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Target Variable Distribution')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    outcome_counts.plot(kind='pie', autopct='%1.1f%%', labels=['Non-Diabetic', 'Diabetic'])
    plt.title('Class Distribution (%)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_target_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/01_target_distribution.png")
    plt.close()
    
    plt.figure(figsize=(14, 10))
    features = [col for col in df.columns if col != 'Outcome']
    
    for idx, feature in enumerate(features, 1):
        plt.subplot(3, 3, idx)
        df[feature].hist(bins=30, edgecolor='black')
        plt.title(f'{feature} Distribution')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/02_feature_distributions.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/03_correlation_heatmap.png")
    plt.close()
    
    plt.figure(figsize=(14, 10))
    for idx, feature in enumerate(features, 1):
        plt.subplot(3, 3, idx)
        df.boxplot(column=feature, by='Outcome')
        plt.title(f'{feature} by Outcome')
        plt.suptitle('') 
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_boxplots_by_outcome.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/04_boxplots_by_outcome.png")
    plt.close()

def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("EARLY-STAGE DIABETES RISK PREDICTION")
    print("Exploratory Data Analysis")
    print("="*70)
    

    df = load_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
 
    basic_statistics(df)
    class_distribution(df)
    zero_value_analysis(df)
    correlation_analysis(df)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    create_visualizations(df)
    
    print("\n" + "="*70)
    print("EDA COMPLETE!")
    print("="*70)
    print("Next steps:")
    print("1. Review generated visualizations")
    print("2. Handle missing values (zeros in medical features)")
    print("3. Apply feature scaling")
    print("4. Address class imbalance")
    print("5. Begin model training")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()