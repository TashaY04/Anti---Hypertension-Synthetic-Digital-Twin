"""
Configuration file for Causal Prototype Network with Adversarial Patient Augmentation
Digital Twin - Hypertension Drug Recommendation System
"""

import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATASET URLS ====================
# These are example datasets - you'll need to update with actual Kaggle/PhysioNet links
DATASETS = {
    "kaggle_hypertension": {
        "url": "https://example.com/hypertension.csv",  # Replace with actual Kaggle link
        "filename": "hypertension_dataset.csv",
        "type": "csv"
    },
    "kaggle_pkpd": {
        "url": "https://example.com/pkpd.csv",  # Replace with actual Kaggle link
        "filename": "pkpd_dataset.csv",
        "type": "csv"
    },
    # PhysioNet and Synthea datasets - update these URLs
    "physionet": {
        "url": "https://example.com/physionet.csv",
        "filename": "physionet_dataset.csv",
        "type": "csv"
    },
    "synthea": {
        "url": "https://example.com/synthea.csv",
        "filename": "synthea_dataset.csv",
        "type": "csv"
    }
}

# ==================== FEATURE DEFINITIONS ====================
NUMERICAL_FEATURES = [
    'age',
    'systolic_bp',
    'diastolic_bp',
    'bmi',
    'heart_rate',
    'duration_years'
]

CATEGORICAL_FEATURES = [
    'gender',
    'diabetes',
    'kidney_disease',
    'smoker',
    'high_stress',
    'sedentary_lifestyle',
    'high_cholesterol',
    'hormonal_imbalance',
    'pregnancy_postpartum'
]

TARGET_COLUMN = 'recommended_drug'

# ==================== COMMON HYPERTENSION DRUGS ====================
DRUG_CLASSES = [
    'ACE_Inhibitor',      # e.g., Lisinopril, Enalapril
    'ARB',                # e.g., Losartan, Valsartan
    'Calcium_Channel_Blocker',  # e.g., Amlodipine, Nifedipine
    'Diuretic',           # e.g., Hydrochlorothiazide, Furosemide
    'Beta_Blocker',       # e.g., Metoprolol, Atenolol
    'Alpha_Blocker',      # e.g., Doxazosin, Prazosin
    'Vasodilator',        # e.g., Hydralazine
    'Combination_Therapy' # Multiple drugs combined
]

# Drug contraindications (safety rules)
CONTRAINDICATIONS = {
    'ACE_Inhibitor': ['pregnancy_postpartum', 'kidney_disease_severe'],
    'ARB': ['pregnancy_postpartum', 'kidney_disease_severe'],
    'Beta_Blocker': ['asthma', 'heart_block'],
    'Diuretic': ['kidney_disease_severe', 'gout'],
    'Calcium_Channel_Blocker': ['heart_failure_severe']
}

# ==================== MODEL HYPERPARAMETERS ====================

# CTGAN (Synthetic Data Generation)
CTGAN_CONFIG = {
    'epochs': 300,
    'batch_size': 500,
    'generator_dim': (256, 256),
    'discriminator_dim': (256, 256),
    'verbose': True
}

# Number of synthetic patients to generate per real patient
SYNTHETIC_RATIO = 1.5  # Generate 1.5x synthetic data

# K-Prototypes Clustering
PROTOTYPE_CONFIG = {
    'n_clusters': 8,  # 8 patient archetypes
    'init': 'Cao',
    'n_init': 10,
    'max_iter': 100,
    'verbose': 1
}

# XGBoost
XGBOOST_CONFIG = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'random_state': 42
}

# Random Forest (Ensemble backup)
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# ==================== ENSEMBLE WEIGHTS ====================
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.40,
    'prototype': 0.35,
    'causal': 0.25
}

# ==================== TRAINING PARAMETERS ====================
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
N_FOLDS = 5  # For cross-validation

# ==================== EVALUATION METRICS ====================
METRICS_TO_TRACK = [
    'accuracy',
    'precision_macro',
    'recall_macro',
    'f1_macro',
    'roc_auc_ovr',  # One-vs-Rest ROC-AUC
    'top_3_accuracy'
]

MINIMUM_REQUIRED_METRICS = {
    'top_3_accuracy': 0.90,
    'precision_macro': 0.90,
    'recall_macro': 0.90,
    'f1_macro': 0.90,
    'roc_auc_ovr': 0.90
}

# ==================== UI INTEGRATION ====================
TOP_K_RECOMMENDATIONS = 3
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to recommend

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(RESULTS_DIR, "training.log")

# ==================== FEATURE ENGINEERING ====================
INTERACTION_FEATURES = [
    ('age', 'systolic_bp'),
    ('diabetes', 'kidney_disease'),
    ('age', 'duration_years'),
    ('bmi', 'sedentary_lifestyle')
]

# ==================== CAUSAL INFERENCE ====================
CAUSAL_CONFIG = {
    'treatment_column': 'recommended_drug',
    'outcome_column': 'bp_reduction',
    'confounders': [
        'age', 'gender', 'bmi', 'diabetes', 
        'kidney_disease', 'duration_years'
    ],
    'method': 'backdoor.propensity_score_weighting'
}

# ==================== EXPLAINABILITY ====================
SHAP_CONFIG = {
    'max_samples': 1000,  # For SHAP value computation
    'plot_top_features': 10
}

print(f"✅ Configuration loaded successfully!")
print(f"📁 Data directory: {DATA_DIR}")
print(f"🎯 Target drugs: {len(DRUG_CLASSES)} classes")
print(f"📊 Minimum required accuracy: {MINIMUM_REQUIRED_METRICS['top_3_accuracy']*100}%")
