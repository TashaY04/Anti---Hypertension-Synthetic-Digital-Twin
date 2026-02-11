"""
Data Loader and Preprocessing Module
Handles downloading, loading, cleaning, and merging of datasets
"""

import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import config

class DataLoader:
    """Handles all data loading and preprocessing operations"""
    
    def __init__(self):
        self.raw_data = {}
        self.merged_data = None
        self.processed_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def download_datasets(self):
        """
        Download datasets from specified URLs
        NOTE: You'll need to manually download from Kaggle/PhysioNet and place in data/raw/
        Kaggle requires API authentication
        """
        print("📥 Downloading datasets...")
        
        # For Kaggle datasets, you need to:
        # 1. Install kaggle: pip install kaggle
        # 2. Set up API credentials: https://www.kaggle.com/docs/api
        # 3. Use: kaggle datasets download -d <dataset-path>
        
        print("\n⚠️  MANUAL DOWNLOAD REQUIRED:")
        print("=" * 60)
        print("Please download the following datasets manually:")
        print("\n1. Kaggle Hypertension Dataset:")
        print("   - Visit: https://www.kaggle.com/datasets")
        print("   - Search: 'hypertension' or 'blood pressure'")
        print("   - Download and place CSV in: data/raw/hypertension_dataset.csv")
        
        print("\n2. Kaggle PK/PD Dataset:")
        print("   - Visit: https://www.kaggle.com/datasets")
        print("   - Search: 'pharmacokinetics pharmacodynamics'")
        print("   - Download and place CSV in: data/raw/pkpd_dataset.csv")
        
        print("\n3. PhysioNet Dataset:")
        print("   - Visit: https://physionet.org/")
        print("   - Download MIMIC-III or relevant cardiovascular dataset")
        print("   - Place CSV in: data/raw/physionet_dataset.csv")
        
        print("\n4. Synthea Dataset:")
        print("   - Visit: https://synthea.mitre.org/")
        print("   - Generate synthetic patient data")
        print("   - Place CSV in: data/raw/synthea_dataset.csv")
        print("=" * 60)
        
        print("\n💡 Alternative: I'll create a synthetic dataset for testing!")
        self.create_synthetic_dataset_for_testing()
        
    def create_synthetic_dataset_for_testing(self):
        """
        Create a synthetic dataset for development/testing
        This simulates the structure of real hypertension data
        """
        print("\n🔧 Creating synthetic test dataset...")
        
        np.random.seed(42)
        n_samples = 5000
        
        # Generate features
        data = {
            # Demographics
            'age': np.random.randint(25, 85, n_samples),
            'gender': np.random.choice(['male', 'female'], n_samples),
            
            # Vital signs
            'systolic_bp': np.random.randint(120, 200, n_samples),
            'diastolic_bp': np.random.randint(70, 120, n_samples),
            'heart_rate': np.random.randint(55, 110, n_samples),
            'bmi': np.random.uniform(18, 40, n_samples),
            
            # Disease history
            'duration_years': np.random.choice([0.5, 1, 2, 3, 5, 7, 10, 15], n_samples),
            
            # Risk factors (binary)
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'kidney_disease': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'smoker': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'high_stress': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'sedentary_lifestyle': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'high_cholesterol': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'hormonal_imbalance': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'pregnancy_postpartum': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Generate target (recommended drug) based on logical rules
        df['recommended_drug'] = self._assign_drugs_intelligently(df)
        
        # Generate outcome (BP reduction after treatment)
        df['bp_reduction'] = self._generate_bp_reduction(df)
        
        # Save synthetic dataset
        synthetic_path = os.path.join(config.RAW_DATA_DIR, "synthetic_dataset.csv")
        df.to_csv(synthetic_path, index=False)
        
        print(f"✅ Created synthetic dataset: {n_samples} samples")
        print(f"📁 Saved to: {synthetic_path}")
        print(f"\n📊 Drug distribution:")
        print(df['recommended_drug'].value_counts())
        
        return df
    
    def _assign_drugs_intelligently(self, df):
        """
        Assign drugs based on patient characteristics (simulating clinical logic)
        """
        drugs = []
        
        for _, row in df.iterrows():
            # Young, mild hypertension → Diuretic
            if row['age'] < 45 and row['systolic_bp'] < 150:
                drug = 'Diuretic'
            
            # Diabetes present → ACE Inhibitor or ARB (renal protective)
            elif row['diabetes'] == 1:
                drug = np.random.choice(['ACE_Inhibitor', 'ARB'])
            
            # Kidney disease → Avoid ACE/ARB, use Calcium Channel Blocker
            elif row['kidney_disease'] == 1:
                drug = 'Calcium_Channel_Blocker'
            
            # Pregnant/postpartum → Avoid ACE/ARB (teratogenic)
            elif row['pregnancy_postpartum'] == 1:
                drug = np.random.choice(['Calcium_Channel_Blocker', 'Beta_Blocker'])
            
            # High BP + older age → Combination therapy
            elif row['systolic_bp'] > 170 and row['age'] > 60:
                drug = 'Combination_Therapy'
            
            # Smoker + high stress → Beta blocker
            elif row['smoker'] == 1 or row['high_stress'] == 1:
                drug = 'Beta_Blocker'
            
            # Default cases
            else:
                drug = np.random.choice([
                    'ACE_Inhibitor', 'ARB', 'Calcium_Channel_Blocker',
                    'Diuretic', 'Beta_Blocker'
                ])
            
            drugs.append(drug)
        
        return drugs
    
    def _generate_bp_reduction(self, df):
        """
        Generate realistic BP reduction based on drug and patient characteristics
        """
        reductions = []
        
        for _, row in df.iterrows():
            drug = row['recommended_drug']
            base_reduction = np.random.uniform(10, 25)
            
            # Adjust based on patient factors
            if row['age'] > 65:
                base_reduction *= 0.9  # Elderly respond less
            if row['diabetes'] == 1:
                base_reduction *= 0.85
            if row['kidney_disease'] == 1:
                base_reduction *= 0.8
            if row['duration_years'] > 10:
                base_reduction *= 0.75  # Chronic cases harder to treat
            
            # Add some noise
            reduction = base_reduction + np.random.normal(0, 3)
            reductions.append(max(5, min(40, reduction)))  # Clamp between 5-40 mmHg
        
        return reductions
    
    def load_data(self):
        """
        Load all available datasets
        """
        print("\n📂 Loading datasets...")
        
        # Check if synthetic dataset exists, if not create it
        synthetic_path = os.path.join(config.RAW_DATA_DIR, "synthetic_dataset.csv")
        if not os.path.exists(synthetic_path):
            print("Synthetic dataset not found. Creating...")
            self.create_synthetic_dataset_for_testing()
        
        # Load datasets
        datasets_to_load = []
        
        # Try loading real datasets first
        for dataset_name, dataset_info in config.DATASETS.items():
            filepath = os.path.join(config.RAW_DATA_DIR, dataset_info['filename'])
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    self.raw_data[dataset_name] = df
                    datasets_to_load.append(dataset_name)
                    print(f"✅ Loaded {dataset_name}: {len(df)} rows")
                except Exception as e:
                    print(f"❌ Error loading {dataset_name}: {e}")
        
        # Always load synthetic dataset
        if os.path.exists(synthetic_path):
            df = pd.read_csv(synthetic_path)
            self.raw_data['synthetic'] = df
            datasets_to_load.append('synthetic')
            print(f"✅ Loaded synthetic dataset: {len(df)} rows")
        
        if not datasets_to_load:
            raise ValueError("No datasets loaded! Please download datasets or use synthetic data.")
        
        print(f"\n📊 Total datasets loaded: {len(datasets_to_load)}")
        return self.raw_data
    
    def preprocess_and_merge(self):
        """
        Preprocess and merge all datasets into a unified format
        """
        print("\n🔄 Preprocessing and merging datasets...")
        
        if not self.raw_data:
            self.load_data()
        
        # For now, we'll use the synthetic dataset
        # In production, you'd merge multiple datasets with careful schema alignment
        if 'synthetic' in self.raw_data:
            self.merged_data = self.raw_data['synthetic'].copy()
        else:
            # Merge multiple datasets (implement when real data available)
            self.merged_data = pd.concat(self.raw_data.values(), ignore_index=True)
        
        # Handle missing values
        self.merged_data = self._handle_missing_values(self.merged_data)
        
        # Encode categorical variables
        self.merged_data = self._encode_categorical(self.merged_data)
        
        # Feature engineering
        self.merged_data = self._engineer_features(self.merged_data)
        
        print(f"✅ Preprocessed data: {len(self.merged_data)} rows, {len(self.merged_data.columns)} columns")
        
        return self.merged_data
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("  - Handling missing values...")
        
        # Numerical: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical variables"""
        print("  - Encoding categorical variables...")
        
        # Gender encoding
        if 'gender' in df.columns and df['gender'].dtype == 'object':
            df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        
        # Target encoding (drug names)
        if 'recommended_drug' in df.columns and df['recommended_drug'].dtype == 'object':
            self.label_encoders['drug'] = LabelEncoder()
            df['drug_encoded'] = self.label_encoders['drug'].fit_transform(df['recommended_drug'])
        
        return df
    
    def _engineer_features(self, df):
        """Create interaction features"""
        print("  - Engineering features...")
        
        # Age groups - convert to int instead of category
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=[0, 1, 2])
        df['age_group'] = df['age_group'].astype(int)  # Convert category to int
        
        # BP severity
        df['bp_severity'] = ((df['systolic_bp'] - 120) + (df['diastolic_bp'] - 80)) / 2
        
        # Risk score (sum of risk factors)
        risk_columns = ['diabetes', 'kidney_disease', 'smoker', 'high_stress', 
                       'sedentary_lifestyle', 'high_cholesterol']
        df['total_risk_score'] = df[risk_columns].sum(axis=1)
        
        # Interaction features
        df['age_bp_interaction'] = df['age'] * df['systolic_bp'] / 100
        df['diabetes_kidney'] = df['diabetes'] * df['kidney_disease']
        df['age_duration'] = df['age'] * df['duration_years'] / 10
        
        return df
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """
        Prepare final train/test split
        """
        print("\n✂️  Splitting data into train/test sets...")
        
        if self.merged_data is None:
            self.preprocess_and_merge()
        
        # Separate features and target
        feature_columns = [col for col in self.merged_data.columns 
                          if col not in ['recommended_drug', 'drug_encoded', 'bp_reduction']]
        
        X = self.merged_data[feature_columns]
        y = self.merged_data['drug_encoded']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        print(f"📊 Feature columns: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self):
        """Save processed data for later use"""
        if self.merged_data is not None:
            output_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_data.csv")
            self.merged_data.to_csv(output_path, index=False)
            print(f"💾 Saved processed data to: {output_path}")


# Utility function for quick testing
if __name__ == "__main__":
    print("=" * 60)
    print("DATA LOADER - Testing Mode")
    print("=" * 60)
    
    loader = DataLoader()
    loader.download_datasets()  # Will create synthetic data
    loader.load_data()
    loader.preprocess_and_merge()
    X_train, X_test, y_train, y_test = loader.prepare_train_test_split()
    loader.save_processed_data()
    
    print("\n✅ Data loading and preprocessing complete!")
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"🎯 Number of drug classes: {len(np.unique(y_train))}")
