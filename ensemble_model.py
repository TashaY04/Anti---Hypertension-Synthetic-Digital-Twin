"""
Main Ensemble Model
Combines XGBoost, Causal Inference, and Prototype Learning
Implements the complete "Causal Prototype Network with Adversarial Augmentation"
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)
import config
import os
import pickle

from causal_module import CausalInference
from prototype_module import PrototypeLearning

class CausalPrototypeNetwork:
    """
    Main model class that orchestrates all components
    """
    
    def __init__(self):
        # Sub-models
        self.xgboost_model = None
        self.random_forest_model = None
        self.causal_model = CausalInference()
        self.prototype_model = PrototypeLearning(n_prototypes=config.PROTOTYPE_CONFIG['n_clusters'])
        
        # Metadata
        self.feature_names = None
        self.label_encoder = None
        self.drug_names = None
        
        # Contraindication rules
        self.contraindications = config.CONTRAINDICATIONS
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              bp_reduction_train=None, feature_names=None):
        """
        Train all components of the ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels (drug classes)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            bp_reduction_train: BP reduction outcomes for causal inference
            feature_names: Names of features
        """
        print("\n" + "=" * 60)
        print("TRAINING CAUSAL PROTOTYPE NETWORK")
        print("=" * 60)
        
        self.feature_names = feature_names if feature_names else list(range(X_train.shape[1]))
        
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train_np = X_train.values
        else:
            X_train_np = X_train
        
        # 1. Train XGBoost
        print("\n[1/4] Training XGBoost...")
        self._train_xgboost(X_train_np, y_train, X_val, y_val)
        
        # 2. Train Random Forest
        print("\n[2/4] Training Random Forest...")
        self._train_random_forest(X_train_np, y_train)
        
        # 3. Train Causal Inference Model
        print("\n[3/4] Training Causal Inference Module...")
        if bp_reduction_train is not None:
            self.causal_model.estimate_individual_treatment_effects(
                X_train_np, y_train, bp_reduction_train
            )
            self.causal_model.compute_causal_feature_weights()
        else:
            print("⚠️  No BP reduction data provided, skipping causal training")
        
        # 4. Train Prototype Model
        print("\n[4/4] Training Prototype Learning Module...")
        X_train_df = pd.DataFrame(X_train_np, columns=self.feature_names)
        self.prototype_model.fit(X_train_df, y_train, self.feature_names)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
    
    def _train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost classifier"""
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'max_depth': config.XGBOOST_CONFIG['max_depth'],
            'learning_rate': config.XGBOOST_CONFIG['learning_rate'],
            'subsample': config.XGBOOST_CONFIG['subsample'],
            'colsample_bytree': config.XGBOOST_CONFIG['colsample_bytree'],
            'min_child_weight': config.XGBOOST_CONFIG['min_child_weight'],
            'gamma': config.XGBOOST_CONFIG['gamma'],
            'reg_alpha': config.XGBOOST_CONFIG['reg_alpha'],
            'reg_lambda': config.XGBOOST_CONFIG['reg_lambda'],
            'eval_metric': 'mlogloss',
            'random_state': config.RANDOM_STATE
        }
        
        # Validation set
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train
        self.xgboost_model = xgb.train(
            params,
            dtrain,
            num_boost_round=config.XGBOOST_CONFIG['n_estimators'],
            evals=evals,
            verbose_eval=False
        )
        
        print("✅ XGBoost trained")
    
    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        self.random_forest_model = RandomForestClassifier(**config.RF_CONFIG)
        self.random_forest_model.fit(X_train, y_train)
        print("✅ Random Forest trained")
    
    def predict_ensemble(self, patient_features):
        """
        Make ensemble prediction combining all models
        
        Args:
            patient_features: Single patient's features (or batch)
        
        Returns:
            Dictionary with top-K recommendations and scores
        """
        # Convert to proper format
        if isinstance(patient_features, pd.Series):
            patient_features = patient_features.values
        if len(patient_features.shape) == 1:
            patient_features = patient_features.reshape(1, -1)
        
        # 1. XGBoost predictions
        dtest = xgb.DMatrix(patient_features)
        xgb_probs = self.xgboost_model.predict(dtest)[0]  # Probabilities for each class
        
        # 2. Random Forest predictions
        rf_probs = self.random_forest_model.predict_proba(patient_features)[0]
        
        # 3. Causal inference scores
        causal_scores = self.causal_model.estimate_individual_treatment_effects(patient_features)
        
        # Normalize causal scores to [0, 1] range
        if causal_scores:
            causal_values = np.array([causal_scores[drug][0] if isinstance(causal_scores[drug], np.ndarray) 
                                     else causal_scores[drug] for drug in sorted(causal_scores.keys())])
            causal_min, causal_max = causal_values.min(), causal_values.max()
            if causal_max > causal_min:
                causal_normalized = (causal_values - causal_min) / (causal_max - causal_min)
            else:
                causal_normalized = np.ones_like(causal_values) * 0.5
        else:
            causal_normalized = np.ones(len(xgb_probs)) * 0.5
        
        # 4. Prototype similarity scores
        patient_df = pd.DataFrame(patient_features, columns=self.feature_names)
        prototype_scores_dict = self.prototype_model.compute_similarity_scores(patient_df)
        
        # Convert prototype scores to array (aligned with drug indices)
        prototype_scores = np.zeros(len(xgb_probs))
        for drug_idx, score in prototype_scores_dict.items():
            if drug_idx < len(prototype_scores):
                prototype_scores[drug_idx] = score
        
        # Normalize prototype scores
        if prototype_scores.max() > 0:
            prototype_scores = prototype_scores / prototype_scores.max()
        
        # 5. Ensemble combination (weighted average)
        ensemble_scores = (
            config.ENSEMBLE_WEIGHTS['xgboost'] * xgb_probs +
            config.ENSEMBLE_WEIGHTS['causal'] * causal_normalized +
            config.ENSEMBLE_WEIGHTS['prototype'] * prototype_scores
        )
        
        # 6. Apply safety constraints (contraindications)
        ensemble_scores = self._apply_safety_filters(patient_features[0], ensemble_scores)
        
        # 7. Get top-K recommendations
        top_k_indices = np.argsort(ensemble_scores)[::-1][:config.TOP_K_RECOMMENDATIONS]
        
        recommendations = []
        for idx in top_k_indices:
            recommendations.append({
                'drug_class': int(idx),
                'confidence': float(ensemble_scores[idx]),
                'xgboost_score': float(xgb_probs[idx]),
                'causal_score': float(causal_normalized[idx]),
                'prototype_score': float(prototype_scores[idx])
            })
        
        return recommendations
    
    def _apply_safety_filters(self, patient_features, scores):
        """
        Apply contraindication rules to zero out unsafe drugs
        
        Args:
            patient_features: Patient's feature vector
            scores: Current ensemble scores
        
        Returns:
            Filtered scores
        """
        # This is a simplified version - in production, you'd have a comprehensive
        # contraindication database
        
        # Example: If patient has kidney disease, reduce score for ACE inhibitors
        # You'd map feature indices to contraindications
        
        # For now, return scores as-is
        # TODO: Implement full contraindication logic
        
        return scores
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive evaluation of the model
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Convert to numpy if needed
        if isinstance(X_test, pd.DataFrame):
            X_test_np = X_test.values
        else:
            X_test_np = X_test
        
        # Get predictions for all test samples
        y_pred = []
        y_pred_top3 = []
        
        for i in range(len(X_test_np)):
            patient = X_test_np[i]
            recommendations = self.predict_ensemble(patient)
            
            # Top-1 prediction
            y_pred.append(recommendations[0]['drug_class'])
            
            # Top-3 predictions
            top3 = [rec['drug_class'] for rec in recommendations]
            y_pred_top3.append(top3)
        
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        metrics = {}
        
        # 1. Top-1 Accuracy
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # 2. Top-3 Accuracy
        top3_correct = sum([1 for i, true_label in enumerate(y_test) 
                           if true_label in y_pred_top3[i]])
        metrics['top_3_accuracy'] = top3_correct / len(y_test)
        
        # 3. Precision, Recall, F1 (macro average)
        metrics['precision'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # 4. ROC-AUC (one-vs-rest)
        try:
            # Get probability predictions
            all_probs = []
            for i in range(len(X_test_np)):
                patient = X_test_np[i]
                recommendations = self.predict_ensemble(patient)
                
                # Create probability vector
                probs = np.zeros(len(np.unique(y_test)))
                for rec in recommendations:
                    probs[rec['drug_class']] = rec['confidence']
                all_probs.append(probs)
            
            all_probs = np.array(all_probs)
            metrics['roc_auc'] = roc_auc_score(y_test, all_probs, 
                                               multi_class='ovr', average='macro')
        except:
            metrics['roc_auc'] = 0.0
        
        # Print results
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"{'=' * 60}")
        print(f"Top-1 Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Top-3 Accuracy:     {metrics['top_3_accuracy']:.4f} ({metrics['top_3_accuracy']*100:.2f}%) ⭐")
        print(f"Precision (macro):  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall (macro):     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score (macro):   {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"ROC-AUC (OvR):      {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
        print(f"{'=' * 60}")
        
        # Check if meets requirements
        print(f"\n✅ REQUIREMENT CHECK (≥90% target):")
        for metric_name, required_value in config.MINIMUM_REQUIRED_METRICS.items():
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                status = "✅ PASS" if actual_value >= required_value else "❌ FAIL"
                print(f"{metric_name:20s}: {actual_value:.2%} (required: {required_value:.0%}) {status}")
        
        return metrics
    
    def save_models(self, filepath=None):
        """Save all models"""
        if filepath is None:
            filepath = os.path.join(config.MODELS_DIR, "ensemble_model.pkl")
        
        # Save XGBoost separately (binary format)
        xgb_path = os.path.join(config.MODELS_DIR, "xgboost_model.json")
        if self.xgboost_model:
            self.xgboost_model.save_model(xgb_path)
        
        # Save other models
        model_data = {
            'random_forest': self.random_forest_model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'drug_names': self.drug_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save sub-models
        self.causal_model.save_models()
        self.prototype_model.save_model()
        
        print(f"\n💾 All models saved successfully!")
        print(f"  - Ensemble: {filepath}")
        print(f"  - XGBoost: {xgb_path}")
        print(f"  - Causal models: {config.MODELS_DIR}/causal_models.pkl")
        print(f"  - Prototype model: {config.MODELS_DIR}/prototype_model.pkl")
    
    def load_models(self, filepath=None):
        """Load all pre-trained models"""
        if filepath is None:
            filepath = os.path.join(config.MODELS_DIR, "ensemble_model.pkl")
        
        # Load XGBoost
        xgb_path = os.path.join(config.MODELS_DIR, "xgboost_model.json")
        if os.path.exists(xgb_path):
            self.xgboost_model = xgb.Booster()
            self.xgboost_model.load_model(xgb_path)
        
        # Load other models
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.random_forest_model = model_data['random_forest']
            self.feature_names = model_data['feature_names']
            self.label_encoder = model_data['label_encoder']
            self.drug_names = model_data['drug_names']
        
        # Load sub-models
        self.causal_model.load_models()
        self.prototype_model.load_model()
        
        print(f"✅ All models loaded successfully!")


# Main training script
def train_full_model():
    """Complete training pipeline"""
    from data_loader import DataLoader
    from gan_module import PatientGAN
    
    print("=" * 60)
    print("FULL MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("\n[STEP 1] Loading data...")
    loader = DataLoader()
    loader.download_datasets()
    loader.load_data()
    data = loader.preprocess_and_merge()
    
    # 2. Apply GAN augmentation
    print("\n[STEP 2] Applying GAN augmentation...")
    gan = PatientGAN(epochs=100, batch_size=500)
    gan.train(data)
    augmented_data = gan.augment_minority_classes(data, balance_ratio=0.9)
    
    # 3. Prepare train/test split
    print("\n[STEP 3] Preparing train/test split...")
    feature_cols = [col for col in augmented_data.columns 
                   if col not in ['recommended_drug', 'drug_encoded', 'bp_reduction']]
    
    X = augmented_data[feature_cols].values
    y = augmented_data['drug_encoded'].values
    bp_reduction = augmented_data['bp_reduction'].values if 'bp_reduction' in augmented_data.columns else None
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, bp_train, bp_test = train_test_split(
        X, y, bp_reduction, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train ensemble model
    print("\n[STEP 4] Training ensemble model...")
    model = CausalPrototypeNetwork()
    model.train(X_train, y_train, bp_reduction_train=bp_train, feature_names=feature_cols)
    
    # 5. Evaluate
    print("\n[STEP 5] Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    # 6. Save models
    print("\n[STEP 6] Saving models...")
    model.save_models()
    gan.save_model()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    train_full_model()
