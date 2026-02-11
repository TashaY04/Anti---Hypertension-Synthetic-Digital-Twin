"""
Causal Inference Module
Estimates Individual Treatment Effects (ITE) for each drug
Uses propensity score weighting and causal feature importance
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import config

class CausalInference:
    """
    Estimates causal effects of drugs on blood pressure reduction
    """
    
    def __init__(self):
        self.propensity_models = {}
        self.outcome_models = {}
        self.causal_weights = {}
        self.feature_importance = {}
        
    def estimate_propensity_scores(self, X, treatment):
        """
        Estimate propensity scores for each treatment (drug)
        Propensity score = P(receiving treatment | patient features)
        
        Args:
            X: Patient features
            treatment: Drug assigned
        
        Returns:
            Dictionary of propensity scores per drug
        """
        print("\n🎯 Estimating propensity scores...")
        
        propensity_scores = {}
        unique_treatments = np.unique(treatment)
        
        for drug in unique_treatments:
            # Binary treatment indicator
            y_binary = (treatment == drug).astype(int)
            
            # Train logistic regression to estimate P(treatment=drug | X)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y_binary)
            
            # Get probability of receiving this drug
            ps = model.predict_proba(X)[:, 1]
            
            # Store
            propensity_scores[drug] = ps
            self.propensity_models[drug] = model
            
            print(f"  Drug {drug}: Mean PS = {ps.mean():.3f}, Std = {ps.std():.3f}")
        
        print("✅ Propensity scores estimated")
        return propensity_scores
    
    def estimate_outcome_model(self, X, treatment, outcome):
        """
        Estimate outcome (BP reduction) model for each treatment
        
        Args:
            X: Patient features
            treatment: Drug assigned
            outcome: BP reduction achieved
        
        Returns:
            Dictionary of outcome models per drug
        """
        print("\n📈 Estimating outcome models...")
        
        unique_treatments = np.unique(treatment)
        
        for drug in unique_treatments:
            # Get samples that received this drug
            mask = (treatment == drug)
            X_drug = X[mask]
            y_drug = outcome[mask]
            
            if len(X_drug) > 10:  # Need minimum samples
                # Train outcome model: E[Y | X, Treatment=drug]
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X_drug, y_drug)
                
                self.outcome_models[drug] = model
                
                # Get feature importance
                self.feature_importance[drug] = model.feature_importances_
                
                print(f"  Drug {drug}: Trained on {len(X_drug)} samples")
            else:
                print(f"  Drug {drug}: Skipped (insufficient samples: {len(X_drug)})")
        
        print("✅ Outcome models estimated")
        return self.outcome_models
    
    def estimate_individual_treatment_effects(self, X, treatment=None, outcome=None):
        """
        Estimate ITE for each patient-drug combination
        ITE = Expected outcome if patient receives drug A - Expected outcome if receives drug B
        
        For recommendation, we estimate: E[BP_reduction | X, Drug=d] for each drug d
        
        Args:
            X: Patient features (can be single patient or batch)
            treatment: Actual treatments (for training, optional)
            outcome: Actual outcomes (for training, optional)
        
        Returns:
            Dictionary mapping drug -> predicted outcome for each patient
        """
        if treatment is not None and outcome is not None:
            # Training mode: estimate propensity scores and outcome models
            self.estimate_propensity_scores(X, treatment)
            self.estimate_outcome_model(X, treatment, outcome)
        
        # Prediction mode: estimate outcomes for all drugs
        print("\n🔮 Estimating individual treatment effects...")
        
        # Convert single sample to array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        ite_scores = {}
        
        for drug, model in self.outcome_models.items():
            # Predict expected outcome if this patient receives this drug
            predicted_outcome = model.predict(X)
            ite_scores[drug] = predicted_outcome
        
        return ite_scores
    
    def compute_causal_feature_weights(self):
        """
        Compute which features have the strongest causal effect on outcomes
        This helps explain WHY a drug works for a patient
        
        Returns:
            Dictionary of feature importance per drug
        """
        print("\n⚖️  Computing causal feature weights...")
        
        if not self.feature_importance:
            print("❌ No feature importance available. Train models first!")
            return None
        
        # Average feature importance across all drugs
        all_importances = np.array(list(self.feature_importance.values()))
        avg_importance = all_importances.mean(axis=0)
        
        self.causal_weights = avg_importance
        
        print("✅ Causal weights computed")
        return self.causal_weights
    
    def get_top_causal_features(self, feature_names, top_k=10):
        """
        Get the most causally important features
        """
        if self.causal_weights is None or len(self.causal_weights) == 0:
            self.compute_causal_feature_weights()
        
        # Sort by importance
        indices = np.argsort(self.causal_weights)[::-1][:top_k]
        
        top_features = []
        for idx in indices:
            if idx < len(feature_names):
                top_features.append({
                    'feature': feature_names[idx],
                    'importance': self.causal_weights[idx]
                })
        
        return top_features
    
    def recommend_drug_causal(self, patient_features, feature_names):
        """
        Recommend drug based on estimated causal effects
        
        Args:
            patient_features: Single patient's features
            feature_names: Names of features
        
        Returns:
            Dictionary with drug recommendations and scores
        """
        # Estimate ITE for this patient
        ite_scores = self.estimate_individual_treatment_effects(patient_features)
        
        # Sort by predicted outcome (higher BP reduction = better)
        ranked_drugs = sorted(
            ite_scores.items(),
            key=lambda x: x[1][0] if isinstance(x[1], np.ndarray) else x[1],
            reverse=True
        )
        
        recommendations = {
            'ranked_drugs': ranked_drugs,
            'top_drug': ranked_drugs[0][0] if ranked_drugs else None,
            'predicted_reduction': ranked_drugs[0][1][0] if ranked_drugs and isinstance(ranked_drugs[0][1], np.ndarray) else None
        }
        
        return recommendations
    
    def explain_recommendation(self, patient_features, drug, feature_names):
        """
        Explain why a drug is recommended for this patient
        using causal feature importance
        
        Args:
            patient_features: Patient's features
            drug: Recommended drug
            feature_names: Feature names
        
        Returns:
            Explanation text
        """
        if drug not in self.feature_importance:
            return "Explanation not available for this drug."
        
        # Get feature importance for this drug
        importance = self.feature_importance[drug]
        
        # Get top 5 most important features
        top_indices = np.argsort(importance)[::-1][:5]
        
        explanation = f"Drug recommendation based on:\n"
        for idx in top_indices:
            if idx < len(feature_names):
                feat_name = feature_names[idx]
                feat_value = patient_features[idx] if idx < len(patient_features) else "N/A"
                importance_score = importance[idx]
                
                explanation += f"  • {feat_name} = {feat_value:.2f} (importance: {importance_score:.3f})\n"
        
        return explanation
    
    def save_models(self, filepath=None):
        """Save causal models"""
        import pickle
        import os
        
        if filepath is None:
            filepath = os.path.join(config.MODELS_DIR, "causal_models.pkl")
        
        models = {
            'propensity_models': self.propensity_models,
            'outcome_models': self.outcome_models,
            'causal_weights': self.causal_weights,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"💾 Causal models saved to: {filepath}")
    
    def load_models(self, filepath=None):
        """Load pre-trained causal models"""
        import pickle
        import os
        
        if filepath is None:
            filepath = os.path.join(config.MODELS_DIR, "causal_models.pkl")
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
            
            self.propensity_models = models['propensity_models']
            self.outcome_models = models['outcome_models']
            self.causal_weights = models['causal_weights']
            self.feature_importance = models['feature_importance']
            
            print(f"✅ Causal models loaded from: {filepath}")
        else:
            print(f"❌ Model file not found: {filepath}")


# Testing function
def test_causal_inference():
    """Test causal inference module"""
    print("=" * 60)
    print("TESTING CAUSAL INFERENCE MODULE")
    print("=" * 60)
    
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    loader.create_synthetic_dataset_for_testing()
    data = loader.load_data()['synthetic']
    loader.preprocess_and_merge()
    
    # Prepare data
    feature_cols = [col for col in data.columns 
                   if col not in ['recommended_drug', 'drug_encoded', 'bp_reduction']]
    
    X = data[feature_cols].values[:1000]  # Use subset for testing
    treatment = data['drug_encoded'].values[:1000]
    outcome = data['bp_reduction'].values[:1000]
    
    # Initialize causal inference
    causal = CausalInference()
    
    # Train
    causal.estimate_individual_treatment_effects(X, treatment, outcome)
    
    # Get causal weights
    weights = causal.compute_causal_feature_weights()
    top_features = causal.get_top_causal_features(feature_cols)
    
    print("\n📊 Top causal features:")
    for feat in top_features:
        print(f"  {feat['feature']}: {feat['importance']:.4f}")
    
    # Test recommendation for single patient
    patient = X[0]
    recommendation = causal.recommend_drug_causal(patient, feature_cols)
    
    print(f"\n💊 Recommended drug: {recommendation['top_drug']}")
    print(f"📉 Predicted BP reduction: {recommendation['predicted_reduction']:.2f} mmHg")
    
    # Explain
    explanation = causal.explain_recommendation(patient, recommendation['top_drug'], feature_cols)
    print(f"\n📝 Explanation:\n{explanation}")
    
    # Save models
    causal.save_models()
    
    print("\n✅ Causal inference testing complete!")


if __name__ == "__main__":
    test_causal_inference()
