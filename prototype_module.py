"""
Prototype Learning Module
Uses K-Prototypes clustering to identify patient archetypes
Then classifies new patients based on nearest prototype
"""

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import pairwise_distances
import config
import os

class PrototypeLearning:
    """
    Learn patient prototypes (archetypes) and use for drug recommendation
    """
    
    def __init__(self, n_prototypes=8):
        self.n_prototypes = n_prototypes
        self.model = None
        self.prototypes = None
        self.prototype_labels = None
        self.prototype_drug_mapping = {}
        self.categorical_indices = []
        self.feature_names = None
        
    def identify_categorical_features(self, X, feature_names):
        """
        Identify which columns are categorical
        K-Prototypes needs to know this
        """
        categorical_cols = [
            'gender', 'diabetes', 'kidney_disease', 'smoker',
            'high_stress', 'sedentary_lifestyle', 'high_cholesterol',
            'hormonal_imbalance', 'pregnancy_postpartum', 'age_group',
            'diabetes_kidney'
        ]
        
        indices = []
        for i, name in enumerate(feature_names):
            if name in categorical_cols:
                indices.append(i)
        
        self.categorical_indices = indices
        print(f"  Identified {len(indices)} categorical features")
        
        return indices
    
    def fit(self, X, y, feature_names):
        """
        Fit K-Prototypes model to learn patient archetypes
        
        Args:
            X: Patient features
            y: Drug labels
            feature_names: Names of features
        """
        print(f"\n🎭 Learning {self.n_prototypes} patient prototypes...")
        
        self.feature_names = feature_names
        
        # Identify categorical features
        self.identify_categorical_features(X, feature_names)
        
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
        
        # Initialize K-Prototypes
        self.model = KPrototypes(
            n_clusters=self.n_prototypes,
            init='Cao',
            n_init=5,
            verbose=1,
            random_state=42
        )
        
        # Fit model
        print("🔄 Clustering patients into prototypes...")
        clusters = self.model.fit_predict(X_df, categorical=self.categorical_indices)
        
        # Store prototypes
        self.prototypes = self.model.cluster_centroids_
        self.prototype_labels = clusters
        
        # Map each prototype to most common drug
        self._map_prototypes_to_drugs(clusters, y)
        
        # Describe prototypes
        self._describe_prototypes(X_df, clusters, y)
        
        print(f"✅ Learned {self.n_prototypes} patient prototypes")
        
        return clusters
    
    def _map_prototypes_to_drugs(self, clusters, y):
        """
        For each prototype, find the most commonly prescribed drug
        """
        print("\n💊 Mapping prototypes to drugs...")
        
        for prototype_id in range(self.n_prototypes):
            # Get all patients in this prototype
            mask = (clusters == prototype_id)
            drugs_in_prototype = y[mask]
            
            if len(drugs_in_prototype) > 0:
                # Find most common drug
                unique, counts = np.unique(drugs_in_prototype, return_counts=True)
                most_common_drug = unique[np.argmax(counts)]
                confidence = counts.max() / len(drugs_in_prototype)
                
                self.prototype_drug_mapping[prototype_id] = {
                    'drug': most_common_drug,
                    'confidence': confidence,
                    'n_patients': len(drugs_in_prototype)
                }
                
                print(f"  Prototype {prototype_id}: Drug {most_common_drug} "
                      f"(confidence: {confidence:.2%}, n={len(drugs_in_prototype)})")
        
        print("✅ Prototype-drug mapping complete")
    
    def _describe_prototypes(self, X, clusters, y):
        """
        Generate human-readable descriptions of each prototype
        """
        print("\n📋 Prototype Descriptions:")
        print("=" * 60)
        
        for prototype_id in range(self.n_prototypes):
            mask = (clusters == prototype_id)
            prototype_patients = X[mask]
            
            if len(prototype_patients) == 0:
                continue
            
            print(f"\nPrototype {prototype_id} ({len(prototype_patients)} patients):")
            
            # Get mean values for numerical features
            numerical_features = ['age', 'systolic_bp', 'diastolic_bp', 'bmi', 'duration_years']
            for feat in numerical_features:
                if feat in X.columns:
                    mean_val = prototype_patients[feat].mean()
                    print(f"  • {feat}: {mean_val:.1f}")
            
            # Get mode for categorical features
            categorical_features = ['gender', 'diabetes', 'kidney_disease', 'smoker']
            for feat in categorical_features:
                if feat in X.columns:
                    mode_val = prototype_patients[feat].mode()
                    if len(mode_val) > 0:
                        print(f"  • {feat}: {mode_val.values[0]}")
            
            # Most common drug
            if prototype_id in self.prototype_drug_mapping:
                drug_info = self.prototype_drug_mapping[prototype_id]
                print(f"  • Most common drug: {drug_info['drug']} "
                      f"({drug_info['confidence']:.1%} of patients)")
        
        print("=" * 60)
    
    def predict_prototype(self, X):
        """
        Assign new patient(s) to nearest prototype
        
        Args:
            X: Patient features
        
        Returns:
            Prototype ID(s)
        """
        if self.model is None:
            raise ValueError("Model not trained! Call fit() first.")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        
        # Predict prototype
        prototype_ids = self.model.predict(X_df, categorical=self.categorical_indices)
        
        return prototype_ids
    
    def recommend_drug_prototype(self, patient_features):
        """
        Recommend drug based on prototype matching
        
        Args:
            patient_features: Single patient's features
        
        Returns:
            Dictionary with recommendation and confidence
        """
        # Find nearest prototype
        prototype_id = self.predict_prototype(patient_features)[0]
        
        # Get drug recommendation from this prototype
        if prototype_id in self.prototype_drug_mapping:
            drug_info = self.prototype_drug_mapping[prototype_id]
            
            recommendation = {
                'prototype_id': prototype_id,
                'drug': drug_info['drug'],
                'confidence': drug_info['confidence'],
                'n_similar_patients': drug_info['n_patients']
            }
        else:
            recommendation = {
                'prototype_id': prototype_id,
                'drug': None,
                'confidence': 0.0,
                'n_similar_patients': 0
            }
        
        return recommendation
    
    def compute_similarity_scores(self, patient_features):
        """
        Compute similarity to ALL prototypes (not just nearest)
        Returns scores for each drug based on weighted similarity
        
        Args:
            patient_features: Single patient's features
        
        Returns:
            Dictionary mapping drug -> similarity score
        """
        if self.model is None:
            raise ValueError("Model not trained! Call fit() first.")
        
        # Convert to DataFrame
        if isinstance(patient_features, np.ndarray):
            if len(patient_features.shape) == 1:
                patient_features = patient_features.reshape(1, -1)
            patient_df = pd.DataFrame(patient_features, columns=self.feature_names)
        else:
            patient_df = patient_features
        
        # Compute distance to each prototype
        distances = []
        for i in range(self.n_prototypes):
            # Get prototype centroid
            prototype = self.prototypes[i]
            
            # Compute distance (K-Prototypes uses custom distance metric)
            # Simplified: use euclidean for numerical, hamming for categorical
            dist = self._compute_prototype_distance(patient_df.values[0], prototype)
            distances.append(dist)
        
        # Convert distances to similarities (inverse)
        similarities = 1 / (1 + np.array(distances))
        
        # Weight by prototype confidence and aggregate by drug
        drug_scores = {}
        for prototype_id, similarity in enumerate(similarities):
            if prototype_id in self.prototype_drug_mapping:
                drug = self.prototype_drug_mapping[prototype_id]['drug']
                confidence = self.prototype_drug_mapping[prototype_id]['confidence']
                
                # Weighted score
                score = similarity * confidence
                
                # Aggregate (take max if drug appears in multiple prototypes)
                if drug not in drug_scores:
                    drug_scores[drug] = score
                else:
                    drug_scores[drug] = max(drug_scores[drug], score)
        
        return drug_scores
    
    def _compute_prototype_distance(self, patient, prototype):
        """
        Compute distance between patient and prototype
        Uses mixed distance metric for numerical + categorical
        """
        distance = 0
        
        # Numerical features: Euclidean distance
        numerical_mask = np.ones(len(patient), dtype=bool)
        numerical_mask[self.categorical_indices] = False
        
        if np.any(numerical_mask):
            patient_num = patient[numerical_mask].astype(float)
            proto_num = prototype[0][numerical_mask].astype(float)  # prototype is tuple
            distance += np.sqrt(np.sum((patient_num - proto_num) ** 2))
        
        # Categorical features: Hamming distance
        if len(self.categorical_indices) > 0:
            patient_cat = patient[self.categorical_indices]
            proto_cat = prototype[1][self.categorical_indices]  # categorical part
            distance += np.sum(patient_cat != proto_cat)
        
        return distance
    
    def get_prototype_description(self, prototype_id):
        """
        Get human-readable description of a prototype
        """
        if prototype_id not in self.prototype_drug_mapping:
            return "Unknown prototype"
        
        drug_info = self.prototype_drug_mapping[prototype_id]
        
        description = f"Prototype {prototype_id}:\n"
        description += f"  • Most common treatment: {drug_info['drug']}\n"
        description += f"  • Based on {drug_info['n_patients']} similar patients\n"
        description += f"  • Confidence: {drug_info['confidence']:.1%}\n"
        
        return description
    
    def save_model(self, filepath=None):
        """Save prototype model"""
        import pickle
        
        if filepath is None:
            filepath = os.path.join(config.MODELS_DIR, "prototype_model.pkl")
        
        model_data = {
            'model': self.model,
            'prototypes': self.prototypes,
            'prototype_labels': self.prototype_labels,
            'prototype_drug_mapping': self.prototype_drug_mapping,
            'categorical_indices': self.categorical_indices,
            'feature_names': self.feature_names,
            'n_prototypes': self.n_prototypes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Prototype model saved to: {filepath}")
    
    def load_model(self, filepath=None):
        """Load pre-trained prototype model"""
        import pickle
        
        if filepath is None:
            filepath = os.path.join(config.MODELS_DIR, "prototype_model.pkl")
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.prototypes = model_data['prototypes']
            self.prototype_labels = model_data['prototype_labels']
            self.prototype_drug_mapping = model_data['prototype_drug_mapping']
            self.categorical_indices = model_data['categorical_indices']
            self.feature_names = model_data['feature_names']
            self.n_prototypes = model_data['n_prototypes']
            
            print(f"✅ Prototype model loaded from: {filepath}")
        else:
            print(f"❌ Model file not found: {filepath}")


# Testing function
def test_prototype_learning():
    """Test prototype learning module"""
    print("=" * 60)
    print("TESTING PROTOTYPE LEARNING MODULE")
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
    
    X = data[feature_cols][:2000]  # Use subset
    y = data['drug_encoded'].values[:2000]
    
    # Initialize and train
    prototype_model = PrototypeLearning(n_prototypes=8)
    clusters = prototype_model.fit(X, y, feature_cols)
    
    # Test prediction for single patient
    patient = X.iloc[0]
    recommendation = prototype_model.recommend_drug_prototype(patient)
    
    print(f"\n💊 Prototype-based recommendation:")
    print(f"  • Assigned to prototype: {recommendation['prototype_id']}")
    print(f"  • Recommended drug: {recommendation['drug']}")
    print(f"  • Confidence: {recommendation['confidence']:.2%}")
    print(f"  • Based on {recommendation['n_similar_patients']} similar patients")
    
    # Get similarity scores for all drugs
    drug_scores = prototype_model.compute_similarity_scores(patient)
    print(f"\n📊 Similarity scores for all drugs:")
    for drug, score in sorted(drug_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  • Drug {drug}: {score:.4f}")
    
    # Save model
    prototype_model.save_model()
    
    print("\n✅ Prototype learning testing complete!")


if __name__ == "__main__":
    test_prototype_learning()
