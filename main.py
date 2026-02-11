"""
MAIN EXECUTION SCRIPT
Digital Twin - Hypertension Drug Recommendation System
Causal Prototype Network with Adversarial Patient Augmentation
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Digital Twin Hypertension Drug Recommendation System'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'ui', 'evaluate'],
        default='train',
        help='Execution mode: train (train models), predict (single prediction), ui (launch UI), evaluate (test model)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save/load models'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIGITAL TWIN - HYPERTENSION DRUG RECOMMENDATION")
    print("Causal Prototype Network with Adversarial Augmentation")
    print("=" * 60)
    
    if args.mode == 'train':
        print("\n🎓 MODE: Training")
        train_mode()
    
    elif args.mode == 'predict':
        print("\n🔮 MODE: Prediction")
        predict_mode()
    
    elif args.mode == 'ui':
        print("\n🖥️  MODE: UI Launch")
        ui_mode()
    
    elif args.mode == 'evaluate':
        print("\n📊 MODE: Evaluation")
        evaluate_mode()


def train_mode():
    """Train all models from scratch"""
    from ensemble_model import train_full_model
    
    print("\n" + "=" * 60)
    print("STARTING FULL TRAINING PIPELINE")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Load and preprocess datasets")
    print("  2. Train CTGAN for synthetic data augmentation")
    print("  3. Train Causal Inference models")
    print("  4. Train Prototype Learning models")
    print("  5. Train XGBoost and Random Forest ensemble")
    print("  6. Evaluate on test set")
    print("  7. Save all models")
    print("\n⏱️  Estimated time: 15-30 minutes (depending on dataset size)")
    
    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    model, metrics = train_full_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📊 Final Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
    
    print("\n💾 Models saved to: models/")
    print("\n✅ Ready for prediction!")


def predict_mode():
    """Make a single prediction"""
    from ui_integration import DigitalTwinPredictor
    
    print("\n" + "=" * 60)
    print("SINGLE PATIENT PREDICTION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('models/ensemble_model.pkl'):
        print("\n❌ No trained model found!")
        print("Please run: python main.py --mode train")
        return
    
    # Load model
    predictor = DigitalTwinPredictor()
    
    # Get patient data interactively
    print("\n📋 Enter patient information:")
    
    patient_data = {}
    
    # Demographics
    patient_data['gender'] = input("Gender (male/female): ").lower()
    patient_data['age'] = int(input("Age: "))
    
    # Vitals
    patient_data['systolic'] = int(input("Systolic BP (mmHg): "))
    patient_data['diastolic'] = int(input("Diastolic BP (mmHg): "))
    
    # Duration
    print("\nDuration of hypertension:")
    print("  1. < 1 year")
    print("  2. 1-5 years")
    print("  3. > 5 years")
    duration_choice = input("Choose (1/2/3): ")
    duration_map = {'1': '< 1 year', '2': '1–5 years', '3': '> 5 years'}
    patient_data['duration'] = duration_map.get(duration_choice, '1–5 years')
    
    # Risk factors
    print("\nRisk factors (y/n):")
    risk_factors = {}
    for risk in ['Diabetes', 'Kidney Disease', 'Smoker', 'High Stress', 
                 'Sedentary Lifestyle', 'High Cholesterol']:
        response = input(f"  {risk}: ").lower()
        risk_factors[risk] = (response == 'y')
    
    patient_data['risks'] = risk_factors
    patient_data['bp_med'] = 'No'
    patient_data['allergy'] = 'No'
    
    # Make prediction
    print("\n🔮 Analyzing patient profile...")
    results = predictor.predict(patient_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("RECOMMENDATION RESULTS")
    print("=" * 60)
    
    print("\n" + results['patient_profile'])
    
    print("\n💊 Top Drug Recommendations:")
    for rec in results['top_recommendations']:
        print(f"\n{rec['rank']}. {rec['drug_name']}")
        print(f"   Confidence: {rec['confidence']:.1f}%")
        print(f"   Expected BP Reduction: {rec['expected_bp_reduction']:.1f} mmHg")
        print(f"   Reasoning: {rec['explanation']}")
    
    if results['safety_warnings']:
        print("\n⚠️  Safety Warnings:")
        for warning in results['safety_warnings']:
            print(f"   {warning}")
    
    print("\n" + "=" * 60)


def ui_mode():
    """Launch the Tkinter UI"""
    print("\n🚀 Launching UI...")
    
    # Check if model exists
    model_exists = os.path.exists('models/ensemble_model.pkl')
    
    if not model_exists:
        print("\n⚠️  WARNING: No trained model found!")
        print("The UI will launch, but predictions will use default rules.")
        print("For full functionality, please train the model first:")
        print("  python main.py --mode train")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Import and run UI
    try:
        # This would import your original UI file
        # For now, we'll show a message
        print("\n" + "=" * 60)
        print("TO LAUNCH UI:")
        print("=" * 60)
        print("\n1. Make sure you have your UI file (the code you provided)")
        print("2. Add this at the end of your UI file:")
        print("\n--- Add to UI file ---")
        print("from ui_integration import integrate_with_ui, DigitalTwinPredictor")
        print("")
        print("# In your AssessmentPage.submit method:")
        print("predictor = DigitalTwinPredictor()")
        print("results = predictor.predict(self.app.user_data)")
        print("# Then display results")
        print("--- End ---")
        print("\n3. Run your UI file normally")
        
    except Exception as e:
        print(f"\n❌ Error launching UI: {e}")


def evaluate_mode():
    """Evaluate model performance"""
    from ensemble_model import CausalPrototypeNetwork
    from data_loader import DataLoader
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('models/ensemble_model.pkl'):
        print("\n❌ No trained model found!")
        print("Please run: python main.py --mode train")
        return
    
    # Load model
    print("\n📂 Loading model...")
    model = CausalPrototypeNetwork()
    model.load_models()
    
    # Load test data
    print("\n📂 Loading test data...")
    loader = DataLoader()
    loader.load_data()
    data = loader.preprocess_and_merge()
    
    feature_cols = [col for col in data.columns 
                   if col not in ['recommended_drug', 'drug_encoded', 'bp_reduction']]
    
    X = data[feature_cols].values
    y = data['drug_encoded'].values
    
    # Use last 20% as test set
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Evaluate
    print("\n📊 Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    # Save results
    import json
    results_path = 'results/evaluation_results.json'
    os.makedirs('results', exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()
