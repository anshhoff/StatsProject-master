import pickle
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class PlacementPredictor:
    def __init__(self):
        self.models = None
        self.scaler = None
        self.feature_names = None
        self.model_info = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and configuration"""
        try:
            # Load trained models
            with open('trained_models.pkl', 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
            
            # Load model configuration
            with open('multi_models.json', 'r') as f:
                self.model_info = json.load(f)
            
            print("✅ Models loaded successfully!")
            print(f"Available models: {list(self.models.keys())}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Please run multi_model_comparison.py first to train the models.")
    
    def predict_single(self, student_data, model_name='logistic_regression'):
        """
        Predict placement probability for a single student
        
        Args:
            student_data: dict with keys matching feature_names
            model_name: which model to use for prediction
        
        Returns:
            dict with prediction results
        """
        if self.models is None:
            return {"error": "Models not loaded"}
        
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found. Available: {list(self.models.keys())}"}
        
        try:
            # Create feature array in correct order
            features = []
            for feature in self.feature_names:
                if feature in student_data:
                    features.append(student_data[feature])
                else:
                    return {"error": f"Missing feature: {feature}"}
            
            # Scale features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Get model
            model = self.models[model_name]
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            return {
                "model_used": model_name,
                "prediction": "Placed" if prediction == 1 else "Not Placed",
                "probability_not_placed": float(probability[0]),
                "probability_placed": float(probability[1]),
                "confidence": float(max(probability)),
                "input_features": student_data
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_all_models(self, student_data):
        """
        Get predictions from all available models
        """
        results = {}
        
        for model_name in self.models.keys():
            result = self.predict_single(student_data, model_name)
            if "error" not in result:
                results[model_name] = {
                    "prediction": result["prediction"],
                    "placed_probability": result["probability_placed"],
                    "confidence": result["confidence"]
                }
            else:
                results[model_name] = {"error": result["error"]}
        
        return results
    
    def get_model_performance(self):
        """Get performance metrics for all models"""
        if self.model_info is None:
            return {"error": "Model info not loaded"}
        
        performance = {}
        for model_name, info in self.model_info['models'].items():
            performance[model_name] = {
                "accuracy": info["accuracy"],
                "precision": info["precision"],
                "recall": info["recall"],
                "auc": info["auc"],
                "cv_mean": info["cv_mean"],
                "cv_std": info["cv_std"]
            }
        
        return performance

def main():
    """Example usage"""
    predictor = PlacementPredictor()
    
    if predictor.models is None:
        return
    
    # Example student data
    example_student = {
        'IQ': 110,
        'Prev_Sem_Result': 7.5,
        'CGPA': 8.2,
        'Academic_Performance': 8,
        'Internship_Experience': 1,  # 1 for Yes, 0 for No
        'Extra_Curricular_Score': 6,
        'Communication_Skills': 7,
        'Projects_Completed': 3
    }
    
    print("\n" + "="*60)
    print("PLACEMENT PREDICTION EXAMPLE")
    print("="*60)
    print("Student Profile:")
    for key, value in example_student.items():
        print(f"  {key}: {value}")
    
    # Get predictions from all models
    print("\n" + "="*60)
    print("PREDICTIONS FROM ALL MODELS")
    print("="*60)
    
    all_predictions = predictor.predict_all_models(example_student)
    
    for model_name, result in all_predictions.items():
        if "error" not in result:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Placement Probability: {result['placed_probability']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"\n{model_name}: {result['error']}")
    
    # Show model performance
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    performance = predictor.get_model_performance()
    if "error" not in performance:
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<8} {'AUC':<8}")
        print("-" * 68)
        for model_name, metrics in performance.items():
            print(f"{model_name.replace('_', ' ').title():<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['auc']:<8.4f}")

if __name__ == "__main__":
    main()