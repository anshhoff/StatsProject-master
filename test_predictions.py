import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random

print("="*70)
print("MODEL VALIDATION & PREDICTION TESTING")
print("="*70)

# Load the dataset
df = pd.read_csv('college_student_placement_dataset.csv')
print(f"\nDataset loaded: {df.shape[0]} records")

# Preprocess the data (same as training script)
df = df.drop('College_ID', axis=1)
df['Internship_Experience'] = (df['Internship_Experience'] == 'Yes').astype(int)

# Prepare features (same order as training)
X = df[['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 
        'Internship_Experience', 'Extra_Curricular_Score', 
        'Communication_Skills', 'Projects_Completed']].copy()

# Feature engineering (must match training script exactly)
X['CGPA_x_Communication'] = X['CGPA'] * X['Communication_Skills']
X['IQ_x_Academic_Performance'] = X['IQ'] * X['Academic_Performance']
X['Projects_x_Internship'] = X['Projects_Completed'] * X['Internship_Experience']
X['CGPA_squared'] = X['CGPA'] ** 2
X['IQ_squared'] = X['IQ'] ** 2
X['Academic_Performance_squared'] = X['Academic_Performance'] ** 2
X['Communication_Skills_squared'] = X['Communication_Skills'] ** 2

# Target variable
y = (df['Placement'] == 'Yes').astype(int)

print(f"Total features (with engineering): {len(X.columns)}")
print(f"Feature names: {X.columns.tolist()}")

# Load the saved model
print("\n" + "="*70)
print("LOADING SAVED MODEL")
print("="*70)

with open('models.json', 'r') as f:
    model_data = json.load(f)

scaler_mean = np.array(model_data['scaler']['mean'])
scaler_scale = np.array(model_data['scaler']['scale'])
coefficients = np.array(model_data['logistic_regression']['coefficients'])
intercept = model_data['logistic_regression']['intercept']
feature_names = model_data['scaler']['feature_names']

print(f"\n✅ Model loaded from models.json")
print(f"   - Features: {len(feature_names)}")
print(f"   - Coefficients: {len(coefficients)}")
print(f"   - Intercept: {intercept:.4f}")

# Verify feature order matches
print(f"\n   Feature order validation:")
if feature_names == X.columns.tolist():
    print(f"   ✅ Feature order matches training script")
else:
    print(f"   ❌ WARNING: Feature order mismatch!")
    print(f"      Expected: {X.columns.tolist()}")
    print(f"      Got: {feature_names}")

# Function to simulate JavaScript prediction
def predict_like_javascript(features_dict):
    """
    Simulates the JavaScript prediction process to verify consistency
    """
    # Create feature array in the same order as JavaScript
    features = [
        features_dict['IQ'],
        features_dict['Prev_Sem_Result'],
        features_dict['CGPA'],
        features_dict['Academic_Performance'],
        features_dict['Internship_Experience'],
        features_dict['Extra_Curricular_Score'],
        features_dict['Communication_Skills'],
        features_dict['Projects_Completed'],
        # Engineered features (same order as JavaScript prepareFeatures)
        features_dict['CGPA'] * features_dict['Communication_Skills'],  # CGPA_x_Communication
        features_dict['IQ'] * features_dict['Academic_Performance'],  # IQ_x_Academic_Performance
        features_dict['Projects_Completed'] * features_dict['Internship_Experience'],  # Projects_x_Internship
        features_dict['CGPA'] ** 2,  # CGPA_squared
        features_dict['IQ'] ** 2,  # IQ_squared
        features_dict['Academic_Performance'] ** 2,  # Academic_Performance_squared
        features_dict['Communication_Skills'] ** 2  # Communication_Skills_squared
    ]
    
    # Standardize features (same as JavaScript)
    features_array = np.array(features)
    standardized = (features_array - scaler_mean) / scaler_scale
    
    # Calculate logit (same as JavaScript)
    logit = intercept + np.dot(standardized, coefficients)
    
    # Calculate probability (same as JavaScript)
    probability = 1 / (1 + np.exp(-logit))
    
    # Prediction (same as JavaScript threshold)
    prediction = 1 if probability >= 0.5 else 0
    
    return prediction, probability

# Function to make Python prediction using loaded model
def predict_python(features_dict):
    """
    Makes prediction using the loaded model parameters (simulating sklearn)
    """
    # Create feature array
    features = np.array([[
        features_dict['IQ'],
        features_dict['Prev_Sem_Result'],
        features_dict['CGPA'],
        features_dict['Academic_Performance'],
        features_dict['Internship_Experience'],
        features_dict['Extra_Curricular_Score'],
        features_dict['Communication_Skills'],
        features_dict['Projects_Completed'],
        features_dict['CGPA'] * features_dict['Communication_Skills'],
        features_dict['IQ'] * features_dict['Academic_Performance'],
        features_dict['Projects_Completed'] * features_dict['Internship_Experience'],
        features_dict['CGPA'] ** 2,
        features_dict['IQ'] ** 2,
        features_dict['Academic_Performance'] ** 2,
        features_dict['Communication_Skills'] ** 2
    ]])
    
    # Standardize
    standardized = (features - scaler_mean) / scaler_scale
    
    # Calculate probability
    logit = intercept + np.dot(standardized, coefficients)[0]
    probability = 1 / (1 + np.exp(-logit))
    prediction = 1 if probability >= 0.5 else 0
    
    return prediction, probability

# Test on 10 random samples
print("\n" + "="*70)
print("TESTING ON 10 RANDOM SAMPLES")
print("="*70)

random.seed(42)
test_indices = random.sample(range(len(df)), 10)
discrepancies = []

print("\nSample | Actual | Python Pred | JS Pred | Python Prob | JS Prob | Match")
print("-" * 85)

for idx, sample_idx in enumerate(test_indices, 1):
    # Get original features
    sample = df.iloc[sample_idx]
    
    features_dict = {
        'IQ': sample['IQ'],
        'Prev_Sem_Result': sample['Prev_Sem_Result'],
        'CGPA': sample['CGPA'],
        'Academic_Performance': sample['Academic_Performance'],
        'Internship_Experience': (sample['Internship_Experience'] if isinstance(sample['Internship_Experience'], int) 
                                 else 1 if sample['Internship_Experience'] == 'Yes' else 0),
        'Extra_Curricular_Score': sample['Extra_Curricular_Score'],
        'Communication_Skills': sample['Communication_Skills'],
        'Projects_Completed': sample['Projects_Completed']
    }
    
    actual = 1 if sample['Placement'] == 'Yes' else 0
    py_pred, py_prob = predict_python(features_dict)
    js_pred, js_prob = predict_like_javascript(features_dict)
    
    match = "✅" if (py_pred == js_pred and abs(py_prob - js_prob) < 0.0001) else "❌"
    
    if match == "❌":
        discrepancies.append({
            'sample': idx,
            'actual': actual,
            'py_pred': py_pred,
            'js_pred': js_pred,
            'py_prob': py_prob,
            'js_prob': js_prob
        })
    
    print(f"  {idx:2d}   |   {actual}    |      {py_pred}      |    {js_pred}    |   {py_prob:.4f}    |  {js_prob:.4f}  | {match}")

print("-" * 85)

if discrepancies:
    print(f"\n❌ Found {len(discrepancies)} discrepancies!")
    print("\nDiscrepancy Details:")
    for d in discrepancies:
        print(f"  Sample {d['sample']}: Python={d['py_pred']} (prob={d['py_prob']:.4f}), "
              f"JS={d['js_pred']} (prob={d['js_prob']:.4f})")
else:
    print(f"\n✅ All predictions match! Python and JavaScript implementations are consistent.")

# Calculate accuracy on entire dataset
print("\n" + "="*70)
print("MODEL ACCURACY ON FULL DATASET")
print("="*70)

all_predictions = []
all_probabilities = []

for idx in range(len(df)):
    sample = df.iloc[idx]
    features_dict = {
        'IQ': sample['IQ'],
        'Prev_Sem_Result': sample['Prev_Sem_Result'],
        'CGPA': sample['CGPA'],
        'Academic_Performance': sample['Academic_Performance'],
        'Internship_Experience': (sample['Internship_Experience'] if isinstance(sample['Internship_Experience'], int) 
                                 else 1 if sample['Internship_Experience'] == 'Yes' else 0),
        'Extra_Curricular_Score': sample['Extra_Curricular_Score'],
        'Communication_Skills': sample['Communication_Skills'],
        'Projects_Completed': sample['Projects_Completed']
    }
    
    pred, prob = predict_python(features_dict)
    all_predictions.append(pred)
    all_probabilities.append(prob)

all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)

accuracy = accuracy_score(y, all_predictions)

print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y, all_predictions, target_names=['Not Placed', 'Placed']))

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nTotal samples: {len(df)}")
print(f"Actual placements: {y.sum()} ({y.sum()/len(df)*100:.2f}%)")
print(f"Predicted placements: {all_predictions.sum()} ({all_predictions.sum()/len(df)*100:.2f}%)")
print(f"\nCorrect predictions: {(all_predictions == y).sum()}")
print(f"Incorrect predictions: {(all_predictions != y).sum()}")

# Probability distribution
print(f"\nPrediction Probability Statistics:")
print(f"  Mean probability: {all_probabilities.mean():.4f}")
print(f"  Std deviation: {all_probabilities.std():.4f}")
print(f"  Min probability: {all_probabilities.min():.4f}")
print(f"  Max probability: {all_probabilities.max():.4f}")
print(f"  Median probability: {np.median(all_probabilities):.4f}")

# Confidence analysis
high_confidence = (all_probabilities > 0.8) | (all_probabilities < 0.2)
print(f"\nHigh confidence predictions (prob > 0.8 or < 0.2): {high_confidence.sum()} ({high_confidence.sum()/len(df)*100:.2f}%)")
print(f"Low confidence predictions (0.4 < prob < 0.6): {((all_probabilities >= 0.4) & (all_probabilities <= 0.6)).sum()}")

print("\n" + "="*70)
print("TEST COMPLETED SUCCESSFULLY")
print("="*70)
