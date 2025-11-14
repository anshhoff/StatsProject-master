import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fall back gracefully if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

print("="*70)
print("STUDENT PLACEMENT PREDICTION - MULTI-MODEL COMPARISON")
print("="*70)

# Load and prepare data
df = pd.read_csv('college_student_placement_dataset.csv')
print(f"\nDataset shape: {df.shape}")
print("\nPlacement Distribution:")
print(df['Placement'].value_counts())

# Data preprocessing
df = df.drop('College_ID', axis=1)
df['Internship_Experience'] = (df['Internship_Experience'] == 'Yes').astype(int)

# Select features
X = df[['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 
        'Internship_Experience', 'Extra_Curricular_Score', 
        'Communication_Skills', 'Projects_Completed']].copy()

# Feature Engineering
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Interaction features
X['CGPA_x_Communication'] = X['CGPA'] * X['Communication_Skills']
X['IQ_x_Academic_Performance'] = X['IQ'] * X['Academic_Performance']
X['Projects_x_Internship'] = X['Projects_Completed'] * X['Internship_Experience']

# Squared features
X['CGPA_squared'] = X['CGPA'] ** 2
X['IQ_squared'] = X['IQ'] ** 2
X['Academic_Performance_squared'] = X['Academic_Performance'] ** 2
X['Communication_Skills_squared'] = X['Communication_Skills'] ** 2

print(f"Total features: {len(X.columns)}")

# Convert target variable
y = (df['Placement'] == 'Yes').astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize models
models = {}

print("\n" + "="*70)
print("TRAINING MULTIPLE MODELS")
print("="*70)

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

models['logistic_regression'] = {
    'model': lr_model,
    'name': 'Logistic Regression',
    'predictions': lr_pred,
    'probabilities': lr_proba,
    'accuracy': accuracy_score(y_test, lr_pred),
    'auc': roc_auc_score(y_test, lr_proba),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'coefficients': lr_model.coef_[0].tolist(),
    'intercept': float(lr_model.intercept_[0]),
    'feature_importance': np.abs(lr_model.coef_[0])
}

# 2. Random Forest
print("2. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

models['random_forest'] = {
    'model': rf_model,
    'name': 'Random Forest',
    'predictions': rf_pred,
    'probabilities': rf_proba,
    'accuracy': accuracy_score(y_test, rf_pred),
    'auc': roc_auc_score(y_test, rf_proba),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'feature_importance': rf_model.feature_importances_
}

# 3. XGBoost (if available)
if XGBOOST_AVAILABLE:
    print("3. Training XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    models['xgboost'] = {
        'model': xgb_model,
        'name': 'XGBoost',
        'predictions': xgb_pred,
        'probabilities': xgb_proba,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'auc': roc_auc_score(y_test, xgb_proba),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'feature_importance': xgb_model.feature_importances_
    }
else:
    print("3. XGBoost skipped (not installed)")

# Print comparison results
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<8} {'Precision':<12} {'Recall':<8}")
print("-" * 70)

for key, model_info in models.items():
    print(f"{model_info['name']:<20} {model_info['accuracy']:<10.4f} {model_info['auc']:<8.4f} "
          f"{model_info['precision']:<12.4f} {model_info['recall']:<8.4f}")

# Export all models to JSON
print("\n" + "="*50)
print("EXPORTING MODELS")
print("="*50)

model_export = {
    'scaler': {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': list(X.columns)
    },
    'models': {}
}

# Export each model's relevant information
for key, model_info in models.items():
    model_data = {
        'name': model_info['name'],
        'accuracy': model_info['accuracy'],
        'auc': model_info['auc'],
        'precision': model_info['precision'],
        'recall': model_info['recall'],
        'feature_importance': model_info['feature_importance'].tolist()
    }
    
    # Add model-specific parameters
    if key == 'logistic_regression':
        model_data['coefficients'] = model_info['coefficients']
        model_data['intercept'] = model_info['intercept']
        model_data['classes'] = model_info['model'].classes_.tolist()
    
    model_export['models'][key] = model_data

# Save to JSON
with open('multi_models.json', 'w') as f:
    json.dump(model_export, f, indent=2)

print(f"\n✅ All models exported to 'multi_models.json'")
print(f"   - Models trained: {list(models.keys())}")
print(f"   - Features: {len(X.columns)}")

# Find best model
best_model_key = max(models.keys(), key=lambda k: models[k]['accuracy'])
best_model = models[best_model_key]
print(f"\n🏆 Best performing model: {best_model['name']} (Accuracy: {best_model['accuracy']:.4f})")

print("\n" + "="*70)
print("TRAINING COMPLETED")
print("="*70)