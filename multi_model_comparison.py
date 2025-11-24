import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
import json
import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
print("Updated with 15% test split and confusion matrices")
print("="*70)

# Load and prepare data
df = pd.read_csv('college_student_placement_dataset.csv')
print(f"\nDataset shape: {df.shape}")
print("\nPlacement Distribution:")
placement_counts = df['Placement'].value_counts()
print(placement_counts)
placement_rate = (placement_counts['Yes'] / len(df)) * 100
print(f"Placement Rate: {placement_rate:.2f}%")

# Data preprocessing
df = df.drop('College_ID', axis=1)
df['Internship_Experience'] = (df['Internship_Experience'] == 'Yes').astype(int)

# Select core features only to reduce overfitting
X = df[['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 
        'Internship_Experience', 'Extra_Curricular_Score', 
        'Communication_Skills', 'Projects_Completed']].copy()

print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)
print(f"Total features: {len(X.columns)}")
print("Features used:", list(X.columns))

# Convert target variable
y = (df['Placement'] == 'Yes').astype(int)

# Split the data with 15% test size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]} ({((1-0.15)*100):.0f}%)")
print(f"Testing set size: {X_test.shape[0]} ({(0.15*100):.0f}%)")
print(f"Training placement rate: {(y_train.sum() / len(y_train) * 100):.2f}%")
print(f"Testing placement rate: {(y_test.sum() / len(y_test) * 100):.2f}%")

# Initialize models with regularization to prevent overfitting
models = {}

print("\n" + "="*70)
print("TRAINING MULTIPLE MODELS WITH ANTI-OVERFITTING MEASURES")
print("="*70)

# 1. Logistic Regression with L2 regularization
print("\n1. Training Logistic Regression with regularization...")
lr_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight='balanced',
    C=0.5,  # Increased regularization
    penalty='l2'
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Cross-validation to check for overfitting
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

models['logistic_regression'] = {
    'model': lr_model,
    'name': 'Logistic Regression',
    'predictions': lr_pred,
    'probabilities': lr_proba,
    'accuracy': accuracy_score(y_test, lr_pred),
    'auc': roc_auc_score(y_test, lr_proba),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'cv_mean': lr_cv_scores.mean(),
    'cv_std': lr_cv_scores.std(),
    'confusion_matrix': confusion_matrix(y_test, lr_pred),
    'coefficients': lr_model.coef_[0].tolist(),
    'intercept': float(lr_model.intercept_[0]),
    'feature_importance': np.abs(lr_model.coef_[0])
}

# 2. Random Forest with reduced complexity
print("2. Training Random Forest with anti-overfitting parameters...")
rf_model = RandomForestClassifier(
    n_estimators=50,  # Reduced from 100
    max_depth=8,      # Limited depth
    min_samples_split=20,  # Increased minimum samples
    min_samples_leaf=10,   # Increased minimum leaf samples
    random_state=42, 
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

models['random_forest'] = {
    'model': rf_model,
    'name': 'Random Forest',
    'predictions': rf_pred,
    'probabilities': rf_proba,
    'accuracy': accuracy_score(y_test, rf_pred),
    'auc': roc_auc_score(y_test, rf_proba),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'cv_mean': rf_cv_scores.mean(),
    'cv_std': rf_cv_scores.std(),
    'confusion_matrix': confusion_matrix(y_test, rf_pred),
    'feature_importance': rf_model.feature_importances_
}

# 3. XGBoost with regularization (if available)
if XGBOOST_AVAILABLE:
    print("3. Training XGBoost with regularization...")
    xgb_model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        max_depth=6,      # Limited depth
        learning_rate=0.1, # Slower learning
        subsample=0.8,    # Sample subset for each tree
        colsample_bytree=0.8,  # Feature subset for each tree
        reg_alpha=0.1,    # L1 regularization
        reg_lambda=0.1,   # L2 regularization
        n_estimators=50   # Reduced number of trees
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    models['xgboost'] = {
        'model': xgb_model,
        'name': 'XGBoost',
        'predictions': xgb_pred,
        'probabilities': xgb_proba,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'auc': roc_auc_score(y_test, xgb_proba),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'cv_mean': xgb_cv_scores.mean(),
        'cv_std': xgb_cv_scores.std(),
        'confusion_matrix': confusion_matrix(y_test, xgb_pred),
        'feature_importance': xgb_model.feature_importances_
    }
else:
    print("3. XGBoost skipped (not installed)")

# Print comparison results with overfitting detection
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Model':<20} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<8} {'Overfitting':<12}")
print("-" * 70)

for key, model_info in models.items():
    test_acc = model_info['accuracy']
    cv_mean = model_info['cv_mean']
    cv_std = model_info['cv_std']
    overfitting = "Yes" if (cv_mean - test_acc) > 0.05 else "No"
    
    print(f"{model_info['name']:<20} {test_acc:<10.4f} {cv_mean:<10.4f} "
          f"{cv_std:<8.4f} {overfitting:<12}")

print("\n" + "="*70)
print("DETAILED METRICS")
print("="*70)
print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<8} {'Precision':<12} {'Recall':<8}")
print("-" * 70)

for key, model_info in models.items():
    print(f"{model_info['name']:<20} {model_info['accuracy']:<10.4f} {model_info['auc']:<8.4f} "
          f"{model_info['precision']:<12.4f} {model_info['recall']:<8.4f}")

# Print confusion matrices
print("\n" + "="*70)
print("CONFUSION MATRICES")
print("="*70)

for key, model_info in models.items():
    print(f"\n{model_info['name']}:")
    cm = model_info['confusion_matrix']
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(f"Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")

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
    
    # Add model-specific weights/parameters
    if key == 'logistic_regression':
        model_data['weights'] = {
            'coefficients': model_info['coefficients'],
            'intercept': model_info['intercept'],
            'classes': model_info['model'].classes_.tolist(),
            'regularization_C': float(model_info['model'].C)
        }
    elif key == 'random_forest':
        # Store tree-based model parameters
        model_data['weights'] = {
            'n_estimators': int(model_info['model'].n_estimators),
            'max_depth': model_info['model'].max_depth,
            'min_samples_split': int(model_info['model'].min_samples_split),
            'min_samples_leaf': int(model_info['model'].min_samples_leaf),
            'random_state': int(model_info['model'].random_state)
        }
    elif key == 'xgboost' and XGBOOST_AVAILABLE:
        model_data['weights'] = {
            'max_depth': int(model_info['model'].max_depth),
            'learning_rate': float(model_info['model'].learning_rate),
            'n_estimators': int(model_info['model'].n_estimators),
            'subsample': float(model_info['model'].subsample),
            'colsample_bytree': float(model_info['model'].colsample_bytree),
            'reg_alpha': float(model_info['model'].reg_alpha),
            'reg_lambda': float(model_info['model'].reg_lambda)
        }
    
    model_export['models'][key] = model_data

# Save to JSON
with open('multi_models.json', 'w') as f:
    json.dump(model_export, f, indent=2)

# Save actual model objects for prediction
model_objects = {}
for key, model_info in models.items():
    model_objects[key] = model_info['model']

# Save models using pickle for prediction
with open('trained_models.pkl', 'wb') as f:
    pickle.dump({
        'models': model_objects,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }, f)

print(f"\n✅ Model weights saved to 'multi_models.json'")
print(f"✅ Trained models saved to 'trained_models.pkl'")
print(f"   - Models trained: {list(models.keys())}")
print(f"   - Features: {len(X.columns)}")
print(f"   - Test split: 15%")

# Find best model
best_model_key = max(models.keys(), key=lambda k: models[k]['accuracy'])
best_model = models[best_model_key]
print(f"\n🏆 Best performing model: {best_model['name']} (Accuracy: {best_model['accuracy']:.4f})")

print("\n" + "="*70)
print("TRAINING COMPLETED")
print("="*70)