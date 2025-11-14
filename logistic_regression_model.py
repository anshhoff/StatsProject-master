import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("="*70)
print("STUDENT PLACEMENT PREDICTION - LOGISTIC REGRESSION")
print("="*70)

df = pd.read_csv('college_student_placement_dataset.csv')

print("\nDataset shape:", df.shape)
print("\nPlacement Distribution:")
print(df['Placement'].value_counts())

# Drop College_ID as it's not useful for prediction
df = df.drop('College_ID', axis=1)

# Convert Internship_Experience from Yes/No to binary 1/0
df['Internship_Experience'] = (df['Internship_Experience'] == 'Yes').astype(int)

# Select features (all columns except Placement)
X = df[['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 
        'Internship_Experience', 'Extra_Curricular_Score', 
        'Communication_Skills', 'Projects_Completed']].copy()

# FEATURE ENGINEERING
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# 1. Interaction features
X['CGPA_x_Communication'] = X['CGPA'] * X['Communication_Skills']
X['IQ_x_Academic_Performance'] = X['IQ'] * X['Academic_Performance']
X['Projects_x_Internship'] = X['Projects_Completed'] * X['Internship_Experience']

print("\nInteraction features created:")
print("  - CGPA × Communication_Skills")
print("  - IQ × Academic_Performance")
print("  - Projects_Completed × Internship_Experience")

# 2. Squared features
X['CGPA_squared'] = X['CGPA'] ** 2
X['IQ_squared'] = X['IQ'] ** 2
X['Academic_Performance_squared'] = X['Academic_Performance'] ** 2
X['Communication_Skills_squared'] = X['Communication_Skills'] ** 2

print("\nSquared features created:")
print("  - CGPA²")
print("  - IQ²")
print("  - Academic_Performance²")
print("  - Communication_Skills²")

# Convert target variable Placement from Yes/No to binary 1/0
y_placement = (df['Placement'] == 'Yes').astype(int)

print("\n" + "="*70)
print(f"Total features (original + engineered): {len(X.columns)}")
print("="*70)
print("\nAll Features:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_placement, test_size=0.2, random_state=42, stratify=y_placement
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
classifier.fit(X_train_scaled, y_train)

print("\nModel trained!")

# Feature Importance Analysis
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

# Get feature importance (absolute values of coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': classifier.coef_[0],
    'Abs_Coefficient': np.abs(classifier.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 15 Most Important Features (by absolute coefficient value):")
print("-" * 70)
for idx, row in feature_importance.head(15).iterrows():
    sign = "+" if row['Coefficient'] > 0 else "-"
    print(f"{row['Feature']:35s} {sign} {row['Abs_Coefficient']:8.4f}  (coef: {row['Coefficient']:8.4f})")
print("-" * 70)

y_test_pred = classifier.predict(X_test_scaled)
y_test_proba = classifier.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"ROC-AUC Score: {test_auc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Not Placed', 'Placed']))

cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Export model with all features including engineered ones
print("\n" + "="*70)
print("MODEL EXPORT")
print("="*70)

model_export = {
    'scaler': {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': list(X.columns)
    },
    'logistic_regression': {
        'coefficients': classifier.coef_[0].tolist(),
        'intercept': float(classifier.intercept_[0]),
        'classes': classifier.classes_.tolist()
    }
}

with open('models.json', 'w') as f:
    json.dump(model_export, f, indent=2)

print(f"\n✅ Model exported to 'models.json'")
print(f"   - Total features: {len(X.columns)}")
print(f"   - Original features: 8")
print(f"   - Engineered features: {len(X.columns) - 8}")
print(f"   - Coefficients exported: {len(classifier.coef_[0])}")

# Visualization
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Create main 2x2 analysis plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Student Placement Prediction Analysis', fontsize=18, fontweight='bold', y=0.995)

# 1. Confusion Matrix Heatmap (Top-Left)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
            cbar_kws={'label': 'Count'},
            xticklabels=['Not Placed', 'Placed'], 
            yticklabels=['Not Placed', 'Placed'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
axes[0, 0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Actual', fontsize=12, fontweight='bold')

# 2. ROC Curve with AUC Score (Top-Right)
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0, 1].plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {test_auc:.4f})', color='darkblue')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
axes[0, 1].fill_between(fpr, tpr, alpha=0.2)
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
axes[0, 1].set_title(f'ROC Curve (AUC = {test_auc:.4f})', fontsize=14, fontweight='bold', pad=10)
axes[0, 1].legend(loc='lower right', fontsize=10)
axes[0, 1].grid(alpha=0.3)

# 3. Feature Importance Bar Chart (Bottom-Left)
top_n = 12  # Show top 12 features
top_features = feature_importance.head(top_n).sort_values('Coefficient')
colors_importance = ['#d62728' if c < 0 else '#2ca02c' for c in top_features['Coefficient']]
axes[1, 0].barh(range(len(top_features)), top_features['Coefficient'], 
                color=colors_importance, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['Feature'], fontsize=9)
axes[1, 0].set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'Top {top_n} Feature Importance (Sorted by Coefficient)', fontsize=14, fontweight='bold', pad=10)
axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
axes[1, 0].grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (idx, row) in enumerate(top_features.iterrows()):
    value = row['Coefficient']
    x_pos = value + (0.1 if value > 0 else -0.1)
    ha = 'left' if value > 0 else 'right'
    axes[1, 0].text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=8, fontweight='bold')

# 4. Correlation Matrix Heatmap (Bottom-Right)
# Create correlation matrix including the target variable
X_with_target = X.copy()
X_with_target['Placement'] = y_placement
correlation_matrix = X_with_target.corr()

# Sort by correlation with Placement
placement_corr = correlation_matrix['Placement'].abs().sort_values(ascending=False)
top_corr_features = placement_corr.head(13).index  # Top 12 features + Placement

# Create subset correlation matrix
corr_subset = correlation_matrix.loc[top_corr_features, top_corr_features]

sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=axes[1, 1], cbar_kws={'label': 'Correlation'},
            square=True, linewidths=0.5, annot_kws={'size': 7})
axes[1, 1].set_title('Feature Correlation Matrix\n(Top Features by Placement Correlation)', 
                     fontsize=14, fontweight='bold', pad=10)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
axes[1, 1].set_yticklabels(axes[1, 1].get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('placement_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Main analysis plot saved: placement_analysis.png")

# Create individual feature importance plot (full size)
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
all_features_sorted = feature_importance.sort_values('Coefficient')
colors_all = ['#d62728' if c < 0 else '#2ca02c' for c in all_features_sorted['Coefficient']]
bars = ax2.barh(range(len(all_features_sorted)), all_features_sorted['Coefficient'], 
                color=colors_all, alpha=0.8, edgecolor='black', linewidth=0.8)
ax2.set_yticks(range(len(all_features_sorted)))
ax2.set_yticklabels(all_features_sorted['Feature'], fontsize=10)
ax2.set_xlabel('Coefficient Value', fontsize=13, fontweight='bold')
ax2.set_title('Complete Feature Importance Analysis\n(Logistic Regression Coefficients)', 
              fontsize=16, fontweight='bold', pad=15)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (idx, row) in enumerate(all_features_sorted.iterrows()):
    value = row['Coefficient']
    x_pos = value + (0.15 if value > 0 else -0.15)
    ha = 'left' if value > 0 else 'right'
    ax2.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance_detailed.png', dpi=300, bbox_inches='tight')
print("✅ Detailed feature importance plot saved: feature_importance_detailed.png")

# Create confusion matrix standalone
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            cbar_kws={'label': 'Count'},
            xticklabels=['Not Placed', 'Placed'], 
            yticklabels=['Not Placed', 'Placed'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax3.set_title('Confusion Matrix - Student Placement Prediction', fontsize=16, fontweight='bold', pad=15)
ax3.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix plot saved: confusion_matrix.png")

# Create ROC curve standalone
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
ax4.plot(fpr, tpr, linewidth=3, label=f'Logistic Regression (AUC = {test_auc:.4f})', color='darkblue')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
ax4.fill_between(fpr, tpr, alpha=0.2, color='blue')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax4.set_title(f'ROC Curve - Student Placement Prediction\nAUC = {test_auc:.4f}', 
              fontsize=16, fontweight='bold', pad=15)
ax4.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax4.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ ROC curve plot saved: roc_curve.png")

print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETED")
print("="*70)
print("\nGenerated files:")
print("  1. placement_analysis.png (2x2 grid with all 4 plots)")
print("  2. feature_importance_detailed.png (all features)")
print("  3. confusion_matrix.png (standalone)")
print("  4. roc_curve.png (standalone)")
print("="*70)
