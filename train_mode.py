"""
Machine Learning Classification - Credit Card Fraud Detection
Trains 5 classification models on the Credit Card Fraud Detection dataset
Target: Detect fraudulent credit card transactions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================
print("="*70)
print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
print("="*70)

print("\n" + "="*70)
print("LOADING DATASET")
print("="*70)

# Load credit card fraud dataset
df = pd.read_csv('credit_card_fraud.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum().sum())

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("\n" + "="*70)
print("DATA PREPROCESSING & EXPLORATION")
print("="*70)

# Check for missing values
if df.isnull().sum().sum() == 0:
    print("\n✓ No missing values found!")
else:
    print(f"\nRemoving {df.isnull().sum().sum()} rows with missing values...")
    df = df.dropna()

# Display class distribution
print(f"\n📊 TARGET VARIABLE DISTRIBUTION:")
print(df['Class'].value_counts())
class_dist = df['Class'].value_counts()
fraud_ratio = (class_dist[1] / len(df)) * 100
print(f"\nFraud Cases: {class_dist[1]} ({fraud_ratio:.2f}%)")
print(f"Legitimate Cases: {class_dist[0]} ({100-fraud_ratio:.2f}%)")
print("\n⚠️  HIGHLY IMBALANCED DATASET - Fraud cases are only {:.2f}%".format(fraud_ratio))

# Separate features and target
# Target: Class (0 = Legitimate, 1 = Fraud)
X = df.drop(['Class', 'Time'], axis=1)  # Drop Time as it's not useful for prediction
y = df['Class']

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Feature names: {list(X.columns[:5])}... (28 total)")
print(f"✓ Target variable: Class (0=Legitimate, 1=Fraud)")

# ============================================
# STEP 3: SCALE FEATURES
# ============================================
print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)

# Most features (V1-V28) are already scaled by PCA
# But let's scale Amount separately and ensure consistency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n✓ Applied StandardScaler to all features")
print(f"✓ Feature means (should be ~0): {X_scaled.mean().mean():.6f}")
print(f"✓ Feature stds (should be ~1): {X_scaled.std().mean():.6f}")

# ============================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================
print("\n" + "="*70)
print("TRAIN-TEST SPLIT (with Stratification)")
print("="*70)

# Use stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution in both sets
)

print(f"\n📊 Training Set:")
print(f"   Size: {X_train.shape[0]} samples")
print(f"   Fraud: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
print(f"   Legitimate: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")

print(f"\n📊 Test Set:")
print(f"   Size: {X_test.shape[0]} samples")
print(f"   Fraud: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")
print(f"   Legitimate: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")

# ============================================
# STEP 5: DEFINE EVALUATION METRICS FUNCTION
# ============================================
def evaluate_model(model_name, y_true, y_pred, y_pred_proba=None):
    """
    Calculate all 6 evaluation metrics for binary classification

    IMPORTANT FOR FRAUD DETECTION:
    - Recall is crucial (catch as many frauds as possible)
    - Precision matters (minimize false alarms)
    - F1 Score balances both
    - MCC good for imbalanced datasets
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC for binary classification
    try:
        if y_pred_proba is not None:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            auc = 0.0
    except Exception as e:
        auc = 0.0

    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'MCC': round(mcc, 4)
    }

    return metrics

# ============================================
# STEP 6: TRAIN ALL MODELS
# ============================================
print("\n" + "="*70)
print("TRAINING ALL CLASSIFICATION MODELS")
print("="*70)
print("\n⏱️  This may take 2-3 minutes for all models...\n")

results = []
models_dict = {}

# 1. LOGISTIC REGRESSION
print("[1/5] Training Logistic Regression...")
print("      (Linear model, good baseline for fraud detection)")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_pred_proba = lr_model.predict_proba(X_test)
lr_metrics = evaluate_model('Logistic Regression', y_test, lr_pred, lr_pred_proba)
results.append(lr_metrics)
models_dict['Logistic Regression'] = lr_model
print(f"      ✓ Accuracy: {lr_metrics['Accuracy']:.4f} | Recall: {lr_metrics['Recall']:.4f} | F1: {lr_metrics['F1 Score']:.4f}\n")

# 2. DECISION TREE CLASSIFIER
print("[2/5] Training Decision Tree Classifier...")
print("      (Tree-based, handles non-linear patterns)")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_pred_proba = dt_model.predict_proba(X_test)
dt_metrics = evaluate_model('Decision Tree', y_test, dt_pred, dt_pred_proba)
results.append(dt_metrics)
models_dict['Decision Tree'] = dt_model
print(f"      ✓ Accuracy: {dt_metrics['Accuracy']:.4f} | Recall: {dt_metrics['Recall']:.4f} | F1: {dt_metrics['F1 Score']:.4f}\n")

# 3. K-NEAREST NEIGHBORS
print("[3/5] Training K-Nearest Neighbors...")
print("      (Instance-based, finds similar transactions)")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_pred_proba = knn_model.predict_proba(X_test)
knn_metrics = evaluate_model('KNN', y_test, knn_pred, knn_pred_proba)
results.append(knn_metrics)
models_dict['KNN'] = knn_model
print(f"      ✓ Accuracy: {knn_metrics['Accuracy']:.4f} | Recall: {knn_metrics['Recall']:.4f} | F1: {knn_metrics['F1 Score']:.4f}\n")

# 4. NAIVE BAYES (GAUSSIAN)
print("[4/5] Training Naive Bayes Classifier...")
print("      (Probabilistic, assumes feature independence)")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_pred_proba = nb_model.predict_proba(X_test)
nb_metrics = evaluate_model('Naive Bayes', y_test, nb_pred, nb_pred_proba)
results.append(nb_metrics)
models_dict['Naive Bayes'] = nb_model
print(f"      ✓ Accuracy: {nb_metrics['Accuracy']:.4f} | Recall: {nb_metrics['Recall']:.4f} | F1: {nb_metrics['F1 Score']:.4f}\n")

# 5. RANDOM FOREST (ENSEMBLE)
print("[5/5] Training Random Forest Classifier...")
print("      (Ensemble model, often best for fraud detection)")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)
rf_metrics = evaluate_model('Random Forest', y_test, rf_pred, rf_pred_proba)
results.append(rf_metrics)
models_dict['Random Forest'] = rf_model
print(f"      ✓ Accuracy: {rf_metrics['Accuracy']:.4f} | Recall: {rf_metrics['Recall']:.4f} | F1: {rf_metrics['F1 Score']:.4f}\n")

# ============================================
# STEP 7: CREATE COMPARISON TABLE
# ============================================
print("=" * 70)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 70)

results_df = pd.DataFrame(results)
print("\n")
print(results_df.to_string(index=False))

# Find best model by different criteria
best_acc_idx = results_df['Accuracy'].idxmax()
best_f1_idx = results_df['F1 Score'].idxmax()
best_recall_idx = results_df['Recall'].idxmax()

print("\n" + "=" * 70)
print("🏆 MODEL RANKINGS")
print("=" * 70)

print(f"\n1️⃣  Highest Accuracy: {results_df.loc[best_acc_idx, 'Model']}")
print(f"    Score: {results_df.loc[best_acc_idx, 'Accuracy']:.4f}")

print(f"\n2️⃣  Highest F1 Score: {results_df.loc[best_f1_idx, 'Model']}")
print(f"    Score: {results_df.loc[best_f1_idx, 'F1 Score']:.4f}")

print(f"\n3️⃣  Highest Recall (catch fraud): {results_df.loc[best_recall_idx, 'Model']}")
print(f"    Score: {results_df.loc[best_recall_idx, 'Recall']:.4f}")

best_model_name = results_df.loc[best_acc_idx, 'Model']
print(f"\n🎯 RECOMMENDED MODEL: {best_model_name}")
print(f"   (Best overall accuracy: {results_df.loc[best_acc_idx, 'Accuracy']:.4f})")

# ============================================
# STEP 8: DETAILED ANALYSIS OF BEST MODEL
# ============================================
print("\n" + "=" * 70)
print(f"DETAILED ANALYSIS - {best_model_name}")
print("=" * 70)

best_model = models_dict[best_model_name]
best_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
print(f"\nConfusion Matrix:")
print(f"                 Predicted Legitimate  Predicted Fraud")
print(f"Actual Legitimate       {cm[0, 0]:6d}              {cm[0, 1]:6d}")
print(f"Actual Fraud            {cm[1, 0]:6d}              {cm[1, 1]:6d}")

# Calculate specific metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

print(f"\n📊 Key Metrics for Fraud Detection:")
print(f"   True Negatives (TN):  {tn} - Legitimate correctly identified")
print(f"   False Positives (FP): {fp} - Legitimate incorrectly flagged as fraud")
print(f"   False Negatives (FN): {fn} - Fraud missed (CRITICAL!)")
print(f"   True Positives (TP):  {tp} - Fraud correctly identified")
print(f"\n   Specificity (catch legitimate): {specificity:.4f}")
print(f"   Sensitivity/Recall (catch fraud): {sensitivity:.4f}")

# Classification Report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, best_pred, target_names=['Legitimate', 'Fraud']))

# ============================================
# STEP 9: SAVE MODELS
# ============================================
print("\n" + "="*70)
print("SAVING TRAINED MODELS")
print("="*70)

if not os.path.exists('models'):
    os.makedirs('models')
    print("\n✓ Created 'models' directory")

model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl'
}

print("\nSaving models:")
for model_name, filename in model_files.items():
    filepath = f'models/{filename}'
    with open(filepath, 'wb') as f:
        pickle.dump(models_dict[model_name], f)
    print(f"  ✓ {model_name:20s} → {filename}")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ {'Scaler':20s} → scaler.pkl")

# Save results to CSV
results_df.to_csv('model_results.csv', index=False)
print(f"  ✓ Saved model results to 'model_results.csv'")

# ============================================
# COMPLETION MESSAGE
# ============================================
print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)

print("\n📁 Generated Files:")
print("  ✓ models/logistic_regression.pkl")
print("  ✓ models/decision_tree.pkl")
print("  ✓ models/knn.pkl")
print("  ✓ models/naive_bayes.pkl")
print("  ✓ models/random_forest.pkl")
print("  ✓ models/scaler.pkl")
print("  ✓ model_results.csv")

print("\n🎯 Dataset: Credit Card Fraud Detection")
print(f"   Total Transactions: {len(df):,}")
print(f"   Fraudulent Transactions: {(y == 1).sum():,}")
print(f"   Fraud Percentage: {fraud_ratio:.2f}%")

print("\n🚀 Next Step: Run 'streamlit run app.py' to start the web app!")
print("="*70)
