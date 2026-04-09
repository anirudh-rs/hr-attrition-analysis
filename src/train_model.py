import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import os

def train():

    # ════════════════════════════════════════════════════════════════════
    # STEP 1 — LOAD SAVED DATA
    # ════════════════════════════════════════════════════════════════════
    print("Loading preprocessed data...")
    X_train = joblib.load('models/X_train.pkl')
    X_test  = joblib.load('models/X_test.pkl')
    y_train = joblib.load('models/y_train.pkl')
    y_test  = joblib.load('models/y_test.pkl')
    feature_names = joblib.load('models/feature_names.pkl')

    print(f"Train: {X_train.shape} | Attrition rate: {y_train.mean():.1%}")
    print(f"Test:  {X_test.shape}  | Attrition rate: {y_test.mean():.1%}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2 — SMOTE
    # Without this the model sees 84% "stayed" and learns to just
    # predict "stayed" for everyone — useless for HR teams
    # SMOTE creates synthetic "left" examples to balance the classes
    # IMPORTANT: SMOTE is applied ONLY to training data, never test data
    # ════════════════════════════════════════════════════════════════════
    print("\nApplying SMOTE to balance training classes...")
    print(f"Before SMOTE: {dict(y_train.value_counts())}")

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE:  {dict(pd.Series(y_train_bal).value_counts())}")
    print(f"Balanced train size: {X_train_bal.shape}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 3 — TRAIN XGBOOST
    # XGBoost is a gradient boosting algorithm — it builds many small
    # decision trees sequentially, each one correcting the last one's
    # mistakes. It's the industry standard for tabular data like this.
    # ════════════════════════════════════════════════════════════════════
    print("\nTraining XGBoost classifier...")

    model = XGBClassifier(
        n_estimators=300,       # number of trees to build
        max_depth=5,            # how deep each tree can go
        learning_rate=0.05,     # how much each tree corrects the last
        subsample=0.8,          # use 80% of rows per tree (prevents overfitting)
        colsample_bytree=0.8,   # use 80% of features per tree
        min_child_weight=3,     # minimum samples needed to split a node
        gamma=0.1,              # minimum loss reduction to make a split
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )

    model.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print("Training complete ✅")

    # ════════════════════════════════════════════════════════════════════
    # STEP 4 — EVALUATE
    # We care most about:
    # - Recall: did we catch most employees who actually left?
    # - Precision: when we flag someone, are we usually right?
    # - ROC-AUC: overall ranking ability (0.5=random, 1.0=perfect)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("MODEL EVALUATION")
    print("="*55)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Stayed', 'Left']
    ))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # ── Confusion Matrix ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Stayed', 'Left'],
        ax=axes[0],
        colorbar=False,
        cmap='Blues'
    )
    axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')

    # ── ROC Curve ────────────────────────────────────────────────────
    RocCurveDisplay.from_predictions(
        y_test, y_prob,
        ax=axes[1],
        color='#e74c3c',
        name=f'XGBoost (AUC = {roc_auc:.3f})'
    )
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[1].legend()

    plt.suptitle('Model Performance', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/model_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Performance plot saved to models/model_performance.png")

    # ════════════════════════════════════════════════════════════════════
    # STEP 5 — SAVE MODEL + PREDICTIONS
    # ════════════════════════════════════════════════════════════════════
    print("\nSaving model and predictions...")

    # Save the trained model
    joblib.dump(model, 'models/xgb_attrition.pkl')
    print("✅ Model saved to models/xgb_attrition.pkl")

    # Generate risk scores for every employee in full dataset
    X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    risk_scores = model.predict_proba(X_full)[:, 1]

    # Load original data to restore readable column names for dashboard
    raw = pd.read_csv('data/master_hr_dataset.csv')

    predictions = raw.copy()
    predictions['AttritionRisk'] = risk_scores
    predictions['RiskLabel'] = pd.cut(
        risk_scores,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    predictions['ActualAttrition'] = y_full.values

    predictions.to_csv('models/predictions.csv', index=False)
    print("✅ Predictions saved to models/predictions.csv")

    # Summary
    print(f"\n{'='*55}")
    print("RISK DISTRIBUTION ACROSS ALL EMPLOYEES")
    print("="*55)
    print(predictions['RiskLabel'].value_counts().to_string())
    print(f"\nHigh risk employees: {(predictions['RiskLabel'] == 'High').sum()}")
    print(f"  → In Sales:  {((predictions['RiskLabel'] == 'High') & (predictions['Department'] == 'Sales')).sum()}")
    print(f"  → In R&D:    {((predictions['RiskLabel'] == 'High') & (predictions['Department'] == 'Research & Development')).sum()}")
    print(f"  → In HR:     {((predictions['RiskLabel'] == 'High') & (predictions['Department'] == 'Human Resources')).sum()}")

    return model, y_pred, y_prob, y_test

if __name__ == '__main__':
    train()