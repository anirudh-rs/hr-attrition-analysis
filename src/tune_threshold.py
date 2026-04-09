import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, f1_score
)
import joblib

def tune_threshold():

    print("Loading model and test data...")
    model         = joblib.load('models/xgb_attrition.pkl')
    X_test        = joblib.load('models/X_test.pkl')
    y_test        = joblib.load('models/y_test.pkl')

    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Find best threshold by F1 score for "Left" class ─────────────
    thresholds = np.arange(0.1, 0.7, 0.01)
    results = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1_left = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        f1_stayed = f1_score(y_test, y_pred_t, pos_label=0, zero_division=0)
        results.append({
            'Threshold': round(t, 2),
            'F1_Left': round(f1_left, 4),
            'F1_Stayed': round(f1_stayed, 4),
            'F1_Macro': round((f1_left + f1_stayed) / 2, 4)
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['F1_Left'].idxmax()]
    best_threshold = best_row['Threshold']

    print(f"\nBest threshold for catching 'Left' employees: {best_threshold}")
    print(f"F1 for Left at this threshold: {best_row['F1_Left']}")
    print(f"F1 Macro at this threshold:    {best_row['F1_Macro']}")

    # ── Show report at best threshold ────────────────────────────────
    y_pred_best = (y_prob >= best_threshold).astype(int)
    print(f"\nClassification Report at threshold={best_threshold}:")
    print(classification_report(
        y_test, y_pred_best,
        target_names=['Stayed', 'Left']
    ))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # ── Plot threshold vs F1 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df['Threshold'], results_df['F1_Left'],
                 color='#e74c3c', label='F1 - Left')
    axes[0].plot(results_df['Threshold'], results_df['F1_Stayed'],
                 color='#2ecc71', label='F1 - Stayed')
    axes[0].plot(results_df['Threshold'], results_df['F1_Macro'],
                 color='#3498db', label='F1 - Macro', linestyle='--')
    axes[0].axvline(best_threshold, color='black',
                    linestyle=':', label=f'Best = {best_threshold}')
    axes[0].set_title('Threshold vs F1 Score', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('F1 Score')
    axes[0].legend()

    # ── Precision-Recall curve ────────────────────────────────────────
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, color='#e74c3c', linewidth=2)
    axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Recall (% of Left employees caught)')
    axes[1].set_ylabel('Precision (% of flags that are correct)')
    axes[1].fill_between(recall, precision, alpha=0.1, color='#e74c3c')

    plt.suptitle('Threshold Tuning Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/threshold_tuning.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Save best threshold for dashboard use ────────────────────────
    joblib.dump(float(best_threshold), 'models/best_threshold.pkl')
    print(f"\n✅ Best threshold saved to models/best_threshold.pkl")
    print(f"✅ Threshold tuning plot saved to models/threshold_tuning.png")

    return best_threshold

if __name__ == '__main__':
    tune_threshold()