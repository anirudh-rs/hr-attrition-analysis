import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def explain():

    print("Loading data...")
    X_train       = joblib.load('models/X_train.pkl')
    X_test        = joblib.load('models/X_test.pkl')
    y_train       = joblib.load('models/y_train.pkl')
    feature_names = joblib.load('models/feature_names.pkl')

    print("Retraining with explicit base_score for SHAP compatibility...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        base_score=0.5,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_bal, y_train_bal, verbose=False)
    joblib.dump(model, 'models/xgb_attrition.pkl')
    print("Model retrained and saved")

    X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    print(f"Full dataset for SHAP: {X_full.shape}")

    print("\nComputing SHAP values (30-60 seconds)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full)
    print(f"SHAP values shape: {shap_values.shape}")

    os.makedirs('models', exist_ok=True)

    print("\nGenerating global importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_full,
        plot_type='bar',
        show=False,
        max_display=20
    )
    plt.title('Top 20 Features by Global SHAP Importance',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('models/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved models/shap_importance.png")

    print("\nGenerating beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_full,
        show=False,
        max_display=20
    )
    plt.title('SHAP Beeswarm - Feature Impact Direction',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('models/shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved models/shap_beeswarm.png")

    print("\nGenerating individual employee explanations...")
    predictions = pd.read_csv('models/predictions.csv')
    top3_idx = predictions.nlargest(3, 'AttritionRisk').index.tolist()

    for rank, idx in enumerate(top3_idx, 1):
        risk_score = predictions.loc[idx, 'AttritionRisk']
        department = predictions.loc[idx, 'Department']

        plt.figure(figsize=(12, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X_full.iloc[idx],
                feature_names=feature_names
            ),
            max_display=12,
            show=False
        )
        plt.title(
            f'Employee #{rank} - Risk Score: {risk_score:.3f} | Dept: {department}',
            fontsize=13, fontweight='bold', pad=20
        )
        plt.tight_layout()
        plt.savefig(f'models/shap_employee_{rank}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved models/shap_employee_{rank}.png")

    print("\nSaving SHAP values...")
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv('models/shap_values.csv', index=False)
    print("Saved models/shap_values.csv")

    mean_abs_shap = pd.DataFrame({
        'Feature': feature_names,
        'MeanAbsSHAP': np.abs(shap_df).mean()
    }).sort_values('MeanAbsSHAP', ascending=False)

    print(f"\n{'='*50}")
    print("TOP 15 FEATURES DRIVING ATTRITION")
    print("="*50)
    print(mean_abs_shap.head(15).to_string(index=False))

    enriched = [
        'EngagementSurvey', 'EmpSatisfaction', 'DaysLateLast30',
        'Absences', 'BurnRate', 'MentalFatigueScore',
        'WFHAvailable', 'ResourceAllocation'
    ]
    engineered = [
        'SalaryPerLevel', 'PromotionStagnation', 'TotalSatisfaction',
        'BurnoutPressureIndex', 'CareerVelocity', 'ManagerStability'
    ]

    top15 = mean_abs_shap.head(15)['Feature'].tolist()
    enriched_in_top15   = [f for f in enriched if f in top15]
    engineered_in_top15 = [f for f in engineered if f in top15]

    print(f"\nEnriched columns in top 15:   {enriched_in_top15}")
    print(f"Engineered columns in top 15: {engineered_in_top15}")

    return shap_values, explainer, X_full, mean_abs_shap

if __name__ == '__main__':
    explain()