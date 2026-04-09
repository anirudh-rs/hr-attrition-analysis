import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess(data_path='data/master_hr_dataset.csv'):

    print("Loading master dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 1 — DROP USELESS COLUMNS
    # EmployeeNumber: just an ID, no predictive value
    # BurnRateBin: a binned version we created in EDA, not needed in model
    # ════════════════════════════════════════════════════════════════════
    drop_cols = ['EmployeeNumber']
    
    # BurnRateBin only exists if EDA notebook was run first
    if 'BurnRateBin' in df.columns:
        drop_cols.append('BurnRateBin')
    
    # Attrition_bin only exists if EDA notebook was run first    
    if 'Attrition_bin' in df.columns:
        drop_cols.append('Attrition_bin')

    df.drop(columns=drop_cols, inplace=True)
    print(f"\nAfter dropping useless columns: {df.shape}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2 — ENCODE CATEGORICAL COLUMNS
    # Machine learning models need numbers — no text allowed
    # We save each encoder so the dashboard can reverse them later
    # ════════════════════════════════════════════════════════════════════
    print("\nEncoding categorical columns...")

    # Target variable first — Yes/No → 1/0
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    print(f"  Attrition → binary (1=Left, 0=Stayed)")

    # OverTime — Yes/No → 1/0
    df['OverTime'] = (df['OverTime'] == 'Yes').astype(int)
    print(f"  OverTime → binary (1=Yes, 0=No)")

    # All remaining text columns → LabelEncoder
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"  Remaining categorical columns: {categorical_cols}")

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  {col} → {list(le.classes_)}")

    # Save encoders for dashboard use
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/label_encoders.pkl')
    print(f"\n  Encoders saved to models/label_encoders.pkl")

    # ════════════════════════════════════════════════════════════════════
    # STEP 3 — FEATURE ENGINEERING
    # Create new columns that combine existing signals in meaningful ways
    # Each one is grounded in real HR research
    # ════════════════════════════════════════════════════════════════════
    print("\nEngineering new features...")

    # 1. Are they underpaid for their job level?
    # Low salary relative to their level = frustration signal
    df['SalaryPerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    print("  ✅ SalaryPerLevel: MonthlyIncome / JobLevel")

    # 2. How long since their last promotion relative to tenure?
    # Stuck employees leave — high ratio means promotion is overdue
    df['PromotionStagnation'] = (
        df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    )
    print("  ✅ PromotionStagnation: YearsSinceLastPromotion / YearsAtCompany")

    # 3. Composite satisfaction score across all 4 satisfaction dimensions
    # One number summarizing overall employee happiness
    satisfaction_cols = [
        'JobSatisfaction', 'EnvironmentSatisfaction',
        'RelationshipSatisfaction', 'WorkLifeBalance'
    ]
    df['TotalSatisfaction'] = df[satisfaction_cols].mean(axis=1)
    print("  ✅ TotalSatisfaction: mean of all 4 satisfaction scores")

    # 4. Burnout pressure index — combines fatigue + resource stretch
    # High fatigue with low resources = burnout risk
    df['BurnoutPressureIndex'] = (
        df['MentalFatigueScore'] * (1 / (df['ResourceAllocation'] + 1))
    )
    print("  ✅ BurnoutPressureIndex: MentalFatigueScore / ResourceAllocation")

    # 5. Career velocity — how fast are they moving up relative to age?
    # Slow movers relative to peers tend to disengage
    df['CareerVelocity'] = df['JobLevel'] / (df['Age'] + 1)
    print("  ✅ CareerVelocity: JobLevel / Age")

    # 6. Manager stability — long tenure with same manager = stability signal
    df['ManagerStability'] = (
        df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    )
    print("  ✅ ManagerStability: YearsWithCurrManager / YearsAtCompany")

    print(f"\nAfter feature engineering: {df.shape}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 4 — SPLIT FEATURES AND TARGET
    # X = everything the model uses to predict
    # y = what we're predicting (Attrition)
    # ════════════════════════════════════════════════════════════════════
    print("\nSplitting features and target...")

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Attrition rate: {y.mean():.1%}")

    # Train/test split — stratified keeps class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain set: {X_train.shape} | Attrition rate: {y_train.mean():.1%}")
    print(f"Test set:  {X_test.shape} | Attrition rate: {y_test.mean():.1%}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 5 — SAVE EVERYTHING
    # ════════════════════════════════════════════════════════════════════
    print("\nSaving processed data...")

    joblib.dump(X_train, 'models/X_train.pkl')
    joblib.dump(X_test,  'models/X_test.pkl')
    joblib.dump(y_train, 'models/y_train.pkl')
    joblib.dump(y_test,  'models/y_test.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')

    # Save full processed dataframe for dashboard use
    df.to_csv('data/processed_hr_dataset.csv', index=False)

    print("  ✅ X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl")
    print("  ✅ feature_names.pkl")
    print("  ✅ processed_hr_dataset.csv")

    print(f"\n{'='*50}")
    print(f"PREPROCESSING COMPLETE")
    print(f"Total features going into model: {X.shape[1]}")
    print(f"Feature list:\n{list(X.columns)}")

    return X_train, X_test, y_train, y_test, X, y

if __name__ == '__main__':
    preprocess()