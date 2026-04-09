import pandas as pd
import numpy as np
import os

# ── Load all three datasets ───────────────────────────────────────────────────
print("Loading datasets...")
ibm = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
hrv14 = pd.read_csv('data/HRDataset_v14.csv')
burnout = pd.read_csv('data/train.csv')

print(f"IBM: {ibm.shape}")
print(f"HRv14: {hrv14.shape}")
print(f"Burnout: {burnout.shape}")

# ════════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — ENRICH FROM HRv14
# Strategy: compute median values of the 4 unique columns grouped by
# Department and Gender, then map those medians onto IBM employees
# with matching Department and Gender
# ════════════════════════════════════════════════════════════════════════════════
print("\nEnriching from HRv14...")

# Standardize Gender so both datasets use the same labels
hrv14['Gender_std'] = hrv14['Sex'].str.strip().str.upper().map(
    {'M': 'Male', 'F': 'Female', 'MALE': 'Male', 'FEMALE': 'Female'}
)

# Standardize Department names to match IBM's 3 departments
dept_map = {
    'IT/IS': 'Research & Development',
    'Software Engineering': 'Research & Development',
    'Production': 'Research & Development',
    'Sales': 'Sales',
    'Admin Offices': 'Human Resources',
    'Executive Office': 'Human Resources'
}
hrv14['Dept_std'] = hrv14['Department'].map(dept_map).fillna('Research & Development')

# Compute median stats per Department + Gender group
hrv14_stats = hrv14.groupby(['Dept_std', 'Gender_std']).agg(
    EngagementSurvey=('EngagementSurvey', 'median'),
    EmpSatisfaction=('EmpSatisfaction', 'median'),
    DaysLateLast30=('DaysLateLast30', 'median'),
    Absences=('Absences', 'median')
).reset_index()

hrv14_stats.columns = [
    'Department', 'Gender',
    'EngagementSurvey', 'EmpSatisfaction',
    'DaysLateLast30', 'Absences'
]

# Merge onto IBM using Department + Gender as the join key
master = ibm.merge(hrv14_stats, on=['Department', 'Gender'], how='left')

# Fill any unmatched rows with overall medians
for col in ['EngagementSurvey', 'EmpSatisfaction', 'DaysLateLast30', 'Absences']:
    master[col] = master[col].fillna(master[col].median())

print(f"After HRv14 enrichment: {master.shape}")
print(f"New columns added: EngagementSurvey, EmpSatisfaction, DaysLateLast30, Absences")

# ════════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — ENRICH FROM BURNOUT
# Strategy: map Designation (1-5 seniority scale) onto IBM's JobLevel (1-5)
# then compute median burnout stats per JobLevel + Gender group
# ════════════════════════════════════════════════════════════════════════════════
print("\nEnriching from Burnout dataset...")

# Standardize Gender
burnout['Gender_std'] = burnout['Gender'].str.strip().str.title()

# Designation in burnout is already 1-5 scale matching IBM's JobLevel 1-5
burnout['JobLevel'] = burnout['Designation'].round().astype(int).clip(1, 5)

# Standardize WFH to binary
burnout['WFH_std'] = burnout['WFH Setup Available'].map({'Yes': 1, 'No': 0})

# Compute median stats per JobLevel + Gender group
burnout_stats = burnout.groupby(['JobLevel', 'Gender_std']).agg(
    BurnRate=('Burn Rate', 'median'),
    MentalFatigueScore=('Mental Fatigue Score', 'median'),
    WFHAvailable=('WFH_std', 'median'),
    ResourceAllocation=('Resource Allocation', 'median')
).reset_index()

burnout_stats.columns = [
    'JobLevel', 'Gender',
    'BurnRate', 'MentalFatigueScore',
    'WFHAvailable', 'ResourceAllocation'
]

# Merge onto master using JobLevel + Gender as the join key
master = master.merge(burnout_stats, on=['JobLevel', 'Gender'], how='left')

# Fill any unmatched rows with overall medians
for col in ['BurnRate', 'MentalFatigueScore', 'WFHAvailable', 'ResourceAllocation']:
    master[col] = master[col].fillna(master[col].median())

print(f"After Burnout enrichment: {master.shape}")
print(f"New columns added: BurnRate, MentalFatigueScore, WFHAvailable, ResourceAllocation")

# ════════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — CLEAN UP
# Drop columns with zero variance that IBM dataset is known to contain
# ════════════════════════════════════════════════════════════════════════════════
print("\nCleaning up...")

drop_cols = ['EmployeeCount', 'Over18', 'StandardHours']
master.drop(columns=drop_cols, inplace=True)

# Verify no nulls remain
null_count = master.isnull().sum().sum()
print(f"Null values remaining: {null_count}")

# ════════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — SAVE
# ════════════════════════════════════════════════════════════════════════════════
os.makedirs('data', exist_ok=True)
master.to_csv('data/master_hr_dataset.csv', index=False)

print(f"\n✅ Master dataset saved to data/master_hr_dataset.csv")
print(f"Final shape: {master.shape}")
print(f"\nFinal columns ({len(master.columns)}):")
print(list(master.columns))