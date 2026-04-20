import pandas as pd

def get_department_alerts(predictions_path='models/predictions.csv'):
    df = pd.read_csv(predictions_path)

    high_risk = df[df['RiskLabel'] == 'High'].copy()

    dept_summary = high_risk.groupby('Department').agg(
        HighRiskCount=('EmployeeNumber', 'count'),
        AvgRiskScore=('AttritionRisk', 'mean'),
        AvgIncome=('MonthlyIncome', 'mean'),
        PctOvertime=('OverTime', lambda x: (
            x.map({'Yes': 1, 'No': 0}).astype(float).mean()
        ))
    ).reset_index().sort_values('HighRiskCount', ascending=False)

    dept_summary['AvgRiskScore'] = dept_summary['AvgRiskScore'].round(3)
    dept_summary['AvgIncome']    = dept_summary['AvgIncome'].round(0)
    dept_summary['PctOvertime']  = (dept_summary['PctOvertime'] * 100).round(1)

    return high_risk, dept_summary

if __name__ == '__main__':
    high_risk, summary = get_department_alerts()
    print(f"Total high-risk employees: {len(high_risk)}")
    print("\nDepartment Alert Summary:")
    print(summary.to_string(index=False))