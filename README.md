# Employee Attrition Intelligence — HR Analytics Platform

A machine learning project that predicts which employees are likely to leave, identifies the key drivers behind attrition, and presents findings in an interactive dashboard built for HR teams.

🔗 **Live Demo:** [HR Attrition Intelligence Dashboard](https://hr-attrition-analytics-dashboard.streamlit.app/)
---

## Live Features

- Predictive model classifying employees as high / medium / low attrition risk
- SHAP explainability showing exactly why each employee is flagged
- Interactive dashboard with filters by department, role, and risk level
- Automated department-level alert system for HR managers
- Real national turnover benchmarks via BLS API (JOLTS Survey)
- Modern Minimal UI with Emerald Green theme

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| ML Model | XGBoost + SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| Dashboard | Streamlit + Plotly |
| Data Pipeline | Pandas, NumPy, Scikit-learn |
| External API | US Bureau of Labor Statistics (BLS) |
| Deployment | Streamlit Community Cloud |
| Visualisation | Tableau Public |

---

## Data Sources

| Dataset | Rows | Contribution |
|---|---|---|
| IBM HR Analytics (Kaggle) | 1,470 | Core attrition labels + 32 features |
| HRDataset v14 (Kaggle) | 311 | Engagement survey, absences, tardiness |
| Employee Burnout (Kaggle) | 22,750 | Burn rate, mental fatigue, WFH access |

All three datasets were merged into a single enriched master dataset of 1,470 employees
with 40 features using statistical matching on shared demographic attributes.

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.7797 |
| Recall (Left) | 0.53 |
| Precision (Left) | 0.44 |
| F1 (Left) | 0.48 |
| Optimized Threshold | 0.29 |

Threshold was tuned from the default 0.50 to 0.29 to maximize recall —
in HR analytics catching at-risk employees matters more than minimizing false alarms.

---

## Top Attrition Drivers (SHAP)

1. StockOptionLevel
2. JobSatisfaction
3. OverTime
4. EnvironmentSatisfaction
5. JobInvolvement
6. EngagementSurvey ← enriched from HRv14
7. DistanceFromHome
8. WorkLifeBalance
9. BusinessTravel
10. NumCompaniesWorked

---

## Project Structure

```
hr_attrition/
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   ├── HRDataset_v14.csv
│   ├── train.csv
│   ├── master_hr_dataset.csv
│   └── processed_hr_dataset.csv
│
├── src/
│   ├── build_dataset.py      ← multi-source data pipeline
│   ├── preprocess.py         ← encoding + feature engineering
│   ├── train_model.py        ← XGBoost + SMOTE training
│   ├── tune_threshold.py     ← threshold optimization
│   ├── explain.py            ← SHAP explainability
│   ├── alerts.py             ← department alert system
│   └── bls_data.py           ← BLS API integration
│
├── models/                   ← saved model + outputs (auto-generated)
│
├── app/
│   ├── dashboard.py          ← Streamlit dashboard
│   └── theme.py              ← color theme configuration
│
├── notebooks/
│   └── 01_eda.ipynb          ← exploratory data analysis
│
├── .streamlit/
│   └── config.toml           ← Streamlit theme configuration
│
├── run.py                    ← smart launcher with auto-regeneration
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hr-attrition-analytics.git
cd hr-attrition-analytics
```

### 2. Create conda environment
```bash
conda create -n hr_attrition python=3.10 -y
conda activate hr_attrition
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install xgboost==1.7.6
```

### 4. Download datasets
Download these three datasets from Kaggle and place them in `data/`:
- [IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) → `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- [HRDataset v14](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) → `HRDataset_v14.csv`
- [Employee Burnout](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out) → `train.csv`

### 5. Run the project
```bash
python run.py
```

This smart launcher automatically checks if all model files exist.
If they do, it skips training and launches the dashboard directly.
If any files are missing, it runs the full pipeline first then launches.

> **Manual pipeline** (only needed if you want to retrain individually):
> ```bash
> python src/build_dataset.py
> python src/preprocess.py
> python src/train_model.py
> python src/tune_threshold.py
> python src/explain.py
> streamlit run app/dashboard.py
> ```

---

## Key Design Decisions

**Why threshold tuning?**
Default XGBoost threshold of 0.50 only caught 36% of employees who actually left.
Tuning to 0.29 raised recall to 53% — meaning HR teams catch significantly more
at-risk employees before they resign.

**Why multi-source data?**
The IBM dataset alone lacks behavioral signals like engagement scores, absenteeism,
and burnout. Enriching it with HRv14 and Burnout data added 8 new features, two of
which (EngagementSurvey, Absences) ranked in the top 15 most important SHAP features.

**Why BLS API?**
Predicted attrition rates mean more in context. Benchmarking against real national
quit rates from the Bureau of Labor Statistics gives HR teams a reference point
beyond the model's internal scores.

**Why SHAP?**
A model that produces a score without a reason is difficult to trust and impossible
to act on. SHAP values turn the black box into an explainable tool that HR managers
can use and defend in real conversations.
