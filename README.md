\# Employee Attrition Intelligence — HR Analytics Platform



A machine learning project that predicts which employees are likely to leave,

identifies the key drivers behind attrition, and presents findings in an

interactive Streamlit dashboard built for HR teams.



\---



\## Live Features



\- Predictive model classifying employees as high / medium / low attrition risk

\- SHAP explainability showing exactly why each employee is flagged

\- Interactive dashboard with filters by department, role, and risk level

\- Automated department-level alert system for HR managers

\- Real national turnover benchmarks via BLS API (JOLTS Survey)



\---



\## Tech Stack



| Layer | Tools |

|---|---|

| Language | Python 3.10 |

| ML Model | XGBoost + SMOTE (imbalanced-learn) |

| Explainability | SHAP (TreeExplainer) |

| Dashboard | Streamlit + Plotly |

| Data Pipeline | Pandas, NumPy, Scikit-learn |

| External API | US Bureau of Labor Statistics (BLS) |



\---



\## Data Sources



| Dataset | Rows | Contribution |

|---|---|---|

| IBM HR Analytics (Kaggle) | 1,470 | Core attrition labels + 32 features |

| HRDataset v14 (Kaggle) | 311 | Engagement survey, absences, tardiness |

| Employee Burnout (Kaggle) | 22,750 | Burn rate, mental fatigue, WFH access |



All three datasets were merged into a single enriched master dataset

of 1,470 employees with 40 features using statistical matching on

shared demographic attributes.



\---



\## Model Performance



| Metric | Score |

|---|---|

| ROC-AUC | 0.7797 |

| Recall (Left) | 0.53 |

| Precision (Left) | 0.44 |

| F1 (Left) | 0.48 |

| Optimized Threshold | 0.29 |



Threshold was tuned from the default 0.50 to 0.29 to maximize recall —

in HR analytics catching at-risk employees matters more than minimizing

false alarms.



\---



\## Top Attrition Drivers (SHAP)



1\. StockOptionLevel

2\. JobSatisfaction

3\. OverTime

4\. EnvironmentSatisfaction

5\. JobInvolvement

6\. EngagementSurvey ← enriched from HRv14

7\. DistanceFromHome

8\. WorkLifeBalance

9\. BusinessTravel

10\. NumCompaniesWorked



\---



\## Project Structure

hr\_attrition/

│

├── data/

│   ├── WA\_Fn-UseC\_-HR-Employee-Attrition.csv

│   ├── HRDataset\_v14.csv

│   ├── train.csv

│   ├── master\_hr\_dataset.csv

│   └── processed\_hr\_dataset.csv

│

├── src/

│   ├── build\_dataset.py      ← multi-source data pipeline

│   ├── preprocess.py         ← encoding + feature engineering

│   ├── train\_model.py        ← XGBoost + SMOTE training

│   ├── tune\_threshold.py     ← threshold optimization

│   ├── explain.py            ← SHAP explainability

│   ├── alerts.py             ← department alert system

│   └── bls\_data.py           ← BLS API integration

│

├── models/                   ← saved model + outputs (auto-generated)

├── app/

│   └── dashboard.py          ← Streamlit dashboard

├── notebooks/

│   └── 01\_eda.ipynb          ← exploratory data analysis

└── requirements.txt



\---



\## Setup \& Installation



\### 1. Clone the repository

```bash

git clone https://github.com/YOUR\_USERNAME/hr\_attrition.git

cd hr\_attrition

```



\### 2. Create conda environment

```bash

conda create -n hr\_attrition python=3.10 -y

conda activate hr\_attrition

```



\### 3. Install dependencies

```bash

pip install -r requirements.txt

pip install xgboost==1.7.6

```



\### 4. Download datasets

Download these three datasets from Kaggle and place them in `data/`:

\- \[IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) → `WA\_Fn-UseC\_-HR-Employee-Attrition.csv`

\- \[HRDataset v14](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) → `HRDataset\_v14.csv`

\- \[Employee Burnout](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out) → `train.csv`



\### 5. Run the pipeline

```bash

python src/build\_dataset.py

python src/preprocess.py

python src/train\_model.py

python src/tune\_threshold.py

python src/explain.py

```



\### 6. Launch the dashboard

```bash

streamlit run app/dashboard.py

```



\---



\## Key Design Decisions



\*\*Why threshold tuning?\*\*

Default XGBoost threshold of 0.50 only caught 36% of employees who

actually left. Tuning to 0.29 raised recall to 53% — meaning HR teams

catch significantly more at-risk employees before they resign.



\*\*Why multi-source data?\*\*

The IBM dataset alone lacks behavioral signals like engagement scores,

absenteeism, and burnout. Enriching it with HRv14 and Burnout data

added 8 new features, two of which (EngagementSurvey, Absences) ranked

in the top 15 most important SHAP features.



\*\*Why BLS API?\*\*

Predicted attrition rates mean more in context. Benchmarking against

real national quit rates from the Bureau of Labor Statistics gives HR

teams a reference point beyond the model's internal scores.

