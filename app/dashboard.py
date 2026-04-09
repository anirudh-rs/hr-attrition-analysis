import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.alerts import get_department_alerts
from src.bls_data import load_bls_data

st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    pred = pd.read_csv('models/predictions.csv')
    raw  = pd.read_csv('data/master_hr_dataset.csv')
    pred['OverTime_label']  = raw['OverTime']
    pred['Attrition_label'] = raw['Attrition']
    return pred, raw

@st.cache_data
def load_shap():
    return pd.read_csv('models/shap_values.csv')

pred_df, raw_df = load_data()
color_map = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}

with st.sidebar:
    st.title("👥 HR Attrition Analytics")
    st.markdown("---")

    departments = ['All'] + sorted(pred_df['Department'].unique().tolist())
    dept_filter = st.selectbox("Department", departments)

    roles = ['All'] + sorted(pred_df['JobRole'].unique().tolist())
    role_filter = st.selectbox("Job Role", roles)

    risk_filter = st.multiselect(
        "Risk Level",
        ['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low']
    )

    st.markdown("---")
    st.markdown("### Model Info")
    st.info(
        "Algorithm: XGBoost\n\n"
        "ROC-AUC: 0.7797\n\n"
        "Threshold: 0.29\n\n"
        "Features: 44\n\n"
        "Data: IBM + HRv14 + Burnout"
    )

filtered = pred_df.copy()
if dept_filter != 'All':
    filtered = filtered[filtered['Department'] == dept_filter]
if role_filter != 'All':
    filtered = filtered[filtered['JobRole'] == role_filter]
filtered = filtered[filtered['RiskLabel'].isin(risk_filter)]

total       = len(filtered)
high_n      = len(filtered[filtered['RiskLabel'] == 'High'])
medium_n    = len(filtered[filtered['RiskLabel'] == 'Medium'])
actual_left = len(filtered[filtered['Attrition_label'] == 'Yes'])
avg_risk    = filtered['AttritionRisk'].mean() if total > 0 else 0

st.title("👥 Employee Attrition Risk Dashboard")
st.caption("XGBoost + SHAP | IBM Dataset enriched with HRv14 & Burnout signals")
st.divider()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Employees", f"{total:,}")
k2.metric("High Risk",       f"{high_n:,}")
k3.metric("Medium Risk",     f"{medium_n:,}")
k4.metric("Hist. Attrition", f"{actual_left:,}")
k5.metric("Avg Risk Score",  f"{avg_risk:.3f}")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "⚠️ Risk Analysis",
    "🔍 Model Insights",
    "🚨 Department Alerts",
    "📈 BLS Benchmark"
])

with tab1:
    st.subheader("Workforce Overview")

    c1, c2 = st.columns(2)

    with c1:
        risk_counts = filtered['RiskLabel'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        fig = px.pie(
            risk_counts, names='Risk', values='Count',
            hole=0.5, title='Risk Level Distribution',
            color='Risk', color_discrete_map=color_map
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        dept_rate = raw_df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        dept_rate.columns = ['Department', 'AttritionRate']
        fig2 = px.bar(
            dept_rate, x='Department', y='AttritionRate',
            title='Historical Attrition Rate by Department (%)',
            color='AttritionRate', color_continuous_scale='Reds',
            text=dept_rate['AttritionRate'].round(1).astype(str) + '%'
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        stayed = raw_df[raw_df['Attrition'] == 'No']['Age']
        left   = raw_df[raw_df['Attrition'] == 'Yes']['Age']

        fig3 = go.Figure()
        fig3.add_trace(go.Box(
            y=stayed, name='Stayed',
            marker_color='#2ecc71',
            boxmean=True
        ))
        fig3.add_trace(go.Box(
            y=left, name='Left',
            marker_color='#e74c3c',
            boxmean=True
        ))
        fig3.update_layout(
            title='Age Distribution by Attrition',
            height=320,
            yaxis_title='Age',
            showlegend=True
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.box(
            raw_df, x='Attrition', y='MonthlyIncome',
            color='Attrition', title='Monthly Income vs Attrition',
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
        )
        fig4.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    role_dept = raw_df.groupby(['Department', 'JobRole'])['Attrition'].apply(
        lambda x: round((x == 'Yes').mean() * 100, 1)
    ).reset_index()
    role_dept.columns = ['Department', 'JobRole', 'AttritionRate']
    fig5 = px.bar(
        role_dept.sort_values('AttritionRate', ascending=True),
        x='AttritionRate', y='JobRole',
        color='Department',
        title='Attrition Rate by Job Role (%)',
        orientation='h'
    )
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
    st.subheader("Employee Risk Analysis")

    c1, c2 = st.columns([2, 1])

    with c1:
        fig = px.scatter(
            filtered,
            x='MonthlyIncome', y='AttritionRisk',
            color='RiskLabel', color_discrete_map=color_map,
            hover_data=['JobRole', 'Department', 'YearsAtCompany'],
            title='Monthly Income vs Attrition Risk Score'
        )
        fig.add_hline(
            y=0.60, line_dash='dash', line_color='red',
            opacity=0.5, annotation_text='High Risk (0.60)'
        )
        fig.add_hline(
            y=0.29, line_dash='dash', line_color='orange',
            opacity=0.5, annotation_text='Model Threshold (0.29)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ot_risk = filtered.groupby(
            'OverTime_label'
        )['AttritionRisk'].mean().reset_index()
        ot_risk.columns = ['OverTime', 'AvgRisk']
        fig2 = px.bar(
            ot_risk, x='OverTime', y='AvgRisk',
            title='Avg Risk by Overtime',
            color='OverTime',
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
        )
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🔴 High Risk Employee List")

    high_risk_table = filtered[filtered['RiskLabel'] == 'High'][[
        'EmployeeNumber', 'Department', 'JobRole',
        'MonthlyIncome', 'YearsAtCompany', 'OverTime_label',
        'JobSatisfaction', 'AttritionRisk'
    ]].copy().sort_values('AttritionRisk', ascending=False)

    high_risk_table['AttritionRisk'] = high_risk_table['AttritionRisk'].round(3)
    high_risk_table.columns = [
        'Emp #', 'Department', 'Job Role',
        'Monthly Income', 'Years', 'Overtime',
        'Job Satisfaction', 'Risk Score'
    ]

    st.dataframe(
        high_risk_table,
        use_container_width=True,
        hide_index=True,
        height=320,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1, format="%.3f"
            ),
            "Monthly Income": st.column_config.NumberColumn(
                "Monthly Income", format="$%d"
            )
        }
    )

    st.download_button(
        "📥 Download High Risk List (CSV)",
        high_risk_table.to_csv(index=False),
        file_name='high_risk_employees.csv',
        mime='text/csv'
    )

with tab3:
    st.subheader("Model Explainability — SHAP Analysis")

    st.info(
        "**Bar chart:** overall feature importance  ·  "
        "**Beeswarm:** direction of impact — red = high value increases risk  ·  "
        "**Waterfall:** individual employee breakdown"
    )

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists('models/shap_importance.png'):
            st.image('models/shap_importance.png',
                     caption='Global Feature Importance',
                     use_container_width=True)
    with c2:
        if os.path.exists('models/shap_beeswarm.png'):
            st.image('models/shap_beeswarm.png',
                     caption='Feature Impact Direction',
                     use_container_width=True)

    st.markdown("### Top 3 High Risk Employee Explanations")
    e1, e2, e3 = st.columns(3)
    for col, rank in zip([e1, e2, e3], [1, 2, 3]):
        path = f'models/shap_employee_{rank}.png'
        if os.path.exists(path):
            col.image(path, caption=f'Employee #{rank}',
                      use_container_width=True)

    st.markdown("### Feature Importance Rankings")
    if os.path.exists('models/shap_values.csv'):
        shap_df    = load_shap()
        feat_names = joblib.load('models/feature_names.pkl')
        mean_shap  = pd.DataFrame({
            'Feature': feat_names,
            'Mean |SHAP|': np.abs(shap_df.values).mean(axis=0).round(4)
        }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
        mean_shap.index += 1

        enriched   = ['EngagementSurvey', 'EmpSatisfaction', 'DaysLateLast30',
                      'Absences', 'BurnRate', 'MentalFatigueScore',
                      'WFHAvailable', 'ResourceAllocation']
        engineered = ['SalaryPerLevel', 'PromotionStagnation', 'TotalSatisfaction',
                      'BurnoutPressureIndex', 'CareerVelocity', 'ManagerStability']

        def tag(f):
            if f in enriched:   return '🔵 Enriched'
            if f in engineered: return '🟣 Engineered'
            return '⚪ Original'

        mean_shap['Source'] = mean_shap['Feature'].apply(tag)
        st.dataframe(
            mean_shap, use_container_width=True, height=380,
            column_config={
                "Mean |SHAP|": st.column_config.ProgressColumn(
                    "Mean |SHAP|",
                    min_value=0,
                    max_value=float(mean_shap['Mean |SHAP|'].max()),
                    format="%.4f"
                )
            }
        )

with tab4:
    st.subheader("Department Alert Summary")

    _, dept_summary = get_department_alerts()

    cols = st.columns(len(dept_summary))
    dept_colors = ['#e74c3c', '#f39c12', '#3498db']

    for col, (_, row), color in zip(cols, dept_summary.iterrows(), dept_colors):
        with col:
            st.markdown(f"""
            <div style='background:{color}15; border-left:4px solid {color};
                        padding:1.2rem; border-radius:8px'>
                <div style='font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; color:{color};
                            letter-spacing:0.08em; margin-bottom:0.4rem'>
                    {row['Department']}
                </div>
                <div style='font-size:2rem; font-weight:800; color:{color}'>
                    {int(row['HighRiskCount'])}
                </div>
                <div style='font-size:0.72rem; color:#666; margin-bottom:0.6rem'>
                    high risk employees
                </div>
                <div style='font-size:0.78rem; color:#444; line-height:1.8'>
                    Avg score: <b>{row['AvgRiskScore']:.3f}</b><br>
                    Avg income: <b>${int(row['AvgIncome']):,}</b><br>
                    Overtime: <b>{row['PctOvertime']}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    fig = px.bar(
        dept_summary, x='Department', y='HighRiskCount',
        color='AvgRiskScore', color_continuous_scale='Reds',
        title='High Risk Count by Department',
        text='HighRiskCount'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **R&D (133 at risk):** Audit overtime and project load.
    Review stock options for junior researchers.

    **Sales (60 at risk):** Review commissions and travel frequency.

    **HR (10 at risk):** Senior staff at risk.
    Schedule confidential retention conversations.
    """)

with tab5:
    st.subheader("National Turnover Benchmark")
    st.caption("Real US quit rates — Bureau of Labor Statistics JOLTS Survey")

    bls_data = load_bls_data()
    our_rate = (pred_df['RiskLabel'] == 'High').mean() * 100

    bls_df  = pd.DataFrame([
        {'Industry': k, 'QuitRate': v, 'Type': 'National Average'}
        for k, v in bls_data.items()
    ])
    our_row = pd.DataFrame([{
        'Industry': 'Our Company (Predicted)',
        'QuitRate': round(our_rate, 2),
        'Type': 'Our Company'
    }])
    combined = pd.concat([bls_df, our_row], ignore_index=True)
    combined = combined.sort_values('QuitRate', ascending=True)

    c1, c2 = st.columns([3, 1])

    with c1:
        fig = px.bar(
            combined,
            x='QuitRate', y='Industry',
            color='Type', orientation='h',
            title='Monthly Quit Rate Comparison (%)',
            color_discrete_map={
                'National Average': '#3498db',
                'Our Company':      '#e74c3c'
            },
            text=combined['QuitRate'].astype(str) + '%'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avg_national = bls_df['QuitRate'].mean()
        diff         = our_rate - avg_national
        direction    = "above" if diff > 0 else "below"

        st.metric("Our Predicted Rate",  f"{our_rate:.1f}%")
        st.metric("National Average",    f"{avg_national:.1f}%")
        st.metric("Variance",
                  f"{abs(diff):.1f}% {direction} avg")

    st.markdown("### Detailed Comparison")
    bls_table = pd.DataFrame([
        {
            'Industry': k,
            'National Quit Rate (%)': v,
            'Our Rate (%)': round(our_rate, 2),
            'Difference': round(our_rate - v, 2)
        }
        for k, v in bls_data.items()
    ])
    st.dataframe(bls_table, use_container_width=True, hide_index=True)

    st.info(
        "Source: US Bureau of Labor Statistics JOLTS Survey 2023-2024. "
        "Our rate = model-predicted high risk employees as % of workforce."
    )