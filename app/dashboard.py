import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os

# Add both the project root and app folder to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, APP)

from theme import (
    PRIMARY, PRIMARY_LIGHT, HIGH_RISK, MEDIUM_RISK, LOW_RISK,
    CHART_MAIN, CHART_SEC, CHART_THIRD, CHART_FOURTH,
    BG, BG_SUBTLE, TEXT_DARK, TEXT_MID, PLOTLY_BASE,
    AXIS_STYLE, RISK_COLOR_MAP, GREEN_SEQ
)
from src.alerts import get_department_alerts
from src.bls_data import load_bls_data

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
color_map = RISK_COLOR_MAP

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
        fig = go.Figure(go.Pie(
            labels=risk_counts['Risk'],
            values=risk_counts['Count'],
            hole=0.6,
            marker=dict(
                colors=[RISK_COLOR_MAP.get(r, '#ccc') for r in risk_counts['Risk']],
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=11, color=TEXT_DARK),
            hovertemplate='<b>%{label}</b><br>%{value} employees<extra></extra>'
        ))
        fig.update_layout(
            **PLOTLY_BASE,
            title='Risk Level Distribution',
            height=350,
            legend=dict(orientation='h', y=-0.1, font=dict(size=10))
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        dept_rate = raw_df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        dept_rate.columns = ['Department', 'AttritionRate']
        dept_rate = dept_rate.sort_values('AttritionRate', ascending=True)
        fig2 = go.Figure(go.Bar(
            x=dept_rate['AttritionRate'],
            y=dept_rate['Department'],
            orientation='h',
            marker=dict(
                color=dept_rate['AttritionRate'],
                colorscale=GREEN_SEQ,
                showscale=False
            ),
            text=dept_rate['AttritionRate'].round(1).astype(str) + '%',
            textposition='outside',
            textfont=dict(size=11, color=TEXT_DARK)
        ))
        fig2.update_layout(
            **PLOTLY_BASE,
            title='Attrition Rate by Department (%)',
            height=350,
            xaxis=dict(**AXIS_STYLE, showticklabels=False),
            yaxis=dict(**AXIS_STYLE)
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        stayed = raw_df[raw_df['Attrition'] == 'No']['Age']
        left   = raw_df[raw_df['Attrition'] == 'Yes']['Age']
        fig3 = go.Figure()
        fig3.add_trace(go.Box(
            y=stayed, name='Stayed',
            marker_color=LOW_RISK,
            boxmean=True,
            line=dict(color=LOW_RISK)
        ))
        fig3.add_trace(go.Box(
            y=left, name='Left',
            marker_color=HIGH_RISK,
            boxmean=True,
            line=dict(color=HIGH_RISK)
        ))
        fig3.update_layout(
            **PLOTLY_BASE,
            title='Age Distribution by Attrition',
            height=320,
            yaxis=dict(**AXIS_STYLE, title='Age'),
            xaxis=dict(**AXIS_STYLE),
            legend=dict(orientation='h', y=1.1, font=dict(size=10))
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = go.Figure()
        for val, color, name in [('No', LOW_RISK, 'Stayed'), ('Yes', HIGH_RISK, 'Left')]:
            subset = raw_df[raw_df['Attrition'] == val]
            fig4.add_trace(go.Box(
                y=subset['MonthlyIncome'],
                name=name,
                marker_color=color,
                boxmean=True,
                line=dict(color=color)
            ))
        fig4.update_layout(
            **PLOTLY_BASE,
            title='Monthly Income vs Attrition',
            height=320,
            yaxis=dict(**AXIS_STYLE, title='Monthly Income ($)'),
            xaxis=dict(**AXIS_STYLE),
            showlegend=True,
            legend=dict(orientation='h', y=-0.15, font=dict(size=10))
        )
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
        orientation='h',
        color_discrete_sequence=[CHART_MAIN, CHART_SEC, CHART_THIRD]
    )
    fig5.update_layout(
        **PLOTLY_BASE,
        height=400,
        xaxis=dict(**AXIS_STYLE, title='Attrition Rate (%)'),
        yaxis=dict(**AXIS_STYLE),
        legend=dict(orientation='h', y=1.02, font=dict(size=10))
    )
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
    st.subheader("Employee Risk Analysis")

    c1, c2 = st.columns([2, 1])

    with c1:
        fig = go.Figure()
        for risk_level, color in RISK_COLOR_MAP.items():
            subset = filtered[filtered['RiskLabel'] == risk_level]
            if len(subset) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=subset['MonthlyIncome'],
                y=subset['AttritionRisk'],
                mode='markers',
                name=risk_level,
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                customdata=subset[['JobRole', 'Department', 'YearsAtCompany']].values,
                hovertemplate=(
                    '<b>%{customdata[1]}</b><br>'
                    'Role: %{customdata[0]}<br>'
                    'Income: $%{x:,.0f}<br>'
                    'Risk Score: %{y:.3f}<br>'
                    'Tenure: %{customdata[2]} yrs'
                    '<extra></extra>'
                )
            ))
        fig.add_hline(
            y=0.60, line_dash='dot',
            line_color=HIGH_RISK, line_width=1.5,
            opacity=0.7,
            annotation_text='High Risk (0.60)',
            annotation_font=dict(size=10, color=HIGH_RISK)
        )
        fig.add_hline(
            y=0.29, line_dash='dot',
            line_color=MEDIUM_RISK, line_width=1.5,
            opacity=0.7,
            annotation_text='Model Threshold (0.29)',
            annotation_font=dict(size=10, color=MEDIUM_RISK)
        )
        fig.update_layout(
            **PLOTLY_BASE,
            title='Monthly Income vs Attrition Risk Score',
            height=400,
            xaxis=dict(**AXIS_STYLE, title='Monthly Income ($)'),
            yaxis=dict(**AXIS_STYLE, title='Risk Score'),
            legend=dict(orientation='h', y=1.1, font=dict(size=10))
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ot_risk = filtered.groupby(
            'OverTime_label'
        )['AttritionRisk'].mean().reset_index()
        ot_risk.columns = ['OverTime', 'AvgRisk']
        fig2 = go.Figure(go.Bar(
            x=ot_risk['OverTime'],
            y=ot_risk['AvgRisk'],
            marker=dict(
                color=[HIGH_RISK if ot == 'Yes' else LOW_RISK
                       for ot in ot_risk['OverTime']],
                line=dict(width=0)
            ),
            text=ot_risk['AvgRisk'].round(3),
            textposition='outside',
            textfont=dict(size=11, color=TEXT_DARK)
        ))
        fig2.update_layout(
            **PLOTLY_BASE,
            title='Avg Risk Score by Overtime',
            height=300,
            xaxis=dict(**AXIS_STYLE),
            yaxis=dict(**AXIS_STYLE, range=[0, 0.7]),
            showlegend=False
        )
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
                    format="%.4f",
                    help="Higher value = stronger influence on attrition prediction"
                ),
                "Source": st.column_config.TextColumn(
                    "Source",
                    help="Original = IBM dataset | Enriched = HRv14/Burnout | Engineered = computed feature"
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

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dept_summary['Department'],
        y=dept_summary['HighRiskCount'],
        marker=dict(
            color=dept_summary['AvgRiskScore'],
            colorscale=GREEN_SEQ,
            showscale=True,
            colorbar=dict(
                title='Avg Risk',
                thickness=12,
                len=0.6,
                tickfont=dict(size=9, color=TEXT_MID)
            ),
            line=dict(width=0)
        ),
        text=dept_summary['HighRiskCount'],
        textposition='outside',
        textfont=dict(size=13, color=TEXT_DARK)
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title='High Risk Employee Count by Department',
        height=380,
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE, title='High Risk Count'),
        showlegend=False
    )
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
        fig = go.Figure()
        for type_label, color in [
            ('National Average', CHART_SEC),
            ('Our Company',      PRIMARY)
        ]:
            subset = combined[combined['Type'] == type_label]
            fig.add_trace(go.Bar(
                x=subset['QuitRate'],
                y=subset['Industry'],
                name=type_label,
                orientation='h',
                marker=dict(color=color, line=dict(width=0)),
                text=subset['QuitRate'].astype(str) + '%',
                textposition='outside',
                textfont=dict(size=10, color=TEXT_DARK)
            ))
        fig.update_layout(
            **PLOTLY_BASE,
            title='Monthly Quit Rate Comparison (%)',
            height=380,
            barmode='group',
            xaxis=dict(**AXIS_STYLE, title='Monthly Quit Rate (%)'),
            yaxis=dict(**AXIS_STYLE),
            legend=dict(orientation='h', y=1.1, font=dict(size=10))
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avg_national = bls_df['QuitRate'].mean()
        diff         = our_rate - avg_national
        direction    = "above" if diff > 0 else "below"

        st.metric("Our Predicted Rate", f"{our_rate:.1f}%")
        st.metric("National Average", f"{avg_national:.1f}%")
        st.metric("Variance", f"{abs(diff):.1f}%")
        st.caption(f"{direction} national average")

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