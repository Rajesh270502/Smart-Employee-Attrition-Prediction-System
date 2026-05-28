import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

# ── Load Saved Artifacts ──────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load('attrition_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    columns  = joblib.load('feature_columns.pkl')
    return model, scaler, encoders, columns

model, scaler, encoders, feature_columns = load_artifacts()

# ── Helper Functions ──────────────────────────────────────────
def categorize_risk(score):
    if score < 0.4:
        return 'Low Risk', '🟢'
    elif score < 0.7:
        return 'Medium Risk', '🟡'
    else:
        return 'High Risk', '🔴'


def preprocess_employee(data):
    df = pd.DataFrame([data])

    for col in df.select_dtypes(include='object').columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    df['OverTimeRisk']       = (df['OverTime'] == 1).astype(int) * df['DistanceFromHome']
    df['SatisfactionIndex']  = df['EnvironmentSatisfaction'] + df['JobSatisfaction']
    df['StockOptionWeight']  = df['StockOptionLevel'] * df['MonthlyIncome']
    df['ExperienceGap']      = df['TotalWorkingYears'] - df['YearsAtCompany']
    df['CareerStability']    = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['CareerGrowthRisk']   = df['YearsSinceLastPromotion'] / (df['JobLevel'] + 1)
    df['UnderPromotionRisk'] = df['TotalWorkingYears'] / (df['JobLevel'] + 1)
    df['BurnoutRisk']        = df['JobLevel'] * df['OverTime'] * (df['DistanceFromHome'] + 1)

    df = df.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(df)


def get_recommendations(data, risk_label):
    recs = []
    if risk_label == 'Low Risk':
        recs.append("✅ Employee is stable. Continue regular check-ins.")
        recs.append("📈 Offer growth opportunities to maintain engagement.")
    else:
        if data.get('OverTime') == 'Yes':
            recs.append("⚠️ Reduce overtime — high burnout risk detected.")
            recs.append("🕐 Offer flexible or remote work options.")
        if data.get('MonthlyIncome', 0) < 5000:
            recs.append("💰 Review salary — below average compensation.")
            recs.append("🎁 Offer performance-based bonuses.")
        if data.get('JobSatisfaction', 3) <= 2:
            recs.append("😞 Low job satisfaction — conduct 1-on-1 feedback sessions.")
            recs.append("🔄 Consider role rotation or new project assignments.")
        if data.get('WorkLifeBalance', 3) <= 2:
            recs.append("🏠 Poor work-life balance — encourage leave utilization.")
            recs.append("🧘 Introduce employee wellness programs.")
        if data.get('YearsSinceLastPromotion', 0) >= 3:
            recs.append("🚀 No promotion in 3+ years — evaluate for advancement.")
            recs.append("📋 Create a clear promotion roadmap.")
        if data.get('DistanceFromHome', 0) >= 20:
            recs.append("🚗 Long commute — offer work-from-home flexibility.")
        if data.get('EnvironmentSatisfaction', 3) <= 2:
            recs.append("🏢 Low environment satisfaction — address workplace culture.")
        if data.get('StockOptionLevel', 0) == 0:
            recs.append("📊 Offer stock options or equity benefits.")
        if data.get('TrainingTimesLastYear', 0) <= 1:
            recs.append("📚 Provide more training & upskilling opportunities.")
        if risk_label == 'High Risk':
            recs.append("🚨 URGENT: Schedule immediate HR retention conversation.")
            recs.append("📝 Create a personalized retention plan within 30 days.")
    return recs


# ── UI ────────────────────────────────────────────────────────
st.title("👥 Smart Employee Attrition Prediction System")
st.markdown("Predict employee attrition risk and get retention recommendations.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("📋 Employee Details")

age             = st.sidebar.slider("Age", 18, 60, 30)
department      = st.sidebar.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
job_role        = st.sidebar.selectbox("Job Role", [
                    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
                    'Manufacturing Director', 'Healthcare Representative',
                    'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
monthly_income  = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
overtime        = st.sidebar.selectbox("OverTime", ['Yes', 'No'])
job_satisfaction= st.sidebar.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 2)
work_life       = st.sidebar.slider("Work Life Balance (1=Low, 4=High)", 1, 4, 2)
env_satisfaction= st.sidebar.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 2)
distance        = st.sidebar.slider("Distance From Home (km)", 1, 50, 10)
years_company   = st.sidebar.slider("Years at Company", 0, 40, 5)
years_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
job_level       = st.sidebar.slider("Job Level (1-5)", 1, 5, 2)
stock_option    = st.sidebar.slider("Stock Option Level (0-3)", 0, 3, 0)
training        = st.sidebar.slider("Training Times Last Year", 0, 6, 2)
total_years     = st.sidebar.slider("Total Working Years", 0, 40, 8)
years_role      = st.sidebar.slider("Years in Current Role", 0, 20, 3)
business_travel = st.sidebar.selectbox("Business Travel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
education       = st.sidebar.slider("Education Level (1-5)", 1, 5, 3)
gender          = st.sidebar.selectbox("Gender", ['Male', 'Female'])
marital_status  = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
num_companies   = st.sidebar.slider("Num Companies Worked", 0, 9, 2)
perf_rating     = st.sidebar.slider("Performance Rating (1-4)", 1, 4, 3)
years_manager   = st.sidebar.slider("Years With Current Manager", 0, 17, 3)

predict_btn = st.sidebar.button("🔍 Predict Attrition", use_container_width=True, type="primary")

# ── Employee Data Dict ────────────────────────────────────────
employee_data = {
    'Age': age,
    'BusinessTravel': business_travel,
    'DailyRate': 800,
    'Department': department,
    'DistanceFromHome': distance,
    'Education': education,
    'EducationField': 'Life Sciences',
    'EnvironmentSatisfaction': env_satisfaction,
    'Gender': gender,
    'HourlyRate': 60,
    'JobInvolvement': 3,
    'JobLevel': job_level,
    'JobRole': job_role,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_status,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': 14000,
    'NumCompaniesWorked': num_companies,
    'OverTime': overtime,
    'PercentSalaryHike': 11,
    'PerformanceRating': perf_rating,
    'RelationshipSatisfaction': 2,
    'StockOptionLevel': stock_option,
    'TotalWorkingYears': total_years,
    'TrainingTimesLastYear': training,
    'WorkLifeBalance': work_life,
    'YearsAtCompany': years_company,
    'YearsInCurrentRole': years_role,
    'YearsSinceLastPromotion': years_promotion,
    'YearsWithCurrManager': years_manager
}

# ── Prediction & Results ──────────────────────────────────────
if predict_btn:

    X_input    = preprocess_employee(employee_data)
    prediction = model.predict(X_input)[0]
    risk_score = model.predict_proba(X_input)[0][1]
    risk_label, risk_icon = categorize_risk(risk_score)

    # Row 1 — 3 metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        result_text = "Will Leave 🚨" if prediction == 1 else "Will Stay ✅"
        color = "#ff4b4b" if prediction == 1 else "#00c853"
        st.markdown(f"""
        <div style='background:{color}22; border:2px solid {color};
                    border-radius:12px; padding:20px; text-align:center;'>
            <h3>Prediction</h3>
            <h2>{result_text}</h2>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:#1e1e2e; border:2px solid #7c7cff;
                    border-radius:12px; padding:20px; text-align:center;'>
            <h3>Risk Score</h3>
            <h2>{risk_score:.2%}</h2>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background:#1e1e2e; border:2px solid #ffcc00;
                    border-radius:12px; padding:20px; text-align:center;'>
            <h3>Risk Category</h3>
            <h2>{risk_icon} {risk_label}</h2>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Row 2 — Gauge + Recommendations
    col_gauge, col_recs = st.columns([1, 1])

    with col_gauge:
        st.subheader("🎯 Risk Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={'text': "Attrition Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4b4b"},
                'steps': [
                    {'range': [0, 40],   'color': "rgba(0, 200, 83, 0.2)"},
                    {'range': [40, 70],  'color': "rgba(255, 204, 0, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.2)"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8,
                    'value': risk_score * 100
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_recs:
        st.subheader("💡 HR Recommendations")
        recs = get_recommendations(employee_data, risk_label)
        for rec in recs:
            st.markdown(f"- {rec}")

    st.divider()

    # Row 3 — Risk Factor Bar Chart
    st.subheader("📊 Key Risk Factors")

    factors = {
        'Job Satisfaction':      job_satisfaction / 4,
        'Work Life Balance':     work_life / 4,
        'Env Satisfaction':      env_satisfaction / 4,
        'Years Since Promotion': min(years_promotion / 10, 1),
        'Distance From Home':    min(distance / 50, 1),
        'OverTime Risk':         1 if overtime == 'Yes' else 0,
        'Stock Option':          stock_option / 3,
        'Training':              training / 6,
    }

    risk_factors = {
        'Job Dissatisfaction':   1 - factors['Job Satisfaction'],
        'Poor Work-Life Balance':1 - factors['Work Life Balance'],
        'Low Env Satisfaction':  1 - factors['Env Satisfaction'],
        'Promotion Gap':         factors['Years Since Promotion'],
        'Long Commute':          factors['Distance From Home'],
        'Overtime Burden':       factors['OverTime Risk'],
        'No Stock Options':      1 - factors['Stock Option'],
        'Low Training':          1 - factors['Training'],
    }

    risk_series = pd.Series(risk_factors).sort_values(ascending=True)

    colors = [
        '#ff4b4b' if v > 0.5 else '#ffcc00' if v > 0.3 else '#00c853'
        for v in risk_series.values
    ]

    fig2 = go.Figure(go.Bar(
        x=risk_series.values,
        y=risk_series.index,
        orientation='h',
        marker_color=colors
    ))

    fig2.update_layout(
        xaxis=dict(range=[0, 1], title="Risk Contribution"),
        height=350,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Fill in employee details in the sidebar and click **Predict Attrition**.")