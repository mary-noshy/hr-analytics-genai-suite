import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from google import genai

st.set_page_config(page_title="Enterprise HR Analytics & Insights", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f1f3f5; border-radius: 6px 6px 0px 0px; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1E3A8A !important; color: white !important; }
    div[data-testid="stMetricValue"] { font-size: 32px; font-weight: bold; color: #1E3A8A; }
    .action-container { background-color: #eef2f6; border-left: 5px solid #1E3A8A; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('HR-Employee-Attrition.csv')

@st.cache_resource
def train_pipeline(df):
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    X = df_clean.drop(columns=['Attrition'])
    y = df_clean['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=150, max_depth=12))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = pipeline.named_steps['classifier'].feature_importances_
    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
    feat_imp_df['feature'] = feat_imp_df['feature'].str.replace('num__', '').str.replace('cat__', '')
    
    return pipeline, X, X_train, X_test, y_train, y_test, feat_imp_df, num_cols, cat_cols

df = load_data()
pipeline, X_orig, X_train, X_test, y_train, y_test, feat_imp_df, num_cols, cat_cols = train_pipeline(df)

# SIDEBAR BUSINESS COST CONFIGURATOR
st.sidebar.title("💰 Business Cost Parameters")
st.sidebar.markdown("Configure estimated capital impact metrics per employee to assess organizational risk overhead values.")
recruitment_cost = st.sidebar.slider("Average Recruitment Cost ($)", 1000, 25000, 5000, step=500)
onboarding_cost = st.sidebar.slider("Average Onboarding/Training Cost ($)", 500, 15000, 3000, step=500)

total_cost_per_head = recruitment_cost + onboarding_cost
historical_churn_count = int((df['Attrition'] == 'Yes').sum())
estimated_historical_loss = historical_churn_count * total_cost_per_head

st.title("📊 Enterprise HR Analytics & GenAI Chatbot Suite")

tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard (EDA)", "🤖 Attrition Predictor & Diagnostics", "💬 GenAI Retention Advisor & Chatbot"])

# TAB 1: DASHBOARD
with tab1:
    st.subheader("🏢 Organizational Talent Landscape")
    total_employees = len(df)
    attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
    avg_income = df['MonthlyIncome'].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Headcount", f"{total_employees:,}")
    m2.metric("Observed Attrition Rate", f"{attrition_rate:.1f}%")
    m3.metric("Average Monthly Income", f"${avg_income:,.0f}")
    m4.metric("Realized Capital Churn Loss", f"${estimated_historical_loss:,.0f}", delta="- Financial Overhead", delta_color="inverse")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        attrition_counts = df['Attrition'].value_counts().reset_index()
        attrition_counts.columns = ['Attrition', 'Count']
        st.plotly_chart(px.pie(attrition_counts, values='Count', names='Attrition', title='Workforce Retention Breakdown', hole=0.4, color_discrete_sequence=['#2E91E5', '#E15F41']), use_container_width=True)
        
        fig3 = df.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')
        st.plotly_chart(px.bar(fig3, x='OverTime', y='Count', color='Attrition', barmode='group', title='Attrition Risks Relative to Overtime Demands', color_discrete_sequence=['#2E91E5', '#E15F41']), use_container_width=True)
    with c2:
        fig2 = px.box(df, x='JobRole', y='MonthlyIncome', color='Attrition', title='Compensation Compaction Across Job Roles', color_discrete_sequence=['#2E91E5', '#E15F41'])
        fig2.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.plotly_chart(px.histogram(df, x='DistanceFromHome', color='Attrition', barmode='overlay', title='Commute Proximity Vectors vs Attrition Tendency', color_discrete_sequence=['#2E91E5', '#E15F41']), use_container_width=True)

# TAB 2: ML MODEL
with tab2:
    diag_col, sim_col = st.columns([4, 5])
    with diag_col:
        st.subheader("📈 Pipeline Diagnostics")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        st.markdown(f"**Model Accuracy:** {accuracy_score(y_test, y_pred)*100:.2f}%")
        st.markdown(f"**ROC-AUC Performance Score:** {roc_auc_score(y_test, y_prob):.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True"), x=['Retained', 'Attrited'], y=['Retained', 'Attrited'], color_continuous_scale='Blues', title='Confusion Matrix Heatmap'), use_container_width=True)
        st.plotly_chart(px.bar(feat_imp_df.head(10), x='importance', y='feature', orientation='h', title='Top 10 Global Features Driving Model Predictions', color='importance', color_continuous_scale='Blues'), use_container_width=True)

    with sim_col:
        st.subheader("🔮 Attrition Risk Sandbox")
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_age = st.slider("Employee Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].median()))
            sim_income = st.slider("Monthly Salary ($)", int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max()), int(df['MonthlyIncome'].median()))
            sim_overtime = st.selectbox("Mandatory Overtime Status", ['No', 'Yes'])
            sim_distance = st.slider("Commute Distance (Miles)", int(df['DistanceFromHome'].min()), int(df['DistanceFromHome'].max()), int(df['DistanceFromHome'].median()))
            sim_working_years = st.slider("Cumulative Working Experience", int(df['TotalWorkingYears'].min()), int(df['TotalWorkingYears'].max()), int(df['TotalWorkingYears'].median()))
        with sc2:
            sim_job_role = st.selectbox("Target Corporate Function", df['JobRole'].unique())
            sim_marital = st.selectbox("Marital Status Category", df['MaritalStatus'].unique())
            sim_env_sat = st.slider("Workplace Environment Satisfaction", 1, 4, 3)
            sim_work_life = st.slider("Work-Life Balance Index Score", 1, 4, 3)
            sim_stock = st.slider("Stock Option Ownership Level", 0, 3, 1)
        
        input_data = {col: (df[col].median() if col in num_cols else df[col].mode()[0]) for col in X_orig.columns}
        input_data.update({ 'Age': sim_age, 'MonthlyIncome': sim_income, 'OverTime': sim_overtime, 'DistanceFromHome': sim_distance, 'TotalWorkingYears': sim_working_years, 'JobRole': sim_job_role, 'MaritalStatus': sim_marital, 'EnvironmentSatisfaction': sim_env_sat, 'WorkLifeBalance': sim_work_life, 'StockOptionLevel': sim_stock })
        
        prob = pipeline.predict_proba(pd.DataFrame([input_data]))[0][1]
        st.markdown("---")
        st.markdown(f"#### Computed Attrition Probability Status: **{prob*100:.1f}%**")
        
        sim_risk_cost_impact = float(prob) * total_cost_per_head
        st.markdown(f"📊 *Proportional Financial Risk Exposure for this profile:* **${sim_risk_cost_impact:,.2f}**")
        
        if prob > 0.50: st.error("🚨 **High Attrition Risk**")
        else: st.success("✅ **Stable Profile**")
        st.progress(float(prob))

# TAB 3: GENAI CHATBOT
with tab3:
    st.header("🧠 Strategic People Operations Advisor")
    st.markdown("Interact dynamically with Gemini 2.5 regarding mitigating employee churn, structured directly on your trained model's feature outputs.")
    
    # Secure validation using explicit client parameter mapping
    client = genai.Client(api_key="AQ.Ab8RN6KLQRHOH8TDE2PupKnq3FqlPun2YgFZUxIrTYLacBviVA")
    top_drivers = feat_imp_df.head(5).to_string(index=False)
    
    system_instruction = f"""You are an expert Chief HR Strategist and Workforce Advisor. Your company's machine learning model points out that the top 5 drivers causing employee attrition are:
{top_drivers}

CRITICAL SAFETY GUARDRAIL: You are strictly restricted to answering questions related to HR strategy, people analytics, workplace management, corporate culture, or the listed data drivers. If the user's input is completely out of scope or irrelevant to professional workplace topics (such as cooking recipes, casual pop culture, coding unrelated algorithms, or random trivia), you must refuse to answer. Respond with: 'I am optimized strictly as a Strategic People Operations Advisor. I cannot assist with out-of-scope inquiries. Please ask a question regarding workforce management or retention strategy.'"""

    st.markdown("<div class='action-container'><strong>📋 Executive Toolbox:</strong> Need an authoritative document for leadership? Click the button below to auto-generate a formalized corporate framework matching our core risk drivers.</div>", unsafe_allow_html=True)
    
    if st.button("✨ Draft Formal Corporate Retention Policy Document"):
        with st.spinner("Invoking Gemini 2.5 Flash to synthesize macro policy documents..."):
            try:
                doc_prompt = f"""
                You are a Principal People Strategy Consultant. Draft a comprehensive, formal corporate policy document titled: 'Enterprise Retention Framework & Mitigation Blueprint'.
                Base the blueprint explicitly on these data drivers identified by our Random Forest model:
                {top_drivers}
                
                The document must use clear markdown formatting and include these exact sections:
                1. **Executive Mandate**: Frame the macro retention challenge and the total business cost associated with churn.
                2. **Targeted Policy Changes**: Practical programs to combat these top drivers (such as salary restructuring configurations and mandatory overtime tracking/caps).
                3. **Implementation Timeline**: A clear 90-day execution calendar broken into Phase 1 (Days 1-30), Phase 2 (Days 31-60), and Phase 3 (Days 61-90).
                4. **C-Suite KPI Dashboard**: Define 3 precise formulas to measure program ROI.
                """
                doc_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=doc_prompt
                )
                st.info("📄 Policy Draft Compiled Successfully! Review and copy the corporate blueprint below:")
                st.markdown(doc_response.text)
                
                st.download_button(
                    label="📥 Download Policy Document (.md)",
                    data=doc_response.text,
                    file_name="Enterprise_Retention_Framework_Blueprint.md",
                    mime="text/markdown"
                )
                st.markdown("---")
            except Exception as doc_err:
                st.error(f"Could not build document: {doc_err}")

    st.subheader("💬 Interactive Operational Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI People Strategy consultant. Ask me anything about mitigating our core risk vectors (Salary, Overtime, Commute, or Tenure Paths)."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me for specific organizational retention strategies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing workforce context vectors..."):
                try:
                    contents_payload = []
                    for msg in st.session_state.messages:
                        contents_payload.append(msg["content"])
                        
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"System Context: {system_instruction}\n\nConversation:\n" + "\n".join(contents_payload)
                    )
                    full_response = response.text
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"API Request Fault: {e}"
                    message_placeholder.markdown(full_response)
                    
        st.session_state.messages.append({"role": "assistant", "content": full_response})
