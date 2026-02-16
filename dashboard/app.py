import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

from fraud_detection.data.loader import DataHandler
from fraud_detection.explainability.feature_importance import get_builtin_feature_importance
from fraud_detection.explainability.predictions import build_prediction_frame, sample_error_cases
from fraud_detection.explainability.shap_explainer import compute_shap_values, explain_single_prediction, get_shap_importance
from fraud_detection.utils.project_root import get_project_root


# =========================================================
# 🎨 PROFESSIONAL THEME CONFIG
# =========================================================
st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}

.section-header {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 10px;
}

.subtitle {
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 📦 RESOURCE LOADING
# =========================================================
@st.cache_resource
def load_model():
    path = get_project_root() / "models" / "ecommerce" / "xgboost_best_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = DataHandler.from_registry(
        "DATA", "processed_dir", "test_original.parquet"
    ).load()
    return df


@st.cache_resource
def get_cached_shap(_model, _X):
    return compute_shap_values(_model, _X)


model = load_model()
df_test = load_data()

TARGET = "class"
X_test = df_test.drop(columns=[TARGET])
y_test = df_test[TARGET]


# =========================================================
# 🛡️ HEADER
# =========================================================
st.title("🛡️ Fraud Detection Intelligence Platform")
st.markdown(
    "<div class='subtitle'>Explainable AI system for transaction risk monitoring and fraud analytics</div>",
    unsafe_allow_html=True
)
st.markdown("---")


# =========================================================
# 📚 SIDEBAR NAVIGATION
# =========================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Feature Importance",
        "SHAP Global Analysis",
        "Error Case Investigation",
        "Single Transaction Explorer",
    ]
)


# =========================================================
# 1️⃣ OVERVIEW
# =========================================================
if page == "Overview":

    st.markdown("<div class='section-header'>Model Overview</div>", unsafe_allow_html=True)

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5)

    y_proba = model.predict_proba(X_test)[:, 1]
    df_pred = build_prediction_frame(X_test, y_test, y_proba, threshold)

    fraud_rate = df_pred["y_pred"].mean()
    total_tx = len(df_pred)
    detected = int(df_pred["y_pred"].sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud Rate (%)", f"{fraud_rate*100:.2f}")
    col2.metric("Total Transactions", f"{total_tx:,}")
    col3.metric("Detected Fraud", detected)

    st.markdown("---")
    st.info("""
    **Model Details**
    - Algorithm: XGBoost Classifier
    - Objective: Binary Fraud Detection
    - Imbalance Handling: SMOTE
    - Evaluation Focus: Precision, Recall, F1-Score
    """)


# =========================================================
# 2️⃣ FEATURE IMPORTANCE
# =========================================================
elif page == "Feature Importance":

    st.markdown("<div class='section-header'>Model Feature Importance</div>", unsafe_allow_html=True)

    importance_df = get_builtin_feature_importance(
        model,
        feature_names=X_test.columns,
        top_n=12
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.invert_yaxis()
    ax.set_title("Top 12 Features (Model Weight)")
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)
    st.dataframe(importance_df, use_container_width=True)


# =========================================================
# 3️⃣ SHAP GLOBAL ANALYSIS
# =========================================================
elif page == "SHAP Global Analysis":

    st.markdown("<div class='section-header'>SHAP Global Feature Impact</div>", unsafe_allow_html=True)
    st.write("Measures average contribution of each feature to fraud probability.")

    with st.spinner("Computing SHAP values..."):
        shap_values, explainer = get_cached_shap(model, X_test)

    shap_importance = get_shap_importance(X_test, shap_values, top_n=10)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(shap_importance["feature"], shap_importance["shap_importance"])
    ax.invert_yaxis()
    ax.set_title("Top 10 SHAP Impact")
    ax.set_xlabel("Mean |SHAP Value|")

    st.pyplot(fig)


# =========================================================
# 4️⃣ ERROR CASE INVESTIGATION
# =========================================================
elif page == "Error Case Investigation":

    st.markdown("<div class='section-header'>Model Error Investigation</div>", unsafe_allow_html=True)

    threshold = st.slider("Threshold", 0.1, 0.9, 0.5)

    y_proba = model.predict_proba(X_test)[:, 1]
    df_pred = build_prediction_frame(X_test, y_test, y_proba, threshold)

    cases = sample_error_cases(df_pred)
    case_type = st.selectbox(
        "Select Case Type",
        ["true_positive", "false_positive", "false_negative"]
    )

    selected_case = cases[case_type]

    st.subheader("Selected Case Data")
    st.table(selected_case.to_frame().T)

    with st.spinner("Generating SHAP explanation..."):
        shap_values, explainer = get_cached_shap(model, X_test)
        explain_single_prediction(explainer, shap_values, X_test, selected_case.name)
        st.pyplot(plt.gcf())


# =========================================================
# 5️⃣ SINGLE TRANSACTION EXPLORER
# =========================================================
elif page == "Single Transaction Explorer":

    st.markdown("<div class='section-header'>Custom Transaction Risk Analyzer</div>", unsafe_allow_html=True)

    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(X_test.columns):
        container = cols[i % 3]
        input_data[col] = container.number_input(
            col,
            value=float(X_test[col].mean()),
            format="%.4f"
        )

    input_df = pd.DataFrame([input_data])

    if st.button("Analyze Transaction", type="primary"):

        proba = model.predict_proba(input_df)[0][1]

        if proba >= 0.5:
            st.error(f"🚨 Fraud Risk Detected | Probability: {proba:.4f}")
        else:
            st.success(f"✅ Legitimate Transaction | Probability: {proba:.4f}")

        shap_values_single, explainer_single = compute_shap_values(model, input_df)

        st.subheader("Decision Breakdown")

        sv = shap_values_single[1][0] if isinstance(shap_values_single, list) else shap_values_single[0]

        shap.force_plot(
            explainer_single.expected_value,
            sv,
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )

        st.pyplot(plt.gcf())
