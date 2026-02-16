import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, auc, confusion_matrix

from fraud_detection.data.loader import DataHandler
from fraud_detection.explainability.feature_importance import get_builtin_feature_importance
from fraud_detection.explainability.predictions import build_prediction_frame, sample_error_cases
from fraud_detection.explainability.shap_explainer import compute_shap_values, explain_single_prediction, get_shap_importance
from fraud_detection.utils.project_root import get_project_root


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Fraud Risk Intelligence Terminal",
    page_icon="🛡️",
    layout="wide",
)

# =========================================================
# THEME TOGGLE
# =========================================================
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True)

if dark_mode:
    background = "#0f172a"
    card_bg = "#111827"
    accent = "#00ff9f"
    text_color = "#f8fafc"
else:
    background = "#f8fafc"
    card_bg = "#ffffff"
    accent = "#2563eb"
    text_color = "#111827"

st.markdown(f"""
<style>

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}}

.metric-card {{
    background: {card_bg};
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.15);
    text-align: center;
}}

.section-card {{
    background: {card_bg};
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.18);
    margin-bottom: 20px;
}}

hr {{
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, {accent}, transparent);
}}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    path = get_project_root() / "models" / "ecommerce" / "xgboost_best_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return DataHandler.from_registry(
        "DATA", "processed_dir", "test_original.parquet"
    ).load()


@st.cache_resource
def cached_shap(_model, _X):
    return compute_shap_values(_model, _X)


model = load_model()
df_test = load_data()

TARGET = "class"
X_test = df_test.drop(columns=[TARGET])
y_test = df_test[TARGET]


# =========================================================
# HEADER
# =========================================================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80
    )

with col_title:
    st.markdown("## FRAUD RISK INTELLIGENCE TERMINAL")
    st.caption("Institutional-grade Explainable AI Risk Monitoring")

st.markdown("---")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Explainability",
        "Error Investigation",
        "Transaction Explorer",
    ]
)


# =========================================================
# OVERVIEW PAGE
# =========================================================
if page == "Overview":

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5)

    y_proba = model.predict_proba(X_test)[:, 1]
    df_pred = build_prediction_frame(X_test, y_test, y_proba, threshold)

    fraud_rate = df_pred["y_pred"].mean()
    detected = int(df_pred["y_pred"].sum())

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    # KPI STRIP
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("AUC", f"{roc_auc:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Rate", f"{fraud_rate*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with k3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Detected Fraud", detected)
        st.markdown('</div>', unsafe_allow_html=True)

    with k4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Threshold", threshold)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("ROC Curve")

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, linewidth=2, color=accent)
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Fraud Probability Distribution")

    fig, ax = plt.subplots()
    ax.hist(y_proba, bins=40, alpha=0.85)
    ax.axvline(threshold, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# MODEL EXPLAINABILITY
# =========================================================
elif page == "Model Explainability":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Global SHAP Feature Impact")

    with st.spinner("Computing SHAP values..."):
        shap_values, explainer = cached_shap(model, X_test)

    shap_importance = get_shap_importance(X_test, shap_values, top_n=10)

    fig, ax = plt.subplots()
    ax.barh(
        shap_importance["feature"],
        shap_importance["shap_importance"],
        color=accent
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Most Influential Features")

    st.pyplot(fig)
    st.dataframe(shap_importance, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ERROR INVESTIGATION
# =========================================================
elif page == "Error Investigation":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    threshold = st.slider("Threshold", 0.1, 0.9, 0.5)

    y_proba = model.predict_proba(X_test)[:, 1]
    df_pred = build_prediction_frame(X_test, y_test, y_proba, threshold)

    cases = sample_error_cases(df_pred)

    case_type = st.selectbox(
        "Select Case Type",
        ["true_positive", "false_positive", "false_negative"]
    )

    selected_case = cases[case_type]

    st.subheader("Transaction Details")
    st.dataframe(selected_case.to_frame().T, use_container_width=True)

    shap_values, explainer = cached_shap(model, X_test)

    st.subheader("SHAP Local Explanation")

    explain_single_prediction(
        explainer,
        shap_values,
        X_test,
        selected_case.name
    )

    st.pyplot(plt.gcf())

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TRANSACTION EXPLORER
# =========================================================
elif page == "Transaction Explorer":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Custom Transaction Risk Analyzer")

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

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba * 100,
            number={'suffix': "%"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': accent},
                'steps': [
                    {'range': [0, 40], 'color': "#22c55e"},
                    {'range': [40, 70], 'color': "#facc15"},
                    {'range': [70, 100], 'color': "#ef4444"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
        ))

        st.plotly_chart(fig, use_container_width=True)

        if proba >= 0.5:
            st.error("🚨 HIGH FRAUD RISK DETECTED")
        else:
            st.success("✅ LOW FRAUD RISK")

        shap_values_single, explainer_single = compute_shap_values(model, input_df)

        sv = shap_values_single[1][0] if isinstance(shap_values_single, list) else shap_values_single[0]

        shap.force_plot(
            explainer_single.expected_value,
            sv,
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )

        st.pyplot(plt.gcf())

    st.markdown('</div>', unsafe_allow_html=True)
