import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Fertilizer Recommendation System", layout="wide")

# ==============================
# LOAD ARTIFACTS
# ==============================
model = joblib.load("artifacts/final_xgb.joblib")
preproc = joblib.load("artifacts/preprocessor.joblib")
target_encoder = joblib.load("artifacts/target_encoder.joblib")
explainer = joblib.load("artifacts/shap_explainer.joblib")
feature_names = joblib.load("artifacts/feature_names.joblib")

# ==============================
# LOAD DATASET
# ==============================
@st.cache_data
def load_dataset():
    return pd.read_csv("fertilizer_recommendation.csv")

df = load_dataset()

num_cols = [
    'Soil_pH','Soil_Moisture','Organic_Carbon','Electrical_Conductivity',
    'Nitrogen_Level','Phosphorus_Level','Potassium_Level',
    'Temperature','Humidity','Rainfall',
    'Fertilizer_Used_Last_Season','Yield_Last_Season'
]

cat_cols = [
    'Soil_Type','Crop_Type','Crop_Growth_Stage',
    'Season','Irrigation_Type','Previous_Crop','Region'
]

feature_units = {
    "Soil_pH": "",
    "Soil_Moisture": "%",
    "Organic_Carbon": "%",
    "Electrical_Conductivity": "dS/m",
    "Nitrogen_Level": "kg/ha",
    "Phosphorus_Level": "kg/ha",
    "Potassium_Level": "kg/ha",
    "Temperature": "°C",
    "Humidity": "%",
    "Rainfall": "mm/year",
    "Fertilizer_Used_Last_Season": "kg/ha",
    "Yield_Last_Season": "ton/ha"
}

# ==============================
# RANGE FUNCTION
# ==============================
def get_feature_range(col):
    return {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "mean": float(df[col].mean())
    }

# ==============================
# CONFIDENCE ENGINE
# ==============================
def compute_decision_confidence(conf1, conf2):
    margin = conf1 - conf2
    score = (0.7 * conf1) + (0.3 * margin)
    score_100 = int(score * 100)

    if score >= 0.65:
        label = "HIGH"
        color = "success"
    elif score >= 0.45:
        label = "MODERATE"
        color = "warning"
    else:
        label = "LOW"
        color = "error"

    return score_100, label, color, margin

# ==============================
# GAUGE
# ==============================
def draw_confidence_gauge(score):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.barh([0], [score], color="#27ae60")
    ax.barh([0], [100-score], left=[score], color="#ecf0f1")
    ax.set_xlim(0,100)
    ax.set_yticks([])
    ax.set_title("Decision Confidence Gauge")
    ax.text(score, 0, f"{score}/100", va="center", ha="right",
            fontsize=12, fontweight="bold")
    st.pyplot(fig)

# ==============================
# TITLE
# ==============================
st.title("🌱 Explainable AI Fertilizer Recommendation System")
st.caption("Data-driven, Site-Specific Fertilizer Decision Support")

left, right = st.columns([1,2])
inputs = {}

# ==============================
# INPUT PANEL
# ==============================
with left:
    st.header("🧾 Farm Conditions Input")

    for col in num_cols:
        r = get_feature_range(col)
        label = f"{col.replace('_',' ')} ({feature_units[col]})"

        inputs[col] = st.number_input(
            label,
            min_value=r["min"],
            max_value=r["max"],
            value=r["mean"],
            step=(r["max"] - r["min"]) / 100
        )

    st.subheader("🌾 Farm Context")

    for col in cat_cols:
        inputs[col] = st.selectbox(
            col.replace('_',' '),
            sorted(df[col].dropna().unique().tolist())
        )

# ==============================
# PREDICTION
# ==============================
if st.button("🚀 Generate Recommendation"):

    df_input = pd.DataFrame([inputs])
    Xp = preproc.transform(df_input)

    probs = model.predict_proba(Xp)[0]
    top2_idx = np.argsort(probs)[::-1][:2]

    fert1 = target_encoder.inverse_transform([top2_idx[0]])[0]
    fert2 = target_encoder.inverse_transform([top2_idx[1]])[0]

    conf1 = probs[top2_idx[0]]
    conf2 = probs[top2_idx[1]]

    score, level, color, margin = compute_decision_confidence(conf1, conf2)

    with right:
        st.header("🎯 Recommendation Result")

        draw_confidence_gauge(score)

        if color == "success":
            st.success(f"{level} Decision Confidence")
        elif color == "warning":
            st.warning(f"{level} Decision Confidence")
        else:
            st.error(f"{level} Decision Confidence")

        c1,c2 = st.columns(2)

        with c1:
            st.metric("Primary Fertilizer", fert1,
                      f"{conf1*100:.1f}% model probability")

        with c2:
            st.metric("Alternative Option", fert2,
                      f"{conf2*100:.1f}% model probability")

        if margin > 0.25:
            st.info("Stable recommendation.")
        elif margin > 0.10:
            st.info("Balanced recommendation.")
        else:
            st.info("Ambiguous conditions — advisory mode.")

        # ==============================
        # SHAP EXPLANATION
        # ==============================
        st.subheader("🧠 Explainable AI Insights")

        shap_exp = explainer(Xp)
        shap_vals = shap_exp.values[0, :, top2_idx[0]]

        fig = plt.figure()
        shap.summary_plot(
            shap_vals.reshape(1, -1),
            features=Xp,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig)

        # ==============================
        # ⭐ NEW REASONING TEXT FEATURE
        # ==============================
        st.subheader("📌 Model Reasoning")

        contrib = list(zip(feature_names, shap_vals))

        # pick top impactful features
        contrib_sorted = sorted(
            contrib,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        reasoning_lines = []

        for feature, value in contrib_sorted:
            clean_name = feature.replace("_"," ")

            if value > 0:
                reasoning_lines.append(
                    f"• {clean_name} increased suitability for {fert1}"
                )
            else:
                reasoning_lines.append(
                    f"• {clean_name} reduced suitability for {fert1}"
                )

        st.success("\n".join(reasoning_lines))
