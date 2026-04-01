import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    main_model = CatBoostClassifier()
    main_model.load_model("catboost_model.cbm")

    ssp_model = CatBoostClassifier()
    ssp_model.load_model("ssp_model.cbm")

    return main_model, ssp_model

model, ssp_model = load_models()

# -----------------------------
# LOAD DATA (for schema)
# -----------------------------
df = pd.read_csv("fertilizer_recommendation.csv")
train_cols = df.drop(columns=["Recommended_Fertilizer"]).columns

cat_cols = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage",
    "Season", "Irrigation_Type", "Previous_Crop",
    "Region"
]

fert_mapping = {
    "Urea": 0,
    "DAP": 1,
    "MOP": 2,
    "NPK": 3,
    "Compost": 4,
    "Zinc Sulphate": 5,
    "SSP": 6
}

# -----------------------------
# UI TITLE
# -----------------------------
st.title("🌱 Fertilizer Recommendation System")
st.write("Confidence-Aware Hybrid Recommendation System")

# -----------------------------
# INPUT FORM
# -----------------------------
st.sidebar.header("Input Parameters")

def user_input():
    data = {
        "Soil_Type": st.sidebar.selectbox("Soil Type", df["Soil_Type"].unique()),
        "Soil_pH": st.sidebar.slider("Soil pH", 4.0, 9.0, 7.0),
        "Soil_Moisture": st.sidebar.slider("Soil Moisture", 10, 50, 25),
        "Organic_Carbon": st.sidebar.slider("Organic Carbon", 0.1, 1.5, 0.5),
        "Electrical_Conductivity": st.sidebar.slider("EC", 0.1, 1.0, 0.3),
        "Nitrogen_Level": st.sidebar.slider("Nitrogen", 0, 150, 50),
        "Phosphorus_Level": st.sidebar.slider("Phosphorus", 0, 100, 30),
        "Potassium_Level": st.sidebar.slider("Potassium", 0, 150, 50),
        "Temperature": st.sidebar.slider("Temperature", 10, 40, 25),
        "Humidity": st.sidebar.slider("Humidity", 20, 100, 60),
        "Rainfall": st.sidebar.slider("Rainfall", 0, 300, 100),
        "Crop_Type": st.sidebar.selectbox("Crop Type", df["Crop_Type"].unique()),
        "Crop_Growth_Stage": st.sidebar.selectbox("Growth Stage", df["Crop_Growth_Stage"].unique()),
        "Season": st.sidebar.selectbox("Season", df["Season"].unique()),
        "Irrigation_Type": st.sidebar.selectbox("Irrigation", df["Irrigation_Type"].unique()),
        "Previous_Crop": st.sidebar.selectbox("Previous Crop", df["Previous_Crop"].unique()),
        "Region": st.sidebar.selectbox("Region", df["Region"].unique()),
        "Fertilizer_Used_Last_Season": st.sidebar.selectbox("Last Fertilizer", list(fert_mapping.keys())),
        "Yield_Last_Season": st.sidebar.slider("Last Yield", 1.0, 10.0, 3.5)
    }
    return pd.DataFrame([data])

input_df = user_input()

# -----------------------------
# PREPROCESS INPUT
# -----------------------------
def preprocess_input(input_df):
    for col in train_cols:
        if col not in input_df.columns:
            input_df[col] = df[col].iloc[0]

    input_df = input_df[train_cols]

    for col in cat_cols:
        input_df[col] = input_df[col].astype(str)

    input_df["Fertilizer_Used_Last_Season"] = input_df[
        "Fertilizer_Used_Last_Season"
    ].map(fert_mapping)

    return input_df

input_df = preprocess_input(input_df)

cat_indices = [input_df.columns.get_loc(col) for col in cat_cols]
pool = Pool(input_df, cat_features=cat_indices)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Get Recommendation"):

    probs = model.predict_proba(pool)[0]
    classes = model.classes_

    top1_idx = np.argmax(probs)
    top1 = classes[top1_idx]
    confidence = probs[top1_idx]

    # SSP prediction
    ssp_prob = ssp_model.predict_proba(pool)[0][1]

    # -----------------------------
    # DECISION LOGIC
    # -----------------------------
    if ssp_prob > 0.5:
        final_label = "SSP"
        status = "SSP OVERRIDE"

    elif confidence < 0.6:
        final_label = "UNCERTAIN"
        status = "LOW CONFIDENCE"

    else:
        final_label = top1
        status = "CONFIDENT"

    # -----------------------------
    # TOP-2
    # -----------------------------
    top2_idx = np.argsort(probs)[-2:][::-1]
    top2 = [(classes[i], probs[i]) for i in top2_idx]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("🌾 Recommendation")

    if final_label == "UNCERTAIN":
        st.warning("⚠️ Low Confidence - Recommend soil testing")
    else:
        st.success(f"Primary: {final_label}")

    st.write(f"Confidence: {confidence:.3f}")
    st.write(f"SSP Confidence: {ssp_prob:.3f}")
    st.write(f"Status: {status}")

    st.subheader("Top-2 Suggestions")
    for fert, prob in top2:
        st.write(f"{fert} ({prob:.3f})")

    # -----------------------------
    # CONFIDENCE GRAPH
    # -----------------------------
    st.subheader("Confidence Distribution")

    fig, ax = plt.subplots()
    ax.bar(classes, probs)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -----------------------------
    # SHAP MODULE (COMPLETE)
    # -----------------------------
    st.subheader("🧠 Model Explanation (SHAP)")

    def get_shap_values(model, input_df, classes, predicted_class):
        explainer = shap.TreeExplainer(model)

        X = input_df.iloc[[0]]  # single row
        shap_values = explainer.shap_values(X)

        class_idx = int(np.where(classes == predicted_class)[0][0])

        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[class_idx])[0]

        else:
            shap_arr = np.array(shap_values)

            if shap_arr.ndim == 2:
                shap_vals = shap_arr[0]

            elif shap_arr.ndim == 3:
                if shap_arr.shape[0] == 1:
                    shap_vals = shap_arr[0, :, class_idx]
                elif shap_arr.shape[0] == len(classes):
                    shap_vals = shap_arr[class_idx, 0, :]
                else:
                    raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")
            else:
                raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")

        return X, shap_vals


    def explain_prediction(input_df, shap_vals, final_label):
        numerical_cols = [
            "Soil_pH", "Soil_Moisture", "Organic_Carbon",
            "Electrical_Conductivity", "Nitrogen_Level",
            "Phosphorus_Level", "Potassium_Level",
            "Temperature", "Humidity", "Rainfall",
            "Yield_Last_Season"
        ]

        feature_imp = list(zip(input_df.columns, shap_vals))
        feature_imp = sorted(feature_imp, key=lambda x: abs(x[1]), reverse=True)[:6]

        st.subheader("🔍 Key Factors")

        explanations = []

        for feature, value in feature_imp:
            actual = input_df.iloc[0][feature]

            if feature in numerical_cols:
                if value > 0:
                    text = f"{feature} ({actual}) increases likelihood of {final_label}"
                else:
                    text = f"{feature} ({actual}) decreases likelihood of {final_label}"

            else:
                text = f"{feature} = '{actual}' influenced the prediction"

            explanations.append(text)
            st.write("• " + text)

        return feature_imp


    def plot_shap_bar(feature_imp):
        names = [f[0] for f in feature_imp]
        values = [abs(f[1]) for f in feature_imp]

        fig, ax = plt.subplots()
        ax.barh(names[::-1], values[::-1])
        ax.set_title("Top Feature Contributions")
        ax.set_xlabel("Impact (|SHAP value|)")
        st.pyplot(fig)


    # -----------------------------
    # EXECUTION
    # -----------------------------
    try:
        X_explain, shap_vals = get_shap_values(model, input_df, classes, top1)

        feature_imp = explain_prediction(X_explain, shap_vals, final_label)

        st.subheader("📊 Feature Impact")
        plot_shap_bar(feature_imp)

    except Exception as e:
        st.warning(f"⚠️ SHAP failed: {e}")