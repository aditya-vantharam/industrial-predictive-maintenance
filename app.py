
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
model_path = Path(__file__).parent / "models" / "xgboost_predictive_maintenance.pkl"
model = joblib.load(model_path)



# -----------------------------
# Title
# -----------------------------
st.title("⚙️ Industrial Equipment Failure Prediction")
st.write(
    "Predictive maintenance system using machine operating parameters "
    "and an XGBoost machine-learning model."
)

st.divider()

# -----------------------------
# Machine inputs
# -----------------------------
st.subheader("Enter Machine Parameters")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1500,
    step=10
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.5
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)

# -----------------------------
# Feature engineering
# -----------------------------
temperature_difference = (
    process_temperature - air_temperature
)

st.info(
    f"Temperature Difference: "
    f"{temperature_difference:.2f} K"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button(
    "🔍 Predict Machine Failure",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Temperature Difference [K]": [
            temperature_difference
        ]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ FAILURE RISK DETECTED"
        )

        st.write(
            "The model predicts that the machine "
            "may be at risk of failure."
        )

    else:

        st.success(
            "✅ MACHINE OPERATING NORMALLY"
        )

        st.write(
            "The model does not currently detect "
            "a high failure risk."
        )

    st.metric(
        "Predicted Failure Probability",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This prediction is intended as a decision-support "
        "tool and should not replace professional maintenance "
        "inspection."
    )
