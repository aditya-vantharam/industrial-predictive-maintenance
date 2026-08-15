import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Machine Failure Predictor",
    page_icon="🔧",
    layout="centered"
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model_path = (
    Path(__file__).parent
    / "models"
    / "xgboost_predictive_maintenance.pkl"
)

try:
    model = joblib.load(model_path)

except Exception as e:
    st.error("Unable to load the prediction model.")
    st.stop()


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("🔧 Machine Failure Predictor")

st.write(
    "Enter the current machine operating conditions "
    "to estimate the probability of failure."
)

st.divider()


# ------------------------------------------------------------
# Machine information
# ------------------------------------------------------------

st.subheader("Machine Information")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)


# ------------------------------------------------------------
# Machine parameters
# ------------------------------------------------------------

st.subheader("Operating Conditions")

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


# ------------------------------------------------------------
# Hidden feature calculation
# ------------------------------------------------------------

temperature_difference = (
    process_temperature - air_temperature
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

st.divider()

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    # Create input for trained model
    input_data = pd.DataFrame({

        "Type": [machine_type],

        "Air temperature [K]": [
            air_temperature
        ],

        "Process temperature [K]": [
            process_temperature
        ],

        "Rotational speed [rpm]": [
            rotational_speed
        ],

        "Torque [Nm]": [
            torque
        ],

        "Tool wear [min]": [
            tool_wear
        ],

        "Temperature Difference [K]": [
            temperature_difference
        ]
    })


    # Make prediction
    try:

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

    except Exception as e:

        st.error("Prediction could not be completed.")
        st.write(e)
        st.stop()


    probability_percent = probability * 100


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    st.subheader("Prediction")

    if prediction == 1:

        st.error(
            "⚠️ Failure risk detected"
        )

        st.write(
            "The model predicts that the machine "
            "may be at risk of failure."
        )

    else:

        st.success(
            "✅ Machine operating normally"
        )

        st.write(
            "The model does not currently detect "
            "a high failure risk."
        )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    st.metric(
        "Failure Probability",
        f"{probability_percent:.2f}%"
    )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if probability_percent >= 70:

        st.warning(
            "Recommended action: Inspect the machine "
            "and perform maintenance as soon as possible."
        )

    elif probability_percent >= 30:

        st.warning(
            "Recommended action: Monitor the machine "
            "more closely and consider preventive maintenance."
        )

    else:

        st.info(
            "Recommended action: Continue normal monitoring "
            "and follow the planned maintenance schedule."
        )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Industrial Predictive Maintenance • "
    "Machine Learning Decision Support"
)
