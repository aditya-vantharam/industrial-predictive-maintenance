import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model_path = (
    Path(__file__).parent
    / "models"
    / "xgboost_predictive_maintenance.pkl"
)

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error("Model could not be loaded.")
    st.write(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏭 Predictive Maintenance")

st.write(
    "Industrial machine failure prediction using "
    "machine-learning based condition analysis."
)

st.divider()


# ============================================================
# SIDEBAR - INPUTS
# ============================================================

st.sidebar.header("Machine Inputs")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.slider(
    "Air Temperature [K]",
    250.0,
    350.0,
    300.0,
    0.1
)

process_temperature = st.sidebar.slider(
    "Process Temperature [K]",
    250.0,
    400.0,
    310.0,
    0.1
)

rotational_speed = st.sidebar.slider(
    "Rotational Speed [rpm]",
    500,
    3000,
    1500,
    10
)

torque = st.sidebar.slider(
    "Torque [Nm]",
    0.0,
    100.0,
    40.0,
    0.5
)

tool_wear = st.sidebar.slider(
    "Tool Wear [min]",
    0,
    300,
    100,
    1
)


# ============================================================
# CALCULATE TEMPERATURE DIFFERENCE
# ============================================================

temperature_difference = (
    process_temperature - air_temperature
)


# ============================================================
# CURRENT MACHINE CONDITION
# ============================================================

st.header("Current Machine Condition")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Machine Type",
        machine_type
    )

with col2:
    st.metric(
        "Air Temperature",
        f"{air_temperature:.1f} K"
    )

with col3:
    st.metric(
        "Process Temperature",
        f"{process_temperature:.1f} K"
    )

with col4:
    st.metric(
        "Temperature Difference",
        f"{temperature_difference:.1f} K"
    )


# ============================================================
# OPERATING PARAMETERS
# ============================================================

st.divider()

st.header("Operating Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rotational Speed",
        f"{rotational_speed} rpm"
    )

with col2:
    st.metric(
        "Torque",
        f"{torque:.1f} Nm"
    )

with col3:
    st.metric(
        "Tool Wear",
        f"{tool_wear} min"
    )


# ============================================================
# PARAMETER WARNINGS
# ============================================================

st.divider()

st.header("Parameter Check")

warnings = []

if rotational_speed >= 2200:
    warnings.append("High rotational speed")

if torque >= 60:
    warnings.append("High torque")

if tool_wear >= 200:
    warnings.append("High tool wear")

if temperature_difference >= 25:
    warnings.append("High temperature difference")


if warnings:

    for warning in warnings:
        st.warning(warning)

else:

    st.success(
        "All entered parameters are within the normal monitoring range."
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

st.header("Machine Assessment")

if st.button(
    "🔍 Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

    except Exception as e:

        st.error("Prediction failed.")
        st.write(e)
        st.stop()


    # --------------------------------------------------------
    # CALCULATE HEALTH
    # --------------------------------------------------------

    failure_percentage = probability * 100

    machine_health = 100 - failure_percentage


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.subheader("Prediction Result")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Failure Probability",
            f"{failure_percentage:.2f}%"
        )

    with result2:
        st.metric(
            "Machine Health",
            f"{machine_health:.1f}%"
        )

    with result3:

        if failure_percentage >= 70:
            status = "CRITICAL"

        elif failure_percentage >= 30:
            status = "WARNING"

        else:
            status = "NORMAL"

        st.metric(
            "Machine Status",
            status
        )


    # --------------------------------------------------------
    # STATUS MESSAGE
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ FAILURE RISK DETECTED"
        )

        st.write(
            "The model predicts that the machine "
            "may be at risk of failure under the "
            "current operating conditions."
        )

    else:

        st.success(
            "✅ MACHINE OPERATING NORMALLY"
        )

        st.write(
            "The model does not currently detect "
            "a high failure risk."
        )


    # --------------------------------------------------------
    # MAINTENANCE RECOMMENDATION
    # --------------------------------------------------------

    st.subheader("Maintenance Recommendation")

    if failure_percentage >= 70:

        st.error(
            "Immediate inspection is recommended. "
            "Check tool wear, torque, rotational speed "
            "and temperature conditions."
        )

    elif failure_percentage >= 30:

        st.warning(
            "Increase monitoring frequency and "
            "consider preventive maintenance."
        )

    else:

        st.info(
            "Continue normal monitoring and follow "
            "the planned preventive-maintenance schedule."
        )


    # --------------------------------------------------------
    # RESULT TABLE
    # --------------------------------------------------------

    st.subheader("Assessment Details")

    result_table = pd.DataFrame({

        "Parameter": [
            "Machine Type",
            "Air Temperature",
            "Process Temperature",
            "Temperature Difference",
            "Rotational Speed",
            "Torque",
            "Tool Wear",
            "Failure Probability",
            "Machine Health"
        ],

        "Value": [
            machine_type,
            f"{air_temperature:.1f} K",
            f"{process_temperature:.1f} K",
            f"{temperature_difference:.1f} K",
            f"{rotational_speed} rpm",
            f"{torque:.1f} Nm",
            f"{tool_wear} min",
            f"{failure_percentage:.2f}%",
            f"{machine_health:.1f}%"
        ]
    })

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Industrial Predictive Maintenance • "
    "XGBoost Decision Support System"
)
