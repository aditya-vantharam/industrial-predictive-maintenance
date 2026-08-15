import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Machine Reliability Monitor",
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model_path = Path(__file__).parent / "models" / "xgboost_predictive_maintenance.pkl"

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error("Model could not be loaded.")
    st.write(e)
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("🏭 Machine Reliability Monitor")

st.write(
    "A machine-learning based system for evaluating "
    "industrial operating conditions and estimating failure risk."
)

st.divider()


# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("⚙️ Machine Inputs")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1500,
    step=10
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.5
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

temperature_difference = (
    process_temperature - air_temperature
)


# =========================================================
# CURRENT CONDITIONS
# =========================================================

st.header("Current Operating Conditions")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Machine",
        machine_type
    )

with c2:
    st.metric(
        "Air Temperature",
        f"{air_temperature:.1f} K"
    )

with c3:
    st.metric(
        "Process Temperature",
        f"{process_temperature:.1f} K"
    )

with c4:
    st.metric(
        "Temperature Difference",
        f"{temperature_difference:.1f} K"
    )


st.divider()


# =========================================================
# OPERATING PARAMETERS
# =========================================================

st.header("Operating Parameters")

p1, p2, p3 = st.columns(3)

with p1:
    st.metric(
        "Rotational Speed",
        f"{rotational_speed} rpm"
    )

with p2:
    st.metric(
        "Torque",
        f"{torque:.1f} Nm"
    )

with p3:
    st.metric(
        "Tool Wear",
        f"{tool_wear} min"
    )


# =========================================================
# QUICK CONDITION CHECK
# =========================================================

st.subheader("Parameter Check")

warnings = []

if rotational_speed > 2200:
    warnings.append("High rotational speed")

if torque > 65:
    warnings.append("High torque")

if tool_wear > 200:
    warnings.append("High tool wear")

if temperature_difference > 30:
    warnings.append("Large temperature difference")


if warnings:

    for warning in warnings:
        st.warning("⚠️ " + warning)

else:

    st.success(
        "✓ No major parameter warning detected."
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

st.header("Failure Risk Assessment")

if st.button(
    "Run Failure Prediction",
    type="primary",
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

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

    except Exception as e:

        st.error("Prediction failed.")
        st.write(e)
        st.stop()


    # =====================================================
    # HEALTH SCORE
    # =====================================================

    health = (1 - probability) * 100

    health = max(
        0,
        min(100, health)
    )


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability >= 0.70:

        risk_level = "CRITICAL"

    elif probability >= 0.30:

        risk_level = "MODERATE"

    else:

        risk_level = "LOW"


    # =====================================================
    # RESULT METRICS
    # =====================================================

    st.subheader("Prediction Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )

    with r2:
        st.metric(
            "Machine Health",
            f"{health:.0f}%"
        )

    with r3:
        st.metric(
            "Risk Classification",
            risk_level
        )


    # =====================================================
    # STATUS
    # =====================================================

    if prediction == 1:

        st.error(
            "⚠️ FAILURE RISK DETECTED"
        )

        st.write(
            "The model predicts that the current operating "
            "conditions may be associated with machine failure."
        )

    else:

        st.success(
            "✅ MACHINE OPERATING NORMALLY"
        )

        st.write(
            "The model does not currently detect a high "
            "failure risk under the entered conditions."
        )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.subheader("Maintenance Recommendation")

    if probability >= 0.70:

        st.error(
            "Immediate inspection is recommended. "
            "Check machine load, rotational speed, "
            "temperature and tool condition."
        )

    elif probability >= 0.30:

        st.warning(
            "Increase monitoring frequency and consider "
            "preventive maintenance."
        )

    else:

        st.info(
            "Continue normal monitoring and follow the "
            "planned preventive-maintenance schedule."
        )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

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
            "Machine Health",
            "Risk Classification"
        ],

        "Value": [
            machine_type,
            f"{air_temperature:.1f} K",
            f"{process_temperature:.1f} K",
            f"{temperature_difference:.1f} K",
            f"{rotational_speed} rpm",
            f"{torque:.1f} Nm",
            f"{tool_wear} min",
            f"{probability * 100:.2f}%",
            f"{health:.0f}%",
            risk_level
        ]
    })

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Machine Reliability Monitor | XGBoost Predictive Maintenance"
)
