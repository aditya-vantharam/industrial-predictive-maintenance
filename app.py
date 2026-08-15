import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Machine Failure Predictor",
    page_icon="🔧",
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
    st.error("Unable to load the prediction model.")
    st.write(e)
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🔧 Machine Failure Predictor")

st.write(
    "Evaluate industrial machine conditions and estimate "
    "the likelihood of equipment failure."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Machine Configuration")

col1, col2 = st.columns(2)

with col1:

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


with col2:

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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

temperature_difference = (
    process_temperature - air_temperature
)


# ============================================================
# QUICK CONDITION SUMMARY
# ============================================================

st.divider()

st.subheader("Condition Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.write("**Machine Type**")
    st.write(f"Type {machine_type}")

with c2:
    st.write("**Temperature Difference**")
    st.write(f"{temperature_difference:.1f} K")

with c3:
    st.write("**Tool Wear**")
    st.write(f"{tool_wear} min")


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Analyze Machine",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

    except Exception as e:

        st.error("Prediction failed.")
        st.write(e)
        st.stop()


    failure_percent = probability * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("Prediction")

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:

        st.metric(
            "Failure Probability",
            f"{failure_percent:.2f}%"
        )

    with result_col2:

        if prediction == 1:

            st.error(
                "⚠️ FAILURE RISK DETECTED"
            )

        else:

            st.success(
                "✅ MACHINE OPERATING NORMALLY"
            )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if failure_percent >= 70:

        risk = "Critical"
        recommendation = (
            "Inspect the machine before continuing "
            "normal operation."
        )

    elif failure_percent >= 30:

        risk = "Moderate"
        recommendation = (
            "Increase monitoring and consider "
            "preventive maintenance."
        )

    else:

        risk = "Low"
        recommendation = (
            "Continue normal operation and follow "
            "the scheduled maintenance plan."
        )


    st.write(
        f"**Risk classification:** {risk}"
    )

    st.info(
        f"**Recommended action:** {recommendation}"
    )


    # ========================================================
    # NEW FEATURE: CONDITION INDICATORS
    # ========================================================

    st.divider()

    st.subheader("Operating Condition Indicators")

    indicators = []

    # Temperature
    if temperature_difference >= 25:
        indicators.append(
            ("Temperature difference", "Attention")
        )
    else:
        indicators.append(
            ("Temperature difference", "Normal")
        )

    # Speed
    if rotational_speed >= 2200:
        indicators.append(
            ("Rotational speed", "Attention")
        )
    else:
        indicators.append(
            ("Rotational speed", "Normal")
        )

    # Torque
    if torque >= 60:
        indicators.append(
            ("Torque", "Attention")
        )
    else:
        indicators.append(
            ("Torque", "Normal")
        )

    # Tool wear
    if tool_wear >= 200:
        indicators.append(
            ("Tool wear", "Attention")
        )
    else:
        indicators.append(
            ("Tool wear", "Normal")
        )


    indicator_df = pd.DataFrame(
        indicators,
        columns=["Parameter", "Condition"]
    )

    st.dataframe(
        indicator_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # NEW FEATURE: SIMPLE RISK GAUGE
    # ========================================================

    st.subheader("Failure Risk")

    st.progress(
        min(probability, 1.0),
        text=f"Estimated failure risk: {failure_percent:.2f}%"
    )


    # ========================================================
    # NEW FEATURE: WHAT-IF ANALYSIS
    # ========================================================

    st.divider()

    st.subheader("What-if Analysis")

    st.write(
        "Change one operating parameter below to see how "
        "the model's predicted probability changes."
    )

    selected_parameter = st.selectbox(
        "Parameter to vary",
        [
            "Tool Wear",
            "Torque",
            "Rotational Speed"
        ]
    )


    if selected_parameter == "Tool Wear":

        test_value = st.slider(
            "Test Tool Wear [min]",
            0,
            300,
            int(tool_wear)
        )

        test_input = input_data.copy()

        test_input["Tool wear [min]"] = test_value


    elif selected_parameter == "Torque":

        test_value = st.slider(
            "Test Torque [Nm]",
            0.0,
            100.0,
            float(torque),
            0.5
        )

        test_input = input_data.copy()

        test_input["Torque [Nm]"] = test_value


    else:

        test_value = st.slider(
            "Test Rotational Speed [rpm]",
            500,
            3000,
            int(rotational_speed),
            10
        )

        test_input = input_data.copy()

        test_input["Rotational speed [rpm]"] = test_value


    try:

        what_if_probability = model.predict_proba(
            test_input
        )[0][1]

        what_if_percent = what_if_probability * 100

        st.metric(
            "What-if Failure Probability",
            f"{what_if_percent:.2f}%"
        )

        difference = (
            what_if_percent - failure_percent
        )

        if difference > 0:

            st.warning(
                f"Risk increased by {difference:.2f} percentage points."
            )

        elif difference < 0:

            st.success(
                f"Risk decreased by {abs(difference):.2f} percentage points."
            )

        else:

            st.info(
                "The predicted risk is unchanged."
            )

    except Exception as e:

        st.warning(
            "What-if analysis could not be calculated."
        )
        st.write(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Industrial Predictive Maintenance | "
    "XGBoost-based decision support"
)
