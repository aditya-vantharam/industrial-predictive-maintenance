import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #f5f5f5;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262a33;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #b8bec9;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Metric cards */
    .metric-card {
        background-color: #171b24;
        border: 1px solid #303642;
        border-radius: 10px;
        padding: 20px;
        min-height: 120px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 600;
    }

    /* Status cards */
    .normal-box {
        background-color: #123d2b;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #22c55e;
        margin: 15px 0;
    }

    .critical-box {
        background-color: #442022;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #ef4444;
        margin: 15px 0;
    }

    .warning-box {
        background-color: #40351a;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #eab308;
        margin: 15px 0;
    }

    /* Recommendation */
    .recommendation {
        background-color: #163452;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #3b82f6;
        font-size: 17px;
    }

    /* Risk bar */
    .risk-container {
        background-color: #252b35;
        border-radius: 10px;
        height: 25px;
        width: 100%;
        overflow: hidden;
    }

    .risk-bar {
        height: 100%;
        border-radius: 10px;
    }

    /* Alert */
    .alert {
        background-color: #332d18;
        border-radius: 8px;
        padding: 14px;
        margin: 8px 0;
        border-left: 4px solid #eab308;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777f8c;
        padding: 30px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

model_path = Path(__file__).parent / "models" / "xgboost_predictive_maintenance.pkl"

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Machine Parameters")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.slider(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.slider(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.slider(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1500,
    step=10
)

torque = st.sidebar.slider(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.5
)

tool_wear = st.sidebar.slider(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)

temperature_difference = (
    process_temperature - air_temperature
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚙️ Industrial Predictive Maintenance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered machine failure prediction and maintenance monitoring dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# MACHINE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">🏭 Machine Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Machine Type</div>
            <div class="metric-value">{machine_type}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Air Temperature</div>
            <div class="metric-value">{air_temperature:.1f} K</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Process Temperature</div>
            <div class="metric-value">{process_temperature:.1f} K</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Temperature Difference</div>
            <div class="metric-value">{temperature_difference:.1f} K</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OPERATING PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Operating Parameters</div>',
    unsafe_allow_html=True
)

parameters = pd.DataFrame({
    "Parameter": [
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],
    "Value": [
        f"{air_temperature:.1f} K",
        f"{process_temperature:.1f} K",
        f"{rotational_speed} rpm",
        f"{torque:.1f} Nm",
        f"{tool_wear} min"
    ]
})

st.dataframe(
    parameters,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Machine Failure Analysis</div>',
    unsafe_allow_html=True
)

predict = st.button(
    "🔍 Predict Machine Failure",
    use_container_width=True
)

if predict:

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Temperature Difference [K]": [temperature_difference]
    })

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:
        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()

    risk_percentage = probability * 100
    health_percentage = 100 - risk_percentage


    # ========================================================
    # RESULT CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Failure Probability",
            f"{risk_percentage:.2f}%"
        )

    with col2:
        st.metric(
            "Machine Health",
            f"{health_percentage:.0f}%"
        )

    with col3:

        if prediction == 1:
            st.metric(
                "Machine Status",
                "🔴 Critical"
            )
        else:
            st.metric(
                "Machine Status",
                "🟢 Normal"
            )


    # ========================================================
    # STATUS
    # ========================================================

    if prediction == 1:

        st.markdown(
            """
            <div class="critical-box">
                <h3>🚨 FAILURE RISK DETECTED</h3>
                <p>
                The predictive model indicates that the machine
                may be at risk of failure.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Alerts based on operating conditions

        if tool_wear >= 200:
            st.markdown(
                """
                <div class="alert">
                    🔧 <b>High tool wear detected.</b>
                    Consider inspecting or replacing the tool.
                </div>
                """,
                unsafe_allow_html=True
            )

        if torque >= 60:
            st.markdown(
                """
                <div class="alert">
                    ⚠️ <b>High torque detected.</b>
                    Inspect the machine for abnormal mechanical load.
                </div>
                """,
                unsafe_allow_html=True
            )

        if rotational_speed >= 2200:
            st.markdown(
                """
                <div class="alert">
                    ⚠️ <b>High rotational speed detected.</b>
                    Check operating conditions and machine stress.
                </div>
                """,
                unsafe_allow_html=True
            )

        if temperature_difference >= 25:
            st.markdown(
                """
                <div class="alert">
                    🌡️ <b>High temperature difference detected.</b>
                    Check cooling and thermal operating conditions.
                </div>
                """,
                unsafe_allow_html=True
            )

        # Recommendation

        st.markdown(
            '<div class="section-title">🛠️ Maintenance Recommendation</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="recommendation">
            Schedule a maintenance inspection as soon as practical
            and investigate the operating parameters contributing
            to the predicted failure risk.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="normal-box">
                <h3>✅ MACHINE OPERATING NORMALLY</h3>
                <p>
                The model does not currently detect a high failure
                risk under the entered operating conditions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🛠️ Maintenance Recommendation</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="recommendation">
            Continue normal monitoring and follow the planned
            preventive-maintenance schedule.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FAILURE RISK VISUALIZATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Failure Risk Visualization</div>',
        unsafe_allow_html=True
    )

    if risk_percentage >= 70:
        bar_color = "#ef4444"
    elif risk_percentage >= 30:
        bar_color = "#eab308"
    else:
        bar_color = "#22c55e"

    st.markdown(
        f"""
        <p><b>Failure Risk: {risk_percentage:.2f}%</b></p>

        <div class="risk-container">
            <div class="risk-bar"
                 style="width:{risk_percentage}%;
                        background-color:{bar_color};">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Prediction Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(f"**Machine Type:** {machine_type}")
        st.write(f"**Air Temperature:** {air_temperature:.1f} K")
        st.write(f"**Process Temperature:** {process_temperature:.1f} K")
        st.write(
            f"**Temperature Difference:** "
            f"{temperature_difference:.1f} K"
        )

    with col2:

        st.write(
            f"**Rotational Speed:** {rotational_speed} rpm"
        )

        st.write(
            f"**Torque:** {torque:.1f} Nm"
        )

        st.write(
            f"**Tool Wear:** {tool_wear} min"
        )

        st.write(
            f"**Predicted Failure Probability:** "
            f"{risk_percentage:.2f}%"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ⚙️ Industrial Predictive Maintenance |
        AI-based decision-support system
    </div>
    """,
    unsafe_allow_html=True
)
