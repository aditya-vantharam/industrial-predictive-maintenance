import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MachineGuard | Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "xgboost_predictive_maintenance.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("The predictive model could not be loaded.")
    st.exception(e)
    st.stop()

# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>

    /* ---------- Global ---------- */

    .stApp {
        background: #f4f6f8;
        color: #17202a;
    }

    [data-testid="stHeader"] {
        background: #f4f6f8;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #17202a;
        border-right: 1px solid #263544;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ---------- Header ---------- */

    .brand {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #52758c;
        text-transform: uppercase;
    }

    .page-title {
        font-size: 42px;
        font-weight: 800;
        color: #17202a;
        margin-top: 5px;
        margin-bottom: 0;
    }

    .page-subtitle {
        color: #68737d;
        font-size: 16px;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    /* ---------- Panels ---------- */

    .panel {
        background: white;
        border: 1px solid #dce2e7;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .panel-heading {
        font-size: 18px;
        font-weight: 750;
        color: #17202a;
        margin-bottom: 16px;
    }

    .panel-note {
        color: #737d86;
        font-size: 13px;
    }

    /* ---------- Machine Identity ---------- */

    .machine-id {
        font-size: 48px;
        font-weight: 800;
        color: #263f4d;
        line-height: 1;
    }

    .machine-label {
        color: #7b858d;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ---------- Health Display ---------- */

    .health-number {
        font-size: 54px;
        font-weight: 850;
        line-height: 1;
    }

    .health-caption {
        color: #737d86;
        margin-top: 5px;
    }

    .healthy {
        color: #16834a;
    }

    .danger {
        color: #c0392b;
    }

    .caution {
        color: #b7791f;
    }

    /* ---------- Risk ---------- */

    .risk-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 7px;
        font-size: 14px;
        font-weight: 650;
    }

    .risk-track {
        height: 14px;
        width: 100%;
        background: #e7ebee;
        border-radius: 20px;
        overflow: hidden;
    }

    .risk-fill {
        height: 100%;
        border-radius: 20px;
    }

    /* ---------- Status ---------- */

    .status-normal {
        background: #e8f6ee;
        border: 1px solid #b9e3ca;
        color: #126b3b;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 700;
    }

    .status-warning {
        background: #fff7df;
        border: 1px solid #f0d68a;
        color: #8a6414;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 700;
    }

    .status-critical {
        background: #fdeceb;
        border: 1px solid #efb8b4;
        color: #9d2d25;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 700;
    }

    /* ---------- Parameter Rows ---------- */

    .parameter-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr;
        align-items: center;
        padding: 13px 0;
        border-bottom: 1px solid #edf0f2;
    }

    .parameter-name {
        font-weight: 650;
        color: #29343d;
    }

    .parameter-value {
        font-weight: 700;
        text-align: right;
        color: #263f4d;
    }

    .parameter-state {
        text-align: right;
        font-size: 12px;
        font-weight: 700;
    }

    .state-good {
        color: #16834a;
    }

    .state-attention {
        color: #b7791f;
    }

    .state-danger {
        color: #c0392b;
    }

    /* ---------- Intelligence ---------- */

    .insight {
        padding: 12px 15px;
        background: #f6f8fa;
        border-left: 4px solid #52758c;
        margin-bottom: 10px;
        border-radius: 5px;
        color: #3d4952;
    }

    /* ---------- Maintenance ---------- */

    .maintenance-box {
        background: #263f4d;
        color: white;
        border-radius: 10px;
        padding: 22px;
    }

    .maintenance-title {
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .maintenance-text {
        color: #d9e2e8;
        line-height: 1.6;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #8a949c;
        font-size: 12px;
        padding: 25px;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🏭 MACHINEGUARD")
st.sidebar.caption("Industrial condition monitoring")

st.sidebar.divider()

st.sidebar.markdown("### Operating Configuration")

machine_type = st.sidebar.selectbox(
    "Machine class",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.slider(
    "Air temperature [K]",
    250.0,
    350.0,
    300.0,
    0.1
)

process_temperature = st.sidebar.slider(
    "Process temperature [K]",
    250.0,
    400.0,
    310.0,
    0.1
)

rotational_speed = st.sidebar.slider(
    "Rotational speed [rpm]",
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
    "Tool wear [min]",
    0,
    300,
    100,
    1
)

temperature_difference = (
    process_temperature - air_temperature
)

st.sidebar.divider()

st.sidebar.caption(
    "Adjust the operating parameters and run "
    "a new condition assessment."
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">Industrial Operations Center</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-title">MachineGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Predictive condition assessment powered by machine learning'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# MACHINE IDENTITY
# ============================================================

left, right = st.columns([1, 2])

with left:

    st.markdown(
        """
        <div class="panel">
            <div class="machine-label">Machine Class</div>
            <div class="machine-id">
        """ +
        machine_type +
        """
            </div>
            <div class="panel-note">
                Current operating profile
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading">
                Current Thermal Condition
            </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Air",
            f"{air_temperature:.1f} K"
        )

    with c2:
        st.metric(
            "Process",
            f"{process_temperature:.1f} K"
        )

    with c3:
        st.metric(
            "Difference",
            f"{temperature_difference:.1f} K"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# OPERATING CONDITION
# ============================================================

st.markdown(
    """
    <div class="panel">
        <div class="panel-heading">
            Operating Condition
        </div>
    """,
    unsafe_allow_html=True
)

def parameter_state(value, low, high):
    if value >= high:
        return "ATTENTION", "state-danger"
    elif value >= low:
        return "MONITOR", "state-attention"
    return "NORMAL", "state-good"

rows = [
    (
        "Rotational speed",
        f"{rotational_speed} rpm",
        *parameter_state(rotational_speed, 2000, 2400)
    ),
    (
        "Torque",
        f"{torque:.1f} Nm",
        *parameter_state(torque, 50, 65)
    ),
    (
        "Tool wear",
        f"{tool_wear} min",
        *parameter_state(tool_wear, 150, 200)
    ),
    (
        "Temperature difference",
        f"{temperature_difference:.1f} K",
        *parameter_state(temperature_difference, 15, 25)
    ),
]

for name, value, state, css in rows:

    st.markdown(
        f"""
        <div class="parameter-row">
            <div class="parameter-name">{name}</div>
            <div class="parameter-value">{value}</div>
            <div class="parameter-state {css}">{state}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ASSESSMENT BUTTON
# ============================================================

st.markdown(
    "### Condition Assessment"
)

run_prediction = st.button(
    "RUN MACHINE ASSESSMENT",
    type="primary",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if run_prediction:

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

        st.error("Unable to generate the machine assessment.")
        st.exception(e)
        st.stop()

    risk = probability * 100
    health = max(0, 100 - risk)

    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    if risk >= 70:

        risk_level = "CRITICAL"
        health_class = "danger"
        fill_color = "#c0392b"

    elif risk >= 30:

        risk_level = "MODERATE"
        health_class = "caution"
        fill_color = "#b7791f"

    else:

        risk_level = "LOW"
        health_class = "healthy"
        fill_color = "#16834a"

    # ========================================================
    # HEALTH + RISK
    # ========================================================

    left, right = st.columns([1, 2])

    with left:

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-heading">
                    Machine Health
                </div>

                <div class="health-number {health_class}">
                    {health:.0f}%
                </div>

                <div class="health-caption">
                    Estimated operating condition
                </div>

                <br>

                <div class="status-{(
                    'critical'
                    if risk >= 70
                    else 'warning'
                    if risk >= 30
                    else 'normal'
                )}">
                    {risk_level} RISK
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-heading">
                    Failure Risk Assessment
                </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="risk-label">
                <span>Predicted failure probability</span>
                <span>{risk:.2f}%</span>
            </div>

            <div class="risk-track">
                <div class="risk-fill"
                     style="width:{min(risk,100)}%;
                            background:{fill_color};">
                </div>
            </div>

            <br>

            <span class="panel-note">
                The probability represents the trained model's
                estimate for the current operating condition.
            </span>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # OPERATIONAL INTELLIGENCE
    # ========================================================

    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading">
                Operational Intelligence
            </div>
        """,
        unsafe_allow_html=True
    )

    insights = []

    if tool_wear >= 200:
        insights.append(
            "Tool wear is elevated and should be inspected."
        )

    if torque >= 60:
        insights.append(
            "Mechanical load is elevated based on torque."
        )

    if rotational_speed >= 2200:
        insights.append(
            "Rotational speed is in a high operating range."
        )

    if temperature_difference >= 25:
        insights.append(
            "Thermal separation between process and air temperature is high."
        )

    if not insights:
        insights.append(
            "No major parameter-level warning was detected "
            "from the configured operating thresholds."
        )

    for insight in insights:

        st.markdown(
            f"""
            <div class="insight">
                • {insight}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # MAINTENANCE PRIORITY
    # ========================================================

    if risk >= 70:

        priority = "HIGH PRIORITY"
        recommendation = (
            "Schedule a maintenance inspection as soon as "
            "practical. Investigate the operating conditions "
            "associated with the elevated predicted risk."
        )

    elif risk >= 30:

        priority = "MONITOR CLOSELY"
        recommendation = (
            "Continue monitoring the machine and review the "
            "operating parameters during the next maintenance check."
        )

    else:

        priority = "ROUTINE"
        recommendation = (
            "Continue normal monitoring and follow the planned "
            "preventive-maintenance schedule."
        )

    st.markdown(
        f"""
        <div class="maintenance-box">

            <div class="maintenance-title">
                🔧 Maintenance Priority: {priority}
            </div>

            <div class="maintenance-text">
                {recommendation}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # ASSESSMENT RECORD
    # ========================================================

    st.markdown("### Assessment Record")

    result_table = pd.DataFrame({
        "Parameter": [
            "Machine class",
            "Air temperature",
            "Process temperature",
            "Temperature difference",
            "Rotational speed",
            "Torque",
            "Tool wear",
            "Failure probability",
            "Risk classification"
        ],
        "Current value": [
            machine_type,
            f"{air_temperature:.1f} K",
            f"{process_temperature:.1f} K",
            f"{temperature_difference:.1f} K",
            f"{rotational_speed} rpm",
            f"{torque:.1f} Nm",
            f"{tool_wear} min",
            f"{risk:.2f}%",
            risk_level
        ]
    })

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Decision-support output only. The prediction should not "
        "replace professional maintenance inspection."
    )

else:

    st.info(
        "Adjust the machine parameters in the control panel "
        "and select RUN MACHINE ASSESSMENT to evaluate the "
        "current operating condition."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        MachineGuard • Industrial Predictive Maintenance •
        XGBoost Decision-Support System
    </div>
    """,
    unsafe_allow_html=True
)
