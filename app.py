import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BFP Digital Twin - XAI & Fault Diagnosis",
    page_icon="⚡",
    layout="wide",
)

st.title(
    "⚡ Boiler Feed Water Pump (BFP) Digital Twin & Fault Diagnosis System"
)
st.caption(
    "Real-time Telemetry Simulation, Min-Max Threshold Monitoring, ML"
    " Inference, and Explainable Root-Cause Analysis"
)

# ==========================================
# 📋 SAFE OPERATING LIMITS (MIN - MAX RANGES)
# ==========================================
LIMITS = {
    "speed": {"label": "Pump Speed", "min": 2700.0, "max": 3200.0, "unit": "RPM"},
    "suction_p": {
        "label": "Suction Pressure",
        "min": 2.0,
        "max": 4.0,
        "unit": "bar",
    },
    "discharge_p": {
        "label": "Discharge Pressure",
        "min": 25.0,
        "max": 45.0,
        "unit": "bar",
    },
    "flow_rate": {
        "label": "Flow Rate",
        "min": 100.0,
        "max": 250.0,
        "unit": "m³/h",
    },
    "vibration": {
        "label": "Vibration",
        "min": 0.0,
        "max": 2.5,
        "unit": "mm/s",
    },
    "motor_current": {
        "label": "Motor Current",
        "min": 30.0,
        "max": 75.0,
        "unit": "A",
    },
}

PRESETS = {
    "Custom / Manual Entry": None,
    "Normal Operation": {
        "speed": 3000.0,
        "suction_p": 2.8,
        "discharge_p": 35.0,
        "flow_rate": 165.0,
        "vibration": 1.2,
        "motor_current": 52.0,
    },
    "Cavitation Risk (Low Suction)": {
        "speed": 3000.0,
        "suction_p": 1.1,
        "discharge_p": 28.0,
        "flow_rate": 115.0,
        "vibration": 3.1,
        "motor_current": 48.0,
    },
    "High Vibration / Bearing Wear": {
        "speed": 3020.0,
        "suction_p": 2.7,
        "discharge_p": 34.0,
        "flow_rate": 155.0,
        "vibration": 4.8,
        "motor_current": 64.0,
    },
    "Motor Overload / High Current": {
        "speed": 3100.0,
        "suction_p": 2.6,
        "discharge_p": 42.0,
        "flow_rate": 210.0,
        "vibration": 2.9,
        "motor_current": 88.0,
    },
}


@st.cache_resource
def load_ml_model(model_path="bfp_model.joblib"):
  if os.path.exists(model_path):
    try:
      return joblib.load(model_path)
    except Exception:
      return None
  return None


ml_model = load_ml_model("bfp_model.joblib")

# ==========================================
# 👨‍🏫 TEACHER'S DEDICATED EVALUATION PANEL
# ==========================================
st.sidebar.header("👨‍🏫 Teacher's Evaluation Panel")

if "sensor_vals" not in st.session_state:
  st.session_state.sensor_vals = {
      "speed": 3000.0,
      "suction_p": 2.8,
      "discharge_p": 35.0,
      "flow_rate": 165.0,
      "vibration": 1.2,
      "motor_current": 52.0,
  }

selected_preset = st.sidebar.selectbox(
    "Quick Test Scenario (Optional)", list(PRESETS.keys())
)

if PRESETS[selected_preset] is not None:
  st.session_state.sensor_vals = PRESETS[selected_preset].copy()

if st.sidebar.button("🎲 Generate Random Sensor Entry"):
  rng = np.random.default_rng()
  st.session_state.sensor_vals = {
      "speed": float(rng.integers(2200, 3800)),
      "suction_p": round(float(rng.uniform(0.5, 5.5)), 1),
      "discharge_p": round(float(rng.uniform(15.0, 55.0)), 1),
      "flow_rate": float(rng.integers(60, 290)),
      "vibration": round(float(rng.uniform(0.2, 8.0)), 1),
      "motor_current": float(rng.integers(15, 95)),
  }

input_mode = st.sidebar.radio(
    "Input Method", ["Sliders", "Direct Number Entry"]
)

st.sidebar.markdown("**Enter or Adjust Sensor Values:**")

if input_mode == "Sliders":
  speed = st.sidebar.slider(
      f"Pump Speed (Safe: {LIMITS['speed']['min']:.0f} -"
      f" {LIMITS['speed']['max']:.0f} RPM)",
      2000.0,
      4000.0,
      float(st.session_state.sensor_vals["speed"]),
      step=25.0,
  )
  suction_p = st.sidebar.slider(
      f"Suction Pressure (Safe: {LIMITS['suction_p']['min']:.1f} -"
      f" {LIMITS['suction_p']['max']:.1f} bar)",
      0.0,
      6.0,
      float(st.session_state.sensor_vals["suction_p"]),
      step=0.1,
  )
  discharge_p = st.sidebar.slider(
      f"Discharge Pressure (Safe: {LIMITS['discharge_p']['min']:.1f} -"
      f" {LIMITS['discharge_p']['max']:.1f} bar)",
      10.0,
      60.0,
      float(st.session_state.sensor_vals["discharge_p"]),
      step=0.5,
  )
  flow_rate = st.sidebar.slider(
      f"Flow Rate (Safe: {LIMITS['flow_rate']['min']:.0f} -"
      f" {LIMITS['flow_rate']['max']:.0f} m³/h)",
      50.0,
      300.0,
      float(st.session_state.sensor_vals["flow_rate"]),
      step=5.0,
  )
  vibration = st.sidebar.slider(
      f"Vibration (Safe: {LIMITS['vibration']['min']:.1f} -"
      f" {LIMITS['vibration']['max']:.1f} mm/s)",
      0.0,
      10.0,
      float(st.session_state.sensor_vals["vibration"]),
      step=0.1,
  )
  motor_current = st.sidebar.slider(
      f"Motor Current (Safe: {LIMITS['motor_current']['min']:.0f} -"
      f" {LIMITS['motor_current']['max']:.0f} A)",
      10.0,
      100.0,
      float(st.session_state.sensor_vals["motor_current"]),
      step=1.0,
  )
else:
  speed = st.sidebar.number_input(
      f"Pump Speed (Safe: {LIMITS['speed']['min']:.0f} -"
      f" {LIMITS['speed']['max']:.0f} RPM)",
      min_value=0.0,
      max_value=6000.0,
      value=float(st.session_state.sensor_vals["speed"]),
      step=10.0,
  )
  suction_p = st.sidebar.number_input(
      f"Suction Pressure (Safe: {LIMITS['suction_p']['min']:.1f} -"
      f" {LIMITS['suction_p']['max']:.1f} bar)",
      min_value=0.0,
      max_value=20.0,
      value=float(st.session_state.sensor_vals["suction_p"]),
      step=0.1,
  )
  discharge_p = st.sidebar.number_input(
      f"Discharge Pressure (Safe: {LIMITS['discharge_p']['min']:.1f} -"
      f" {LIMITS['discharge_p']['max']:.1f} bar)",
      min_value=0.0,
      max_value=120.0,
      value=float(st.session_state.sensor_vals["discharge_p"]),
      step=0.5,
  )
  flow_rate = st.sidebar.number_input(
      f"Flow Rate (Safe: {LIMITS['flow_rate']['min']:.0f} -"
      f" {LIMITS['flow_rate']['max']:.0f} m³/h)",
      min_value=0.0,
      max_value=600.0,
      value=float(st.session_state.sensor_vals["flow_rate"]),
      step=5.0,
  )
  vibration = st.sidebar.number_input(
      f"Vibration (Safe: {LIMITS['vibration']['min']:.1f} -"
      f" {LIMITS['vibration']['max']:.1f} mm/s)",
      min_value=0.0,
      max_value=25.0,
      value=float(st.session_state.sensor_vals["vibration"]),
      step=0.1,
  )
  motor_current = st.sidebar.number_input(
      f"Motor Current (Safe: {LIMITS['motor_current']['min']:.0f} -"
      f" {LIMITS['motor_current']['max']:.0f} A)",
      min_value=0.0,
      max_value=200.0,
      value=float(st.session_state.sensor_vals["motor_current"]),
      step=1.0,
  )

st.session_state.sensor_vals = {
    "speed": speed,
    "suction_p": suction_p,
    "discharge_p": discharge_p,
    "flow_rate": flow_rate,
    "vibration": vibration,
    "motor_current": motor_current,
}

# ==========================================
# 🔍 FAULT & ROOT-CAUSE LOGIC ENGINE
# ==========================================
values = {
    "speed": speed,
    "suction_p": suction_p,
    "discharge_p": discharge_p,
    "flow_rate": flow_rate,
    "vibration": vibration,
    "motor_current": motor_current,
}

faults = []

if speed < LIMITS["speed"]["min"]:
  faults.append({
      "sensor": "Pump Speed Sensor",
      "fault": "Under-Speed Operation",
      "warning": (
          f"Speed is {speed:.0f} RPM (Normal Range:"
          f" {LIMITS['speed']['min']:.0f}–{LIMITS['speed']['max']:.0f} RPM)."
      ),
      "cause": (
          "Supply frequency dip, VFD/drive under-frequency, or excessive"
          " mechanical drag."
      ),
      "recommendation": (
          "Inspect VFD speed reference, verify power supply stability, and"
          " check shaft rotation resistance."
      ),
  })
elif speed > LIMITS["speed"]["max"]:
  faults.append({
      "sensor": "Pump Speed Sensor",
      "fault": "Over-Speed Operation",
      "warning": (
          f"Speed is {speed:.0f} RPM (Normal Range:"
          f" {LIMITS['speed']['min']:.0f}–{LIMITS['speed']['max']:.0f} RPM)."
      ),
      "cause": (
          "Control loop overshoot, drive setpoint error, or sudden load drop."
      ),
      "recommendation": (
          "Check speed control loop parameters and verify discharge valve"
          " operating state."
      ),
  })

if suction_p < LIMITS["suction_p"]["min"]:
  faults.append({
      "sensor": "Suction Pressure Sensor",
      "fault": "Low Suction Pressure / Cavitation Risk",
      "warning": (
          f"Suction pressure is {suction_p:.1f} bar (Normal Range:"
          f" {LIMITS['suction_p']['min']:.1f}–{LIMITS['suction_p']['max']:.1f}"
          " bar)."
      ),
      "cause": (
          "Choked suction strainer, partially closed inlet valve, or low"
          " deaerator tank level."
      ),
      "recommendation": (
          "Clean suction strainer, confirm inlet isolation valve is fully open,"
          " and restore feed tank level."
      ),
  })
elif suction_p > LIMITS["suction_p"]["max"]:
  faults.append({
      "sensor": "Suction Pressure Sensor",
      "fault": "High Suction Pressure",
      "warning": (
          f"Suction pressure is {suction_p:.1f} bar (Normal Range:"
          f" {LIMITS['suction_p']['min']:.1f}–{LIMITS['suction_p']['max']:.1f}"
          " bar)."
      ),
      "cause": (
          "Booster pump over-pressure or deaerator pressure regulation"
          " abnormality."
      ),
      "recommendation": (
          "Check upstream booster pressure and inspect deaerator pressure"
          " control valves."
      ),
  })

if discharge_p < LIMITS["discharge_p"]["min"]:
  faults.append({
      "sensor": "Discharge Pressure Sensor",
      "fault": "Low Discharge Pressure / Internal Wear or Leakage",
      "warning": (
          f"Discharge pressure is {discharge_p:.1f} bar (Normal Range:"
          f" {LIMITS['discharge_p']['min']:.1f}–{LIMITS['discharge_p']['max']:.1f}"
          " bar)."
      ),
      "cause": (
          "Worn wear rings, impeller erosion, air entrainment, or bypass"
          " leakage."
      ),
      "recommendation": (
          "Check recirculation/bypass valve position, inspect for air pockets,"
          " and plan wear-ring inspection."
      ),
  })
elif discharge_p > LIMITS["discharge_p"]["max"]:
  faults.append({
      "sensor": "Discharge Pressure Sensor",
      "fault": "High Discharge Pressure / Dead-Head Risk",
      "warning": (
          f"Discharge pressure is {discharge_p:.1f} bar (Normal Range:"
          f" {LIMITS['discharge_p']['min']:.1f}–{LIMITS['discharge_p']['max']:.1f}"
          " bar)."
      ),
      "cause": (
          "Throttled or closed discharge isolation valve or downstream line"
          " restriction."
      ),
      "recommendation": (
          "Verify discharge valve opening immediately and ensure minimum flow"
          " protection is active."
      ),
  })

if flow_rate < LIMITS["flow_rate"]["min"]:
  faults.append({
      "sensor": "Flow Rate Sensor",
      "fault": "Low Flow / Recirculation & Heating Risk",
      "warning": (
          f"Flow rate is {flow_rate:.0f} m³/h (Normal Range:"
          f" {LIMITS['flow_rate']['min']:.0f}–{LIMITS['flow_rate']['max']:.0f}"
          " m³/h)."
      ),
      "cause": (
          "High discharge head resistance, partial blockage, or low boiler"
          " feed demand."
      ),
      "recommendation": (
          "Verify automatic recirculation valve operation and inspect feedline"
          " restrictions."
      ),
  })
elif flow_rate > LIMITS["flow_rate"]["max"]:
  faults.append({
      "sensor": "Flow Rate Sensor",
      "fault": "High Flow / Run-Out Condition",
      "warning": (
          f"Flow rate is {flow_rate:.0f} m³/h (Normal Range:"
          f" {LIMITS['flow_rate']['min']:.0f}–{LIMITS['flow_rate']['max']:.0f}"
          " m³/h)."
      ),
      "cause": (
          "Excessive downstream demand, line leak, or reduced system head."
      ),
      "recommendation": (
          "Check downstream piping for leaks and monitor motor current against"
          " overload limits."
      ),
  })

if vibration > LIMITS["vibration"]["max"]:
  faults.append({
      "sensor": "Vibration Sensor",
      "fault": "High Vibration / Misalignment, Unbalance, or Bearing Wear",
      "warning": (
          f"Vibration is {vibration:.1f} mm/s (Normal Range:"
          f" {LIMITS['vibration']['min']:.1f}–{LIMITS['vibration']['max']:.1f}"
          " mm/s)."
      ),
      "cause": (
          "Bearing degradation, shaft misalignment, rotor unbalance, or"
          " cavitation."
      ),
      "recommendation": (
          "Check bearing housing temperature/lubrication, verify shaft"
          " alignment, and run vibration spectrum analysis."
      ),
  })

if motor_current < LIMITS["motor_current"]["min"]:
  faults.append({
      "sensor": "Motor Current Sensor",
      "fault": "Low Current / Under-Load or Loss of Priming",
      "warning": (
          f"Motor current is {motor_current:.0f} A (Normal Range:"
          f" {LIMITS['motor_current']['min']:.0f}–{LIMITS['motor_current']['max']:.0f}"
          " A)."
      ),
      "cause": (
          "Dry/low-fluid pumping, coupling issue, or reduced hydraulic load."
      ),
      "recommendation": (
          "Check fluid priming and suction supply immediately to avoid seal"
          " damage."
      ),
  })
elif motor_current > LIMITS["motor_current"]["max"]:
  faults.append({
      "sensor": "Motor Current Sensor",
      "fault": "Motor Overload / High Mechanical or Electrical Load",
      "warning": (
          f"Motor current is {motor_current:.0f} A (Normal Range:"
          f" {LIMITS['motor_current']['min']:.0f}–{LIMITS['motor_current']['max']:.0f}"
          " A)."
      ),
      "cause": (
          "High fluid density/flow load, internal rubbing, bearing seizure, or"
          " voltage imbalance."
      ),
      "recommendation": (
          "Check phase currents/voltage, inspect motor & pump bearing"
          " temperatures, and reduce load if unsafe."
      ),
  })

ml_status_text = "Rule + Threshold XAI Engine Active"
if ml_model is not None:
  features = np.array(
      [[speed, suction_p, discharge_p, flow_rate, vibration, motor_current]]
  )
  pred = ml_model.predict(features)[0]
  ml_status_text = f"ML Model Prediction: {pred} | XAI Rule Engine Active"

# ==========================================
# 📊 MAIN DASHBOARD LAYOUT
# ==========================================
st.info(f"**Engine Status:** {ml_status_text}")

col_left, col_right = st.columns([1.15, 1.25], gap="large")

with col_left:
  st.subheader("📌 Sensor Min-Max Limits & Current Status")
  rows = []
  for key, meta in LIMITS.items():
    val = values[key]
    in_range = meta["min"] <= val <= meta["max"]
    rows.append({
        "Sensor": meta["label"],
        "Current": f"{val:g} {meta['unit']}",
        "Normal Range (Min–Max)": f"{meta['min']:g} – {meta['max']:g} {meta['unit']}",
        "Status": "✅ Normal" if in_range else "⚠️ Out of Range",
    })
  st.table(pd.DataFrame(rows))

  st.subheader("📈 Normalized Range Comparison (0%–100% of Slider Span)")
  slider_spans = {
      "speed": (2000.0, 4000.0),
      "suction_p": (0.0, 6.0),
      "discharge_p": (10.0, 60.0),
      "flow_rate": (50.0, 300.0),
      "vibration": (0.0, 10.0),
      "motor_current": (10.0, 100.0),
  }
  chart_df = pd.DataFrame({
      "Sensor": [LIMITS[k]["label"] for k in LIMITS],
      "Normalized Level (%)": [
          round(
              (values[k] - slider_spans[k][0])
              / (slider_spans[k][1] - slider_spans[k][0])
              * 100,
              1,
          )
          for k in LIMITS
      ],
  }).set_index("Sensor")
  st.bar_chart(chart_df)

with col_right:
  st.subheader("🔍 Diagnostic Alert, Root Cause & Recommendation")

  if not faults:
    st.success("SYSTEM STATUS: NORMAL ✅")
    st.write(
        "All six BFP sensor readings are within their defined **Minimum to"
        " Maximum** safe operating ranges. No fault or warning is triggered."
    )
  else:
    st.error(
        f"SYSTEM STATUS: FAULT / WARNING DETECTED ⚠️ ({len(faults)} sensor"
        " issue(s))"
    )
    for item in faults:
      with st.expander(
          f"🚨 {item['sensor']} — {item['fault']}", expanded=True
      ):
        st.warning(f"**Warning:** {item['warning']}")
        st.write(f"**Root Cause:** {item['cause']}")
        st.info(f"**Recommendation:** {item['recommendation']}")