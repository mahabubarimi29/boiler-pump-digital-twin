import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn  # sklearn মডিউল নিশ্চিত করার জন্য

# Page Configuration
st.set_page_config(
    page_title="Boiler Feed Water Pump Digital Twin",
    page_icon="⚡",
    layout="wide"
)

st.title("🤖 AI-Based Digital Twin for Boiler Feed Water Pump")
st.write("### Predictive Maintenance & Fault Detection Dashboard")

# Model Load safely with caching
@st.cache_resource
def load_pump_model():
    return joblib.load('pump_model.pkl')

try:
    model = load_pump_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.sidebar.header("🔧 Pump Sensor Controls")

speed = st.sidebar.slider("Pump Speed (RPM)", 2500.0, 3500.0, 3000.0)
suction_pressure = st.sidebar.slider("Suction Pressure (bar)", 0.5, 3.5, 2.5)
discharge_pressure = st.sidebar.slider("Discharge Pressure (bar)", 100.0, 200.0, 150.0)
flow_rate = st.sidebar.slider("Flow Rate (m³/h)", 50.0, 120.0, 100.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 1.0, 10.0, 2.0)
motor_current = st.sidebar.slider("Motor Current (A)", 35.0, 65.0, 45.0)

input_data = pd.DataFrame([[speed, suction_pressure, discharge_pressure, flow_rate, vibration, motor_current]],
columns=['Speed', 'Suction_Pressure', 'Discharge_Pressure', 'Flow_Rate', 'Vibration', 'Motor_Current'])

prediction = model.predict(input_data)[0]

st.subheader("📊 Live Sensor Status")
st.dataframe(input_data)

st.subheader("🔍 Fault Diagnosis & Recommendation")

if str(prediction).lower() == 'normal':
    st.success("✅ **Status: Normal** - The Boiler Feed Water Pump is operating smoothly. No issues detected.")
else:
    st.error(f"⚠️ **Warning! Detected Fault: {prediction}**")
    st.warning("⚠️ **Root Cause:** Abnormal sensor readings detected in vibration or pressure parameters.")
    st.info("💡 **Recommended Action:** Please inspect the pump unit, check lubrication, and review valve positions immediately.")