import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('pump_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("⚙️ AI-Based Digital Twin for Boiler Feed Water Pump")
st.write("### Predictive Maintenance & Fault Detection Dashboard")

st.sidebar.header("🔧 Pump Sensor Controls")

speed = st.sidebar.slider("Pump Speed (RPM)", 2500, 3500, 3000)
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

if prediction == 'Normal':
    st.success("✅ **Status: Normal** - The Boiler Feed Water Pump is operating smoothly. No issues detected.")
else:
    st.error(f"⚠️ **Warning! Detected Fault: {prediction}**")
    
    if prediction == 'Cavitation':
        st.warning("**Root Cause:** Low suction pressure combined with high vibration.")
        st.info("**Recommended Action:** Please check the suction line, valves, and pump inlet immediately.")
    elif prediction == 'Bearing Failure':
        st.warning("**Root Cause:** Abnormal increase in vibration and motor current.")
        st.info("**Recommended Action:** Inspect and lubricate or replace the pump bearings.")
    elif prediction == 'Impeller Failure':
        st.warning("**Root Cause:** Drop in flow rate and discharge pressure.")
        st.info("**Recommended Action:** Check the pump impeller for wear, damage, or clogging.")
    elif prediction == 'Pipe Blockage':
        st.warning("**Root Cause:** Excessive discharge pressure with reduced flow rate.")
        st.info("**Recommended Action:** Inspect the discharge pipeline for any blockage or closed valves.")
