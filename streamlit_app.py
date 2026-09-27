import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 🎈 Web Page Title & Layout Configurations
st.set_page_config(page_title="LiDAR Kalman Filter Sim", layout="wide")
st.title("📡 LiDAR Measurements vs Kalman Filter Simulation")
st.write("Interact with the sliders on the left sidebar to change filter variables in real-time.")

# 🛠️ Left Sidebar Controls (Dynamic Inputs)
st.sidebar.header("Simulation Parameters")
kalman_gain = st.sidebar.slider("Kalman Gain", min_value=0.01, max_value=1.00, value=0.91, step=0.01)
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
threshold = st.sidebar.slider("Static/Dynamic Distance Threshold (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Split screen into two columns for Case 1 and Case 2 side-by-side
col1, col2 = st.columns(2)

# --- CASE 01: STATIC OBSTACLE ---
with col1:
    st.subheader("💡 CASE 01: Static Obstacle")
    actual_distance_static = 5
    noises_static = []
    measurements_static = []
    
    np.random.seed(42) 
    for i in range(20):
        noise = np.random.normal(0, noise_std)
        noises_static.append(round(noise, 2))
        measurements_static.append(round(actual_distance_static + noise, 2))
        
    filtered_static = []
    estimate = measurements_static[0]
    for measurement in measurements_static:
        estimate = estimate + kalman_gain * (measurement - estimate)
        filtered_static.append(round(estimate, 2))
        
    fig1, ax1 = plt.subplots()
    time_lidar = range(1, 21)
    ax1.plot(time_lidar, measurements_static, marker="o", label="LiDAR measurements", color="red")
    ax1.plot(time_lidar, filtered_static, marker="o", label="Kalman filtered", color="blue")
    ax1.set_xlabel("Measurement Number")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Static Obstacle Analysis")
    ax1.legend()
    st.pyplot(fig1)
    
    static_change = max(filtered_static) - min(filtered_static)
    status_static = "🔴 Obstacle is STATIC" if static_change < threshold else "🟢 Obstacle is DYNAMIC"
    st.metric(label="Distance Change Calculated", value=f"{round(static_change, 2)} m")
    st.success(status_static)

# --- CASE 02: DYNAMIC OBSTACLE ---
with col2:
    st.subheader("🏃‍♂️ CASE 02: Dynamic Obstacle")
    dynamic_actual_distance = [11, 10.5, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5]
    noises_dynamic = []
    measurements_dynamic = []
    
    for distance in dynamic_actual_distance:
        noise = np.random.normal(0, noise_std)
        noises_dynamic.append(round(noise, 2))
        measurements_dynamic.append(round(distance + noise, 2))
        
    filtered_dynamic = []
    estimate = measurements_dynamic[0]
    for measurement in measurements_dynamic:
        estimate = estimate + kalman_gain * (measurement - estimate)
        filtered_dynamic.append(round(estimate, 2))
        
    fig2, ax2 = plt.subplots()
    time_dynamic = range(1, 21)
    ax2.plot(time_dynamic, measurements_dynamic, marker="o", label="LiDAR measurements", color="red")
    ax2.plot(time_dynamic, filtered_dynamic, marker="o", label="Kalman filtered", color="blue")
    ax2.set_xlabel("Measurement Number")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("Dynamic Obstacle Analysis")
    ax2.legend()
    st.pyplot(fig2)
    
    dynamic_change = max(filtered_dynamic) - min(filtered_dynamic)
    status_dynamic = "🔴 Obstacle is STATIC" if dynamic_change < threshold else "🟢 Obstacle is DYNAMIC"
    st.metric(label="Distance Change Calculated", value=f"{round(dynamic_change, 2)} m")
    st.warning(status_dynamic)
