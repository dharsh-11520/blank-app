import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 🎈 Layout Configurations
st.set_page_config(page_title="LiDAR AUKF Simulation", layout="wide")
st.title("📡 LiDAR Measurements vs Adaptive Unscented Kalman Filter (AUKF)")
st.write("This simulation evaluates sensor tracking loops using an advanced Unscented Kalman filter matrix pipeline.")

# 🛠️ Left Sidebar Controls for Presentation Tweaking
st.sidebar.header("Filter Adjustments")
noise_input_std = st.sidebar.slider("Sensor Noise (Std Dev)", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
threshold = st.sidebar.slider("Classification Threshold (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Core AUKF Logic Function from the updated file
def adaptive_ukf(measurements, initial_r=0.1, noise_std=0.1):
    x = np.array([measurements[0], 0.0]) 
    P = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    Q = np.array([[0.01, 0.0], [0.0, 0.01]]) 
    R = initial_r
    
    n = 2
    alpha = 0.5
    beta = 2
    kappa = 0
    lambda_ = alpha**2 * (n + kappa) - n
    
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wc = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    
    filtered_distances = []
    velocities = []
    dt = 1.0
    
    for measurement in measurements:
        try:
            sqrt_matrix = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            P = P + np.eye(n) * 1e-6
            sqrt_matrix = np.linalg.cholesky((n + lambda_) * P)
            
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_matrix[:, i]
            sigma_points[i + 1 + n] = x - sqrt_matrix[:, i]
            
        predicted_sigma = np.zeros_like(sigma_points)
        for i in range(2 * n + 1):
            distance = sigma_points[i, 0]
            velocity = sigma_points[i, 1]
            predicted_sigma[i, 0] = distance + velocity * dt
            predicted_sigma[i, 1] = velocity
            
        x_pred = np.sum(Wm[:, None] * predicted_sigma, axis=0)
        
        P_pred = np.zeros((n, n))
        for i in range(2 * n + 1):
            difference = predicted_sigma[i] - x_pred
            P_pred += Wc[i] * np.outer(difference, difference)
        P_pred += Q
        
        predicted_measurements = predicted_sigma[:, 0]
        z_pred = np.sum(Wm * predicted_measurements)
        
        S = 0.0
        for i in range(2 * n + 1):
            difference = (predicted_measurements[i] - z_pred)
            S += Wc[i] * difference**2
        S += R
        
        cross_covariance = np.zeros(n)
        for i in range(2 * n + 1):
            state_difference = (predicted_sigma[i] - x_pred)
            measurement_difference = (predicted_measurements[i] - z_pred)
            cross_covariance += (Wc[i] * state_difference * measurement_difference)
            
        innovation = measurement - z_pred
        R = 0.95 * R + 0.05 * max(innovation**2 - S + R, 0.01)
        
        K = cross_covariance / S
        x = x_pred + K * innovation
        P = P_pred - np.outer(K, K) * S
        P = (P + P.T) / 2
        P += np.eye(n) * 1e-6
        
        filtered_distances.append(float(round(x[0], 2)))
        velocities.append(float(round(x[1], 2)))
        
    return filtered_distances, velocities

col1, col2 = st.columns(2)

# --- CASE 01: STATIC ---
with col1:
    st.subheader("💡 CASE 01: Static Obstacle")
    actual_distance_static = 5
    measurements_static = [4.95, 5.06, 5.07, 4.91, 5.13, 4.89, 5.05, 4.92, 4.8, 5.05, 5.12, 4.96, 5.0, 4.94, 4.96, 5.07, 5.01, 5.08, 5.01, 5.03]
    
    filtered_static, _ = adaptive_ukf(measurements_static, noise_std=noise_input_std)
    
    fig1, ax1 = plt.subplots()
    time_lidar = range(1, 21)
    ax1.plot(time_lidar, measurements_static, marker="o", label="LiDAR measurements", color="red")
    ax1.plot(time_lidar, filtered_static, marker="o", label="AUKF filtered", color="blue")
    ax1.set_xlabel("Measurement Number")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Static Obstacle Tracking")
    ax1.legend()
    st.pyplot(fig1)
    
    static_change = max(filtered_static) - min(filtered_static)
    status_static = "🔴 Obstacle is STATIC" if static_change < threshold else "🟢 Obstacle is DYNAMIC"
    st.metric(label="Calculated Variance Delta", value=f"{round(static_change, 2)} m")
    st.success(status_static)

# --- CASE 02: DYNAMIC ---
with col2:
    st.subheader("🏃‍♂️ CASE 02: Dynamic Obstacle")
    measurements_dynamic = [10.91, 10.38, 9.95, 9.62, 8.96, 8.58, 8.0, 7.48, 7.08, 6.28, 6.05, 5.49, 4.99, 4.47, 4.06, 3.31, 3.1, 2.59, 1.89, 1.35]
    
    filtered_dynamic, _ = adaptive_ukf(measurements_dynamic, noise_std=noise_input_std)
    
    fig2, ax2 = plt.subplots()
    time_dynamic = range(1, 21)
    ax2.plot(time_dynamic, measurements_dynamic, marker="o", label="LiDAR measurements", color="red")
    ax2.plot(time_dynamic, filtered_dynamic, marker="o", label="AUKF filtered", color="blue")
    ax2.set_xlabel("Measurement Number")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("Dynamic Obstacle Tracking")
    ax2.legend()
    st.pyplot(fig2)
    
    dynamic_change = max(filtered_dynamic) - min(filtered_dynamic)
    status_dynamic = "🔴 Obstacle is STATIC" if dynamic_change < threshold else "🟢 Obstacle is DYNAMIC"
    st.metric(label="Calculated Variance Delta", value=f"{round(dynamic_change, 2)} m")
    st.warning(status_dynamic)
