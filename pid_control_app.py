import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# PID + Plant Simulation (Room + Heater/AC)
# ==========================
def simulate_pid(Kp, Ki, Kd, setpoint, disturbance, tau, sim_time, dt, u_min=-np.inf, u_max=np.inf, ambient_temp=25.0):
    t = np.arange(0, sim_time + dt, dt)
    n = len(t)
    y = np.ones(n) * 32.0   # Room temperature starting at 32°C
    u = np.zeros(n)   # Heater/AC power
    e = np.zeros(n)   # Error
    integral = 0.0
    prev_error = 0.0
    Kp_plant = 1.0    # Plant gain (heater/AC effectiveness)

    # ambient_temp = 30.0  # Ambient temperature
    for i in range(1, n):
        # Temperature error
        e[i] = setpoint - y[i-1]

        # PID terms
        integral += e[i] * dt
        derivative = (e[i] - prev_error) / dt
        u_raw = Kp * e[i] + Ki * integral + Kd * derivative

        # Actuator limits (heater/AC power)
        u_clamped = max(u_min, min(u_max, u_raw))
        if u_clamped != u_raw:
            # Anti-windup
            integral -= e[i] * dt
        u[i] = u_clamped

        # Disturbance (external heat/cold)
        dist = disturbance(t[i]) if callable(disturbance) else disturbance

        # Room temperature dynamics
        dydt = -(1.0/tau) * (y[i-1] - ambient_temp) + Kp_plant * u[i] + dist
        y[i] = y[i-1] + dydt * dt
        prev_error = e[i]

    return t, y, u, e


# ==========================
# Disturbance options
# ==========================
def make_disturbance(kind, magnitude, start, duration, sim_time):
    if kind == 'None':
        return 0.0
    elif kind == 'Step':
        return lambda t: magnitude if t >= start else 0.0
    elif kind == 'Pulse':
        return lambda t: magnitude if (t >= start and t < start + duration) else 0.0
    elif kind == 'Sine':
        return lambda t: magnitude * np.sin(2*np.pi*(1.0/(sim_time/5)) * t)
    else:
        return 0.0


# ==========================
# Streamlit UI
# ==========================
st.title("🌡️ HVAC Feedback Control - Heater + Air Conditioner")

# --- Sidebar: Controller ---
# st.sidebar.header("⚙️ PID Controller Gains (Thermostat)")
Kp = 3.0;#st.sidebar.slider("Kp (Proportional)", 0.0, 20.0, 3.0, 0.1)
Ki = 1.2;#st.sidebar.slider("Ki (Integral)", 0.0, 10.0, 1.2, 0.05)
Kd = 0.2#st.sidebar.slider("Kd (Derivative)", 0.0, 5.0, 0.2, 0.01)

# --- Sidebar: Plant ---
st.sidebar.header("🏠 Room + HVAC Dynamics")
tau = st.sidebar.slider("Room thermal constant (τ)", 0.05, 10.0, 1.5, 0.05)
sim_time = st.sidebar.slider("Simulation time (s)", 1.0, 100.0, 20.0, 1.0)
dt = st.sidebar.slider("Time step (dt)", 0.001, 0.1, 0.01, 0.001)
setpoint = st.sidebar.slider("Set Temperature (°C)", 15.0, 35.0, 28.0, 0.5)
u_min = st.sidebar.number_input("AC max cooling (negative)", value=-10.0)
u_max = st.sidebar.number_input("Heater max heating (positive)", value=10.0)

ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 10.0, 40.0, 32.0, 0.5)

# --- Sidebar: Disturbance ---
# st.sidebar.header("🌪 External Disturbance (Weather, Window Open)")
kind = 'None'#st.sidebar.selectbox("Type", ['None', 'Step', 'Pulse', 'Sine'])
mag = -0.5#st.sidebar.slider("Magnitude", -5.0, 5.0, -0.5, 0.1)
start = 10.0#st.sidebar.slider("Start (s)", 0.0, sim_time, 10.0, 0.1)
dur = 5.0#st.sidebar.slider("Duration (s, for pulse)", 0.1, sim_time, 5.0, 0.1)

# Create disturbance
disturbance = make_disturbance(kind, mag, start, dur, sim_time)

# Run simulation
t, y, u, e = simulate_pid(Kp, Ki, Kd, setpoint, disturbance, tau, sim_time, dt, u_min, u_max, ambient_temp)

# ==========================
# Plotting
# ==========================
st.subheader("📊 Room Temperature vs Setpoint")
fig1 = plt.figure(figsize=(8,3.5))
plt.plot(t, y, label="Room Temperature y(t)")
plt.plot(t, np.ones_like(t)*setpoint, 'r--', label="Setpoint r(t)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
st.pyplot(fig1)

st.subheader("📊 Temperature Error e(t) = r(t) - y(t)")
fig3 = plt.figure(figsize=(8,3.5))
plt.plot(t, e, label="Error")
plt.xlabel("Time (s)")
plt.ylabel("Error (°C)")
plt.grid(True)
plt.legend()
st.pyplot(fig3)

st.subheader("📊 Heater(+)/AC(-) Power (Control Signal u(t))")
fig2 = plt.figure(figsize=(8,3.5))
plt.plot(t, u, label="Heater(+)/AC(-) Output")
plt.xlabel("Time (s)")
plt.ylabel("Power")
plt.grid(True)
plt.legend()
st.pyplot(fig2)

# ==========================
# Metrics
# ==========================
steady_state_error = e[-1]
overshoot = (np.max(y) - setpoint)
tol = 0.02 * (abs(setpoint) + 1e-9)
settling_time = None
for i in range(len(t)-1, -1, -1):
    if abs(y[i] - setpoint) > tol:
        settling_time = t[i+1] if i+1 < len(t) else None
        break

# st.subheader("📈 Performance Metrics")
# st.write(f"**Steady-state error**: {steady_state_error:.4f} °C")
# st.write(f"**Overshoot**: {overshoot:.4f} °C")
# if settling_time is not None:
#     st.write(f"**Settling time** ≈ {settling_time:.2f} s (±2%)")
# else:
#     st.write("Settling time: not reached within simulation time")