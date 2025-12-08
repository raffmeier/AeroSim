import numpy as np
from matplotlib import pyplot as plt
import time

J_mot = 1.3 * 10**-4  # kg*m2
b = 0.00035       # kg*m2/s
k_T = 0.0649      # Nm/A
k_e = k_T
R = 0.103         # Ohm
L = 0.048 * 10**-3  # H

J_prop = 1.79 * 10**-3 # kg*m2 (propeller inertia)

J = J_mot + J_prop

V_max = 50  # V

dt = 0.00001    # 10 µs
timesteps = 50000

states = np.zeros((2, timesteps))  # [omega, current]

# logging
omega_des = 500.0
omega_ref_log = np.zeros(timesteps)
i_ref = np.zeros(timesteps)
voltage = np.zeros(timesteps)
power = np.zeros(timesteps)

# controller gains
# outer speed loop
Kp_w = 10
Ki_w = 200

# inner current loop
Kp_i = 0.013
Ki_i = 80

speed_error = np.zeros(timesteps)
speed_int_error = np.zeros(timesteps)

current_error = np.zeros(timesteps)
current_int_error = np.zeros(timesteps)

i_max = 90   # A
u_min, u_max = 0.0, 1.0

input_u = 0.0 

def state_dynamics(state, u, tau_load):
    V = V_max * u
    omega_dot = (k_T * state[1] - b * state[0] - tau_load) / J
    current_dot = (V - R * state[1] - k_e * state[0]) / L
    return np.array([omega_dot, current_dot])

def rk4(state, u, tau_load, dt):
    k1 = state_dynamics(state, u, tau_load)
    k2 = state_dynamics(state + 0.5*dt*k1, u, tau_load)
    k3 = state_dynamics(state + 0.5*dt*k2, u, tau_load)
    k4 = state_dynamics(state + dt*k3,     u, tau_load)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

start = time.time()

for step in range(1, timesteps):

    # if step > 20000:
    #     omega_des = 500.0

    omega_ref_log[step] = omega_des

    # prop torque
    tau_prop = 1.4 * 10**-5 * states[0, step-1]**2  # propeller load torque

    # integrate motor dynamics
    states[:, step] = rk4(states[:, step-1], u=input_u, tau_load=tau_prop, dt=dt)

    # current physical limit
    states[1, step] = np.clip(states[1, step], 0.0, i_max)


    # controller
    omega = states[0, step]
    current = states[1, step]

    # outer loop: speed controller (omega -> i_ref)
    speed_error[step] = omega_des - omega
    speed_int_error[step] = speed_int_error[step-1] + speed_error[step] * dt

    i_ref_unsat = Kp_w * speed_error[step] + Ki_w * speed_int_error[step]
    i_ref_sat = np.clip(i_ref_unsat, 0.0, i_max)

    # anti-windup for speed integrator
    if i_ref_unsat != i_ref_sat:
        speed_int_error[step] -= speed_error[step] * dt

    i_ref[step] = i_ref_sat

    # inner loop: current controller (i_ref -> u)
    current_error[step] = i_ref[step] - current
    current_int_error[step] = current_int_error[step-1] + current_error[step] * dt

    u_unsat = Kp_i * current_error[step] + Ki_i * current_int_error[step]
    u_sat = np.clip(u_unsat, u_min, u_max)

    # anti-windup for current integrator
    if u_unsat != u_sat:
        current_int_error[step] -= current_error[step] * dt

    input_u = u_sat

    # logging
    voltage[step] = input_u * V_max
    power[step] = voltage[step] * current

elapsed = time.time() - start
print("Simulation finished in " + str(elapsed) + " seconds.")

timeaxis = np.arange(timesteps) * dt

fig = plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
plt.title('Speed (rad/s)')
plt.plot(timeaxis, states[0, :], label='omega')
plt.plot(timeaxis, omega_ref_log, 'k--', label='omega_ref')
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.title('Current (A)')
plt.plot(timeaxis, states[1, :], label='i')
plt.plot(timeaxis, i_ref, 'r--', label='i_ref')
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.title('Voltage (V)')
plt.plot(timeaxis, voltage)
plt.grid()

plt.subplot(2, 2, 4)
plt.title('Power (W)')
plt.plot(timeaxis, power)
plt.grid()

plt.tight_layout()
plt.show()
