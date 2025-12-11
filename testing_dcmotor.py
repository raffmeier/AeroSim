import numpy as np
from matplotlib import pyplot as plt
import time

sim_time = 45          # s

el_dynamics = False

brake = False

J_mot = 1.3 * 10**-4    # kg*m2
b = 0.00035             # kg*m2/s
k_T = 0.0649            # Nm/A
k_e = k_T
R_w0 = 0.103            # Ohm
R_brake = 1             # Ohm
L = 0.048 * 10**-3      # H

c_thermal = 19.45 # J / K
R_th = 0.55 # K / W
T_amb = 20 # degC

V_bat = 50              # V (max voltage)
i_max = 60              # A
u_min, u_max = 0.0, 1.0

J_prop = 1.79 * 10**-3  # kg*m2 (propeller inertia)
J = J_mot + J_prop

if el_dynamics == True:
    dt = 0.00001    # 10 µs
else:
    dt = 0.001      # 1 ms

timesteps = int(sim_time / dt)

states = np.zeros((3, timesteps))  # [omega, motor (internal) current, winding temp]
states[2, 0] = T_amb # initial winding temperature

# logging
omega_des = 300.0
omega_ref_log = np.zeros(timesteps)
i_ref = np.zeros(timesteps)
voltage = np.zeros(timesteps)
power_el = np.zeros(timesteps)
power_mech = np.zeros(timesteps)
power_resistive = np.zeros(timesteps)
resistance = np.zeros(timesteps)
bus_current = np.zeros(timesteps)

# controller gains
# outer speed loop
Kp_w = 10
Ki_w = 200

# inner current loop
Kp_i = 0.153
Ki_i = 110

speed_error = np.zeros(timesteps)
speed_int_error = np.zeros(timesteps)

current_error = np.zeros(timesteps)
current_int_error = np.zeros(timesteps)


input_u = 0.0
R_w = R_w0 * (1 + 0.00393 * T_amb)  # initial winding resistance
resistance[0] = R_w
R_tot = R_w

def state_dynamics(state, u, tau_load):
    omega, current, temp = state

    if el_dynamics == True:
        V = V_bat * u
        current_dot = (V - R_tot * current - k_e * omega) / L
    else:
        current_dot = 0.0

    omega_dot = (k_T * current - b * omega - tau_load) / J
    temp_dot = 1/c_thermal * ((current ** 2) * R_tot - (temp - T_amb)/R_th)

    return np.array([omega_dot, current_dot, temp_dot])

def rk4(state, u, tau_load, dt):
    k1 = state_dynamics(state, u, tau_load)
    k2 = state_dynamics(state + 0.5*dt*k1, u, tau_load)
    k3 = state_dynamics(state + 0.5*dt*k2, u, tau_load)
    k4 = state_dynamics(state + dt*k3,     u, tau_load)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

start = time.time()

for step in range(1, timesteps):

    # --- Dynamics ---
    omega_prev = states[0, step-1]
    temp_prev = states[2, step-1]
    tau_prop = 1.4 * 10**-5 * omega_prev**2  # propeller load torque
    R_w = R_w0 * (1 + 0.00393 * temp_prev)  # winding resistance

    if not el_dynamics:
        current = (input_u * V_bat - k_e * omega_prev) / R_tot # algebraic current (electrical dynamics + controller are fast)

    braking_mode = brake and (omega_prev > omega_des + 10.0)
    if braking_mode:
        R_tot = R_w + R_brake
        input_u = 0.0           # brake mode
    else:
        R_tot = R_w             # normal mode

    # motor dynamics
    states[:, step] = rk4(states[:, step-1], u=input_u, tau_load=tau_prop, dt=dt)

    omega  = states[0, step]
    if el_dynamics:
        current = states[1, step]
    else:
        # recompute algebraic current at the *new* omega for logging
        # current = (input_u * V_bat - k_e * omega) / R_tot
        states[1, step] = current

    # logging using plant input used this step
    voltage[step] = input_u * V_bat
    power_el[step]   = voltage[step] * current
    resistance[step] = R_tot
    power_mech[step] = k_T * omega * current
    power_resistive[step] = (current ** 2) * R_tot
    bus_current[step] = current

    # --- Controller ---
    #if step > 250:
    #    omega_des = 0

    omega_ref_log[step] = omega_des

    # controller
    if braking_mode:
        # braking mode: no active current control
        speed_error[step] = omega_des - omega
        i_ref[step] = 0.0

        # reset controllers
        speed_int_error[step]   = 0.0                      # hard reset
        current_error[step]     = 0.0
        current_int_error[step] = 0.0

        # logging
        voltage[step] = 0.0
        power_el[step]   = 0.0
        bus_current[step] = 0.0

        continue    # skip the normal PI control this step

    # normal mode: cascaded PI
    # outer loop: speed controller (omega -> i_ref)
    speed_error[step] = omega_des - omega
    speed_int_error[step] = speed_int_error[step-1] + speed_error[step] * dt

    i_ref_unsat = Kp_w * speed_error[step] + Ki_w * speed_int_error[step]
    i_ref_sat = np.clip(i_ref_unsat, 0.0, i_max)

    # anti-windup for speed integrator
    if i_ref_unsat != i_ref_sat:
        speed_int_error[step] -= speed_error[step] * dt

    i_ref[step] = i_ref_sat

    if el_dynamics == True:
        # inner loop: current controller (i_ref -> u)
        current_error[step] = i_ref[step] - current
        current_int_error[step] = current_int_error[step-1] + current_error[step] * dt

        u_unsat = Kp_i * current_error[step] + Ki_i * current_int_error[step]
        u_sat = np.clip(u_unsat, u_min, u_max)

        # anti-windup for current integrator
        if u_unsat != u_sat:
            current_int_error[step] -= current_error[step] * dt

    else:
        # algebraic mapping: i_ref -> u
        u_ff = (R_tot * i_ref[step] + k_e * omega) / V_bat
        u_sat = np.clip(u_ff, u_min, u_max)

    input_u = u_sat


elapsed = time.time() - start

print("Simulation finished in " + str(elapsed) + " seconds.")

timeaxis = np.arange(timesteps) * dt
fig = plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.title('Speed (rad/s)')
plt.plot(timeaxis, states[0, :], label='omega')
plt.plot(timeaxis, omega_ref_log, 'k--', label='omega_ref')
plt.grid()
plt.legend()

plt.subplot(3, 2, 2)
plt.title('Current (A)')
plt.plot(timeaxis, states[1, :], label='i')
plt.plot(timeaxis, i_ref, 'r--', label='i_ref')
plt.plot(timeaxis, bus_current, label='i_bus')
plt.grid()
plt.legend()

plt.subplot(3, 2, 3)
plt.title('Voltage (V)')
plt.plot(timeaxis, voltage)
plt.grid()

plt.subplot(3, 2, 4)
plt.title('Power (W)')
plt.plot(timeaxis, power_el, label='Electrical Power')
plt.plot(timeaxis, power_mech, label='Mechanical Power')
plt.plot(timeaxis, power_resistive, label='Resistive Power')
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.title('Winding temperature')
plt.plot(timeaxis, states[2, :])
plt.grid()

plt.subplot(3, 2, 6)
plt.title('Winding resistance (Ohm)')
plt.plot(timeaxis, resistance)
plt.grid()

plt.tight_layout()
plt.show()