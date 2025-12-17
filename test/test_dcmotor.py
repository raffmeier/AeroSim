import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
import numpy as np
from actuator.dcmotor import DCMotor, MotorParam
from controller.motor_velocity_pi import MotorVelocityController
from matplotlib import pyplot as plt

dt = 0.001
timesteps = 100000
V_bat = 50.0
T_amb = 20.0
omega_ref = 3600 * 0.10472
#J_prop1 = 7.15 * 10**-5
J_prop2 = 1.21 * 10**-4

kq_prop1 = 1.36 * 10**-7
kq_prop2 = 1.9 * 10**-7

simulate_electrical_dynamics = False

motor_param = MotorParam()
motor = DCMotor(motor_param, T_amb, J_prop2, simulate_electrical_dynamics)

ctrl = MotorVelocityController(motor, dt, simulate_electrical_dynamics)

# Logs
states = np.zeros((3, timesteps))          # [omega, current, temp]
omega_ref_log = np.zeros(timesteps)
u_log = np.zeros(timesteps)

voltage = np.zeros(timesteps)
bus_current = np.zeros(timesteps)
power_el = np.zeros(timesteps)
power_mech = np.zeros(timesteps)
power_resistive = np.zeros(timesteps)
resistance = np.zeros(timesteps)
iref = np.zeros(timesteps)

u = 0

start = time.time()

for k in range(1, timesteps):

    #if k == 500:
    #    omega_ref = 0
    #if k == 800:
    #    omega_ref = 300

    motor.update(u, V_bat=V_bat, is_braking=ctrl.is_braking)

    tau_prop = kq_prop2 * motor.omega**2

    state_dot = motor.get_state_derivative(motor.get_state(), u, V_bat, tau_prop, 20)

    omega = motor.omega + dt * state_dot[0]
    current = motor.current + dt * state_dot[1]
    temp = motor.temp + dt * state_dot[2]

    motor.set_state(np.array([omega, current, temp]))

    u = ctrl.update(omega_ref, V_bat)

    # --- log using motor.log() ---
    m = motor.log()
    c = ctrl.log()
    states[0, k] = m["omega"]
    states[1, k] = m["current"]
    states[2, k] = m["T_winding"]
    resistance[k] = m["R_tot"]
    iref[k] = c["iref"]

    omega_ref_log[k] = omega_ref
    u_log[k] = u

    # Derived signals (simple/consistent definitions)
    voltage[k] = u * V_bat

    power_el[k] = voltage[k] * m["current"]                       # electrical power into motor terminals
    power_mech[k] = motor.params.k_T * m["current"] * m["omega"]   # mechanical air-gap power (approx)
    power_resistive[k] = (m["current"] ** 2) * m["R_tot"]          # copper + brake resistor if enabled


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
plt.plot(timeaxis, bus_current, label='i_bus')
plt.plot(timeaxis, iref, 'r--', label='i_ref')
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
plt.plot(timeaxis, np.full(timesteps,155), color='red')
plt.grid()

plt.subplot(3, 2, 6)
plt.title('Winding resistance (Ohm)')
plt.plot(timeaxis, resistance)
plt.grid()

plt.tight_layout()
plt.show()