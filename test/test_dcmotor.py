import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
import numpy as np
from actuator.dcmotor import DCMotor, MotorParam
from controller.motor_velocity_pi import MotorVelocityController, MotorVelocityControllerParam
from matplotlib import pyplot as plt
from logger import Logger

dt = 0.001
timesteps = 20000
V_bat = 50.0
T_amb = 20.0
omega_ref = 8000 * 0.10472
J_prop1 = 7.15 * 10**-5
J_prop2 = 1.21 * 10**-4

kq_prop1 = 1.36 * 10**-7
kq_prop2 = 1.9 * 10**-7

omega_ref = 800

simulate_electrical_dynamics = False

motor_param = MotorParam('maxon_ecx32_flat_uav')
motor = DCMotor(motor_param, T_amb, J_prop1, simulate_electrical_dynamics)

ctrl_param = MotorVelocityControllerParam('ctrl_maxon_ecx32_flat_uav')
ctrl = MotorVelocityController(ctrl_param, motor, dt, simulate_electrical_dynamics)

logger = Logger()

# Logs
u = 0
motor.log(logger, "")
ctrl.log(logger, "")


start = time.time()

def step(state, u, dt, V_bat, T_amb, tau_prop):
            k1 = motor.get_state_derivative(state, u, V_bat, tau_prop, T_amb)
            k2 = motor.get_state_derivative(state + 0.5*dt*k1, u, V_bat, tau_prop, T_amb)
            k3 = motor.get_state_derivative(state + 0.5*dt*k2, u, V_bat, tau_prop, T_amb)
            k4 = motor.get_state_derivative(state + dt*k3, u, V_bat, tau_prop, T_amb)
            return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

for k in range(1, timesteps):

    if k == 10000:
        omega_ref = 0
    elif k == 15000:
        omega_ref = 400

    motor.update(u, V_bat=V_bat, is_braking=ctrl.is_braking)

    tau_prop = kq_prop1 * motor.state[0]**2

    state = step(motor.get_state(), u, dt, V_bat, T_amb, tau_prop)

    motor.set_state(state)

    u = ctrl.update(omega_ref, V_bat)

    motor.log(logger, "")
    ctrl.log(logger, "")


elapsed = time.time() - start
print("Simulation finished in " + str(elapsed) + " seconds.")

timeaxis = np.arange(timesteps) * dt
fig = plt.figure(figsize=(10, 10))

omega_log = np.array(logger.data["omega"])
omega_setpoint_log = np.array(logger.data["omegapid.setpoint"])
current_log = np.array(logger.data["current"])
current_setpoint_log = np.array(logger.data["current_ref"])
voltage_log = np.array(logger.data["voltage"])
resistance_log = np.array(logger.data["resistance"])
winding_temp_log = np.array(logger.data["temp"])

plt.subplot(3, 2, 1)
plt.title('Speed (rad/s)')
plt.plot(timeaxis, omega_log, label='omega')
plt.plot(timeaxis, omega_setpoint_log, 'k--', label='omega_ref')
plt.grid()
plt.legend()

plt.subplot(3, 2, 2)
plt.title('Current (A)')
plt.plot(timeaxis, current_log, label='i')
plt.plot(timeaxis, current_setpoint_log, 'r--', label='i_ref')
plt.grid()
plt.legend()

plt.subplot(3, 2, 3)
plt.title('Voltage (V)')
plt.plot(timeaxis, voltage_log)
plt.grid()

plt.subplot(3, 2, 4)
plt.title('Power (W)')
plt.plot(timeaxis, voltage_log * current_log, label='Electrical Power')
# plt.plot(timeaxis, power_mech, label='Mechanical Power')
plt.plot(timeaxis, current_log **2 * resistance_log, label='Resistive Power')
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.title('Winding temperature')
plt.plot(timeaxis, winding_temp_log)
plt.plot(timeaxis, np.full(timesteps,155), color='red')
plt.grid()

plt.subplot(3, 2, 6)
plt.title('Winding resistance (Ohm)')
plt.plot(timeaxis, resistance_log)
plt.grid()

plt.tight_layout()
plt.show()