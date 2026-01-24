import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
import numpy as np
from model.actuator.dcmotor import DCMotor, MotorParam
from controller.motor_velocity_pi import MotorVelocityController, MotorVelocityControllerParam
from matplotlib import pyplot as plt
from logger import Logger
from plot import plot_motor_dashboard

dt = 0.001
timesteps = 100000
V_bat = 50.0
T_amb = 20.0
J_prop1 = 7.15 * 10**-5
J_prop2 = 1.21 * 10**-4

kq_prop1 = 1.36 * 10**-7
kq_prop2 = 1.9 * 10**-7
kq_prop3 = 1.386 * 10**-5

omega_ref = 8200 * 0.10472

simulate_electrical_dynamics = False

motor_param = MotorParam('maxon_ecx32_flat_uav')
motor = DCMotor(motor_param, T_amb, J_prop2, simulate_electrical_dynamics)

ctrl_param = MotorVelocityControllerParam('ctrl_maxon_ecx32_flat_uav')
ctrl = MotorVelocityController(ctrl_param, motor, dt, simulate_electrical_dynamics)

logger = Logger()

# Logs
u = 0
logger.log_scalar('t', 0.0)
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

    # if k == 5000:
    #     omega_ref = 0
    # elif k == 15000:
    #     omega_ref = 400

    motor.update(u, V_bat=V_bat, is_braking=ctrl.is_braking)

    tau_prop = kq_prop2 * motor.state[0]**2

    state = step(motor.get_state(), u, dt, V_bat, T_amb, tau_prop)

    motor.set_state(state)

    u = ctrl.update(omega_ref, V_bat)

    logger.log_scalar('t', k*dt)
    motor.log(logger, "")
    ctrl.log(logger, "")


elapsed = time.time() - start
print("Simulation finished in " + str(elapsed) + " seconds.")

plot_motor_dashboard(logger.data, '')
plt.show()