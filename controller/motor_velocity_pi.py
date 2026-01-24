import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from controller.pid import PID
from model.actuator.dcmotor import DCMotor
from logger import Logger
import json

class MotorVelocityControllerParam():
    def __init__(self, motor_ctrl_name: str):

        base_dir = os.path.dirname(__file__)
        motor_ctrl_file = os.path.join(
            base_dir,
            "..",
            "parameter",
            "motorctrl",
            f"{motor_ctrl_name}.json"
        )

        if not os.path.isfile(motor_ctrl_file):
            raise FileNotFoundError(
                f"Motorcontroller file not found: {motor_ctrl_file}"
            )

        with open(motor_ctrl_file, "r") as f:
            data = json.load(f)

        self.Kp_w = data["Kp_omega"]
        self.Ki_w = data["Ki_omega"]
        self.Kp_i = data["Kp_current"]
        self.Ki_i = data["Ki_current"]
        self.i_max = data["i_max"]
        self.omega_max = data["omega_max"]

class MotorVelocityController():

    def __init__(self, params: MotorVelocityControllerParam, motor: DCMotor, dt, command_type="omega_ref", simulate_electrical_dynamics=False):

        self.motor = motor

        # Motor controller parameters
        self.params = params

        self.command_type = command_type # "omega_ref" or "actuator_effort"

        self.simulate_electrical_dynamics = simulate_electrical_dynamics

        self.dynamic_braking_enabled = False
        self.is_braking = False
        self.dynamic_braking_threshold = 10 # rad/s

        self.i_ref = 0
        
        self.current_pid = PID(dt, kp=self.params.Kp_i, ki=self.params.Ki_i, kd=0, out_min=0, out_max=1, derivative_on_measurement=False)
        self.omega_pid = PID(dt, kp=self.params.Kp_w, ki=self.params.Ki_w, kd=0, out_min=0, out_max=self.params.i_max, derivative_on_measurement=False)
    
    def update(self, command, V_bat):

        if self.command_type == "actuator_effort":
            omega_ref = command * self.params.omega_max
        else:
            omega_ref = command

        omega_meas = self.motor.state[0]
        i_meas = self.motor.state[1]

        if self.dynamic_braking_enabled and omega_meas > omega_ref + self.dynamic_braking_threshold:
            # turn off controller
            # TO DO: reset pid controllers
            self.is_braking = True
            u = 0.0
            self.i_ref = 0.0
            return u
        else:
            self.is_braking = False
        
        self.i_ref = self.omega_pid.update(omega_ref, omega_meas, feedforward=0)

        if self.simulate_electrical_dynamics == True:
            u = self.current_pid.update(self.i_ref, i_meas, feedforward=0)
        else:
            # algebraic mapping: i_ref -> u
            u_ff = (self.motor.R_tot * self.i_ref + self.motor.params.k_e * self.motor.state[0]) / V_bat
            u = np.clip(u_ff, 0, 1)
        
        return u

    def log(self, L: Logger, prefix: str):
        L.log_scalar(prefix + "current_ref", self.i_ref)
        self.omega_pid.log(L, prefix + "omegapid.")