import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from controller.pid import PID
from actuator.dcmotor import DCMotor

class MotorVelocityController():

    def __init__(self, motor: DCMotor, dt, simulate_electrical_dynamics=False):
        # controller gains
        # outer speed loop
        Kp_w = 10
        Ki_w = 200

        # inner current loop
        Kp_i = 0.153
        Ki_i = 110

        self.motor = motor

        self.simulate_electrical_dynamics = simulate_electrical_dynamics

        self.dynamic_braking_enabled = False
        self.is_braking = False
        self.dynamic_braking_threshold = 10 # rad/s

        self.imax = 60
        
        self.current_pid = PID(dt, kp=Kp_i, ki=Ki_i, kd=0, out_min=0, out_max=1, derivative_on_measurement=False)
        self.omega_pid = PID(dt, kp=Kp_w, ki=Ki_w, kd=0, out_min=0, out_max=self.imax, derivative_on_measurement=False)
    
    def update(self, omega_ref, V_bat):

        omega_meas = self.motor.omega
        i_meas = self.motor.current

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
            u_ff = (self.motor.R_tot * self.i_ref + self.motor.params.k_e * self.motor.omega) / V_bat
            u = np.clip(u_ff, 0, 1)
        
        return u

    def log(self):
        log = {
            "iref": self.i_ref,
            "is_braking": self.is_braking,
            "omega_ctrl": self.omega_pid.log(),
            "current_ctrl": self.current_pid.log()
        }
        return log