import numpy as np
from rigid_body_6dof import RigidBody6DOF
from actuator.dcmotor import DCMotor, MotorParam
from propeller import Propeller, PropellerParam
from common.utils import *
from vehicle.vehicle import Vehicle
from logger import Logger
import os
import json

class MulticopterParam():

    def __init__(self, multicopter_name: str):

        base_dir = os.path.dirname(__file__)
        multicopter_file = os.path.join(
            base_dir,
            "..",
            "parameter",
            "multicopter",
            f"{multicopter_name}.json"
        )

        if not os.path.isfile(multicopter_file):
            raise FileNotFoundError(
                f"Multicopter file not found: {multicopter_file}"
            )

        with open(multicopter_file, "r") as f:
            data = json.load(f)


        # --- load parameters ---

        self.mass = data["mass"]

        I = data["inertia"]
        self.inertia = np.array([
            [I["ixx"], I["ixy"], I["ixz"] ],
            [I["ixy"], I["iyy"], I["iyz"] ],
            [I["ixz"], I["iyz"], I["izz"] ]
        ])

        CdA = data["CdA"]
        self.CdA = np.array([CdA["CdAx"], CdA["CdAy"], CdA["CdAz"]])
        self.arm = data["arm_length"]

        self.motor_param_name = data["motor"]
        self.propeller_param_name = data["propeller"]


class Multicopter(Vehicle):

    def __init__(self, params: MulticopterParam):

        self.params = params
        
        self.rb = RigidBody6DOF(self.params.mass, self.params.inertia)

        pp = PropellerParam(self.params.propeller_param_name)
        #self.prop = Propeller(pp) <-- ToDo: Implement propeller model in propeller.py
        #For now: Just get the propeller parameter values here

        self.kT = pp.kT
        self.kQ = pp.kQ

        mp = MotorParam(self.params.motor_param_name)
        self.motors = [DCMotor(mp, T_amb=20, J_load=pp.J_prop, simulate_electrical_dynamics=False) for _ in range(4)]

        self.rho = 1.225 # ToDo: Implement environment model
    
    def get_state(self):
        rb_state = self.rb.state
        motor_state = np.hstack([motor.get_state() for motor in self.motors])

        return np.hstack((rb_state, motor_state))
    
    def set_state(self, state):
        self.rb.state = state[0:13]

        motor_state = state[13:]
        for i, motor in enumerate(self.motors):
            motor.set_state(motor_state[3*i : 3*i + 3])
    
    def get_state_derivative(self, state, u, V_bat, T_amb):

        rb_state = state[0:13]
        motor_state = state[13:25]

        force_body, torque_body, torque_prop = self.compute_vehicle_forces_torques(state)

        rb_state_dot = self.rb.get_state_derivative(rb_state, force_body, torque_body)
        
        motor_state_dot = np.zeros(12)
        for i, motor in enumerate(self.motors):
            motor_state_dot[3*i : 3*i+3] = motor.get_state_derivative(motor_state[3*i : 3*i+3], u[i], V_bat, torque_prop[i], T_amb)

        state_dot = np.hstack((rb_state_dot, motor_state_dot))

        return state_dot

    def compute_vehicle_forces_torques(self, state):

        force_body = np.zeros(3)
        torque_body = np.zeros(3)

        # Unpack state
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        w = state[10:13]

        omega = state[13::3]

        R_wb = quat_to_R(q)

        thrust_prop, torque_prop = self.compute_prop_force_torque(omega)

        # Forces
        F_thrust = np.array([0, 0, -np.sum(thrust_prop)])

        vel_body = R_wb.T @ vel
        F_drag = -0.5 * self.rho * self.params.CdA * vel_body * np.abs(vel_body)

        force_body = F_thrust + F_drag

        # Torques
        torque_actuators = self.torque_mixer_quad_plus(thrust_prop, torque_prop, self.params.arm)

        # To Do: add aerodynamic rotational drag

        torque_body = torque_actuators

        return force_body, torque_body, torque_prop

    # ToDo: Put into propeller class
    def compute_prop_force_torque(self, omega):

        omega_squared = np.square(omega)

        motor_thrust = self.kT * omega_squared
        motor_torque = self.kQ * omega_squared

        return motor_thrust, motor_torque
    
    def torque_mixer_quad_plus(self, motor_thrust, motor_torque, arm):
        roll_torque = arm * (-motor_thrust[1] + motor_thrust[3])
        pitch_torque = arm * (motor_thrust[0] - motor_thrust[2])
        yaw_torque = -motor_torque[0] + motor_torque[1] - motor_torque[2] + motor_torque[3]

        return np.array([roll_torque, pitch_torque, yaw_torque])
    
    def pre_step(self, u, V_bat, is_braking):
        for i, motor in enumerate(self.motors):
            motor.update(u[i], V_bat, is_braking[i])

    def post_step(self):
        force_body, _, _ = self.compute_vehicle_forces_torques(self.get_state()) # Recalculate the forces to get the acceleration at final state
        self.rb.post_step(force_body)

    def get_accel(self):
        return self.rb.accel
    
    def log(self, L: Logger, time):
        L.log_scalar("t", time)
        
        self.rb.log(L)

        for i, motor in enumerate(self.motors):
            motor.log(L, f"m{i}.")
