import numpy as np
from vehicle.vehicle import Vehicle
import os, json
from rigid_body_6dof import RigidBody6DOF
from propeller import Propeller, PropellerParam
from actuator.dcmotor import DCMotor, MotorParam
from ground import GroundContact
from battery import Battery
from actuator.control_surface import ControlSurface
from common.utils import quat_to_R, wind_to_body_matrix
from logger import Logger

class FixedWingParam():
    def __init__(self, fw_name: str):

        base_dir = os.path.dirname(__file__)
        fw_file = os.path.join(
            base_dir,
            "..",
            "parameter",
            "fixedwing",
            f"{fw_name}.json"
        )

        if not os.path.isfile(fw_file):
            raise FileNotFoundError(
                f"Fixedwing file not found: {fw_file}"
            )

        with open(fw_file, "r") as f:
            data = json.load(f)


        # --- load parameters ---

        self.mass = data["mass"]

        I = data["inertia"]
        self.inertia = np.array([
            [I["ixx"], I["ixy"], I["ixz"] ],
            [I["ixy"], I["iyy"], I["iyz"] ],
            [I["ixz"], I["iyz"], I["izz"] ]
        ])

        self.motor_param_name = data["motor"]
        self.propeller_param_name = data["propeller"]

        self.sref = data["sref"]
        self.chord = data["chord"]
        self.span = data["span"]

        self.cL0 = data["cL0"]
        self.cLa = data["cLa"]
        self.cLq = data["cLq"]
        self.cLde = -data["cLde"] * 180 / np.pi # AVL control surface coefficients are 1/deg, convert here to 1/rad

        self.cD0 = data["cD0"]
        self.k = data["k"]

        self.cYb = data["cYb"]
        self.cYp = data["cYp"]
        self.cYr = data["cYr"]
        self.cYda = -data["cYda"] * 180 / np.pi
        self.cYdr = data["cYdr"] * 180 / np.pi

        self.clb = data["clb"]
        self.clp = data["clp"]
        self.clr = data["clr"]
        self.clda = -data["clda"] * 180 / np.pi
        self.cldr = data["cldr"] * 180 / np.pi

        self.cm0 = data["cm0"]
        self.cma = data["cma"]
        self.cmq = data["cmq"]
        self.cmde = -data["cmde"] * 180 / np.pi # PX4 assumes positive deflection -> trailing edge up (doesnt match AVL --> *-1)

        self.cnb = data["cnb"]
        self.cnp = data["cnp"]
        self.cnr = data["cnr"]
        self.cnda = -data["cnda"] * 180 / np.pi
        self.cndr = data["cndr"] * 180 / np.pi


class FixedWing(Vehicle):

    def __init__(self, params: FixedWingParam, initial_state):
        
        self.params = params

        self.rb = RigidBody6DOF(mass=self.params.mass, inertia=self.params.inertia, initial_state=initial_state)

        pp = PropellerParam(self.params.propeller_param_name)
        self.prop = Propeller(pp, model='dynamic')

        mp = MotorParam(self.params.motor_param_name)
        self.motor = DCMotor(mp, T_amb=20, J_load=pp.J_prop, simulate_electrical_dynamics=False)

        self.control_surface = [ControlSurface() for _ in range(4)] # Left aileron, Right aileron, Elevator, Rudder

        self.ground_contact = GroundContact(z0=0.0, mu_lin=5.0, mu_ang=5.0)

        self.battery = Battery(dt=0.001)

        self.rho = 1.225 # ToDo: Implement environment model
        self.min_airspeed_aero_model = 0.5

        self.alpha=0.0
        self.beta=0.0

        self.lift=0.0
        self.drag=0.0
        self.side=0.0

        self.roll_mom = 0.0
        self.pitch_mom = 0.0
        self.yaw_mom = 0.0

        self.prop_thrust = 0.0
        self.prop_torque = 0.0

    def get_state(self):
        rb_state = self.rb.state
        motor_state = self.motor.get_state()
        cs_state = np.hstack([cs.get_state() for cs in self.control_surface])

        return np.hstack((rb_state, motor_state, cs_state))
    
    def set_state(self, state):
        self.rb.state = state[0:13]
        self.motor.set_state(state[13:17])

        cs_state = state[17:21]
        for i, cs in enumerate(self.control_surface):
            cs.set_state(cs_state[i])
    
    def get_state_derivative(self, state, u, T_amb):

        rb_state = state[0:13]
        motor_state = state[13:17]
        cs_state = state[17:21]

        force_body, torque_body, torque_prop = self.compute_vehicle_forces_torques(state)

        rb_state_dot = self.rb.get_state_derivative(rb_state, force_body, torque_body)
        motor_state_dot = self.motor.get_state_derivative(motor_state, u[0], self.battery.get_pack_voltage(), torque_prop, T_amb)

        cs_state_dot = np.zeros(4)
        for i, cs in enumerate(self.control_surface):
            cs_state_dot[i] = cs.get_state_derivative(cs_state[i])

        state_dot = np.hstack((rb_state_dot, motor_state_dot, cs_state_dot))

        return state_dot
    
    def compute_vehicle_forces_torques(self, state):

        force_body = np.zeros(3)
        torque_body = np.zeros(3)

        # Unpack state
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        w = state[10:13]
        omega = state[13]
        defl_ail_l = state[17]
        defl_ail_r = state[18]
        defl_elev = state[19]
        defl_rudd = state[20]

        R_wb = quat_to_R(q)

        vel_body = R_wb.T @ vel #ToDo: Add environment wind

        airspeed = np.linalg.norm(vel_body)
        qbar = 0.5 * self.rho * airspeed**2 # Dynamic pressure

        if airspeed >= self.min_airspeed_aero_model:
            alpha = np.arctan2(vel_body[2], vel_body[0]) # Angle of attack
            beta = np.arctan2(vel_body[1], np.sqrt(vel_body[0]**2 + vel_body[2]**2)) # Side slip angle
        else:
            alpha = 0.0
            beta = 0.0

        airspeed_denom = max(airspeed, self.min_airspeed_aero_model)

        phat = (w[0] * self.params.span) / (2 * airspeed_denom)
        qhat = (w[1] * self.params.chord) / (2 * airspeed_denom)
        rhat = (w[2] * self.params.span) / (2 * airspeed_denom)

        cL = ((self.params.cL0) 
              + (self.params.cLa * alpha) 
              + (self.params.cLq * qhat) 
              + (self.params.cLde * defl_elev)
              )
        
        cD = ((self.params.cD0) 
              + (self.params.k * cL**2)
              )
        
        cY = (
            (self.params.cYb * beta) 
              + (self.params.cYp * phat) 
              + (self.params.cYr * rhat) 
              + (self.params.cYda * defl_ail_r) 
              + (-self.params.cYda * defl_ail_l) # Minus sign as the aileron coefficients are defined for the right aileron only
              + (self.params.cYdr * defl_rudd)
              )
        
        cl = (
            (self.params.clb * beta)
              + (self.params.clp * phat)
              + (self.params.clr * rhat)
              + (self.params.clda * defl_ail_r)
              + (-self.params.clda * defl_ail_l) # Minus sign as the aileron coefficients are defined for the right aileron only
              + (self.params.cldr * defl_rudd)
               )
        
        cm = (
            (self.params.cm0)
              + (self.params.cma * alpha)
              + (self.params.cmq * qhat)
              + (self.params.cmde * defl_elev)
              )
        
        cn = (
            (self.params.cnb * beta)
              + (self.params.cnp * phat)
              + (self.params.cnr * rhat)
              + (self.params.cnda * defl_ail_r)
              + (-self.params.cnda * defl_ail_l) # Minus sign as the aileron coefficients are defined for the right aileron only
              + (self.params.cndr * defl_rudd)
              )
        
        F_lift_wind = qbar * self.params.sref * cL
        F_drag_wind = qbar * self.params.sref * cD
        F_side_wind = qbar * self.params.sref * cY
        
        force_body = wind_to_body_matrix(alpha, beta) @ np.array([-F_drag_wind, F_side_wind, -F_lift_wind])

        torque_roll = qbar * self.params.sref * self.params.span * cl
        torque_pitch = qbar * self.params.sref * self.params.chord * cm
        torque_yaw = qbar * self.params.sref * self.params.span * cn

        torque_body = np.array([torque_roll, torque_pitch, torque_yaw])

        thrust_prop, torque_prop = self.prop.get_force_torque(omega, airspeed)


        force_body[0] += thrust_prop
        torque_body[0] += torque_prop

        self.alpha = alpha
        self.beta = beta
        self.lift=F_lift_wind
        self.drag=F_drag_wind
        self.side=F_side_wind

        self.roll_mom = torque_roll
        self.pitch_mom = torque_pitch
        self.yaw_mom = torque_yaw

        self.prop_thrust = thrust_prop
        self.prop_torque = torque_prop

        return force_body, torque_body, torque_prop

    def pre_step(self, u, is_braking):
        self.motor.update(u[0], self.battery.get_pack_voltage(), is_braking)

        u_lail = u[1] * np.deg2rad(20) # Left aileron [-1.0, 1.0] -> 20deg
        u_rail = u[2] * np.deg2rad(20) # Right aileron [-1.0, 1.0] -> 20deg
        u_elev = u[3] * np.deg2rad(20) # Elevator [-1.0, 1.0] -> 20deg
        u_rudd = u[4] * np.deg2rad(20) # Rudder [-1.0, 1.0] -> 20deg

        self.control_surface[0].set_angle_des(u_lail)
        self.control_surface[1].set_angle_des(u_rail)
        self.control_surface[2].set_angle_des(u_elev)
        self.control_surface[3].set_angle_des(u_rudd)
    
    def post_step(self):
        force_body, _, _ = self.compute_vehicle_forces_torques(self.get_state()) # Recalculate the forces to get the acceleration at final state
        self.rb.post_step(force_body)

        # Battery update
        motor_el_power = np.zeros(4)
        motor_el_power = self.motor.state[1] * self.motor.voltage
        self.battery.update(motor_el_power) 
    
    def get_accel(self):
        return self.rb.accel
    
    def log(self, L: Logger, time):
        L.log_scalar("t", time)
        L.log_scalar("alpha", self.alpha)
        L.log_scalar("beta", self.beta)
        L.log_scalar("F_lift", self.lift)
        L.log_scalar("F_drag", self.drag)
        L.log_scalar("F_side", self.side)
        L.log_scalar("roll_mom", self.roll_mom)
        L.log_scalar("pitch_mom", self.pitch_mom)
        L.log_scalar("yaw_mom", self.yaw_mom)

        L.log_scalar("prop_thrust", self.prop_thrust)
        L.log_scalar("prop_torque", self.prop_torque)
        
        self.rb.log(L)
        self.battery.log(L)
        self.motor.log(L, "m0.")

        for i, cs in enumerate(self.control_surface):
            cs.log(L, f"cs{i}.")
    
    def get_airspeed(self, state):
        vel = state[3:6]
        q = state[6:10]

        R_wb = quat_to_R(q)

        vel_body = R_wb.T @ vel
        airspeed = np.linalg.norm(vel_body)

        return airspeed
    
    def get_motor(self):
        return np.array([self.motor])