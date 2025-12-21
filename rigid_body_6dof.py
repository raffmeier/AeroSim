import numpy as np
from common.utils import *
import common.constants as constants
from logger import Logger
from common.utils import quat_to_euler

class RigidBody6DOF():

    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(self.inertia)

        # State
        self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) # 3 position, 3 velocity, 4 attitude quaternion, 3 body rates
        
        # Extended metrics
        self.accel = np.zeros(3)

        self.gravity_NED = constants.GRAVITY_NED
    
    def get_state_derivative(self, state, force_body, torque_body):
        
        state_dot = np.zeros(13)

        # Unpack state
        # pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        w = state[10:13]

        R_wb = quat_to_R(q) # body to world r_w = R_wb * r_b

        # Linear kinematics
        state_dot[0:3] = vel

        # Linear dynamics: NSL in world frame
        force_world = R_wb @ force_body
        state_dot[3:6] = 1/self.mass * force_world + self.gravity_NED

        # Rotational kinematics
        state_dot[6:10] = 0.5 * omega_matrix_from_q(w) @ q # Quaternion dynamics

        # Rotational dynamics in body frame
        state_dot[10:13] = self.inv_inertia @ (torque_body - np.cross(w, self.inertia @ w)) 

        return state_dot
    
    def post_step(self, force_body):
        # Normalize quaternion
        q = self.state[6:10]
        q /= np.linalg.norm(q)
        self.state[6:10] = q

        # Acceleration
        R = quat_to_R(q)
        self.accel = 1/self.mass * R @ force_body + self.gravity_NED

        # Ground clipping
        if(self.state[2] >= 0):
            self.state[2] = 0
            self.state[5] = 0
            self.accel[2] = 0
    
    def log(self, L: Logger):
        state = self.state
        L.log_vector("pos", state[0:3])
        L.log_vector("vel", state[3:6])
        L.log_vector("quat", state[6:10])
        L.log_vector("rate", state[10:13])
        L.log_vector("accel", self.accel)
        L.log_vector("eulerangles", quat_to_euler(state[6:10]))