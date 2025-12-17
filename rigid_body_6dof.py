import numpy as np
from common.utils import *
import common.constants as constants

class RigidBody6DOF():

    def __init__(self):
        self.mass = 1.5
        self.inertia = np.diag(np.array([0.0348, 0.0459, 0.0977]))
        self.inv_inertia = np.linalg.inv(self.inertia)

        self.gravity_NED = constants.GRAVITY_NED

        self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) # 3 position, 3 velocity, 4 attitude quaternion, 3 body rates

        self.accel = np.zeros(3)
    
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