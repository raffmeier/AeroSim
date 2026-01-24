import numpy as np
from common.utils import quat_to_R

class GroundContact():

    def __init__(self, z0, mu_lin, mu_ang):
        self.z0 = z0
        self.linear_friction = mu_lin
        self.angular_friction = mu_ang
    
    def get_friction_force_torque(self, state):

        pos = state[0:3]
        vel = state[3:6]
        q   = state[6:10]
        w   = state[10:13]

        if pos[2] < self.z0:
            return np.zeros(3), np.zeros(3), False
    
        R_wb = quat_to_R(q)

        # linear friction
        F_fric_world = np.array([-self.linear_friction * vel[0], -self.linear_friction * vel[1], 0.0])
        F_fric_body = R_wb.T @ F_fric_world

        # angular friction
        tau_body = -self.angular_friction * w

        return F_fric_body, tau_body, True