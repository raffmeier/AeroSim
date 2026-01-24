import numpy as np
from logger import Logger

class ControlSurface():

    def __init__(self):
        self.state = 0.0 # angle (rad)
        self.angle_des = 0.0

        self.tau = 0.1
        self.rate_limit = 6
    
    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state
    
    def set_angle_des(self, angle_des):
        self.angle_des = angle_des

    def get_state_derivative(self, state):
        angle = state

        angle_dot_raw = (self.angle_des - angle) / self.tau
        angle_dot = np.clip(angle_dot_raw, -self.rate_limit, self.rate_limit)

        return angle_dot

    def log(self, L: Logger, prefix: str):
        L.log_scalar(f"{prefix}angle", self.state)
        L.log_scalar(f"{prefix}setpoint", self.angle_des)