import numpy as np
from .pid import PID

class AttitudePID:
    def __init__(self, dt):

        self.roll_pid = PID(dt, 2, 0, 1, -6, 6, True)
        self.pitch_pid = PID(dt, 2, 0, 1, -6, 6, True)
        self.yaw_pid = PID(dt, 4, 0, 1, -np.inf, np.inf, True)

        self.mixer_matrix = np.array([[0, 0.5, -0.25, 0.25],
                                      [-0.5, 0, 0.25, 0.25],
                                      [0, -0.5, -0.25, 0.25],
                                      [0.5, 0, 0.25, 0.25]])
        
        self.motor_commands = np.zeros(4)
    
    def angle_loop(self, setpoint, measurement, feedforward = np.zeros(3)):
        u_roll = self.roll_pid.update(setpoint[0], measurement[0], feedforward[0])
        u_pitch = self.pitch_pid.update(setpoint[1], measurement[1], feedforward[1])
        u_yaw = self.yaw_pid.update(setpoint[2], measurement[2], feedforward[2])

        return np.array([u_roll, u_pitch, u_yaw])
    
    def update(self, angle_setpoint, angle_measurement, angle_feedforward, throttle):

        u = self.angle_loop(angle_setpoint, angle_measurement, angle_feedforward)

        u = self.mixer_matrix @ np.hstack((u, throttle))

        u = np.clip(u, 0, np.inf)

        self.motor_commands = np.sqrt(u/(8.54858 * 10**-6))

        return self.motor_commands