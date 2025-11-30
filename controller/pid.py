import numpy as np

class PID:
    def __init__(self, kp, ki, kd, dt, out_min, out_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max

        self.prev_err = 0
        self.integrated_err = 0

        self.setpoint = 0
        self.prev_measurement = 0
        self.err = 0
        self.control = 0

    def update(self, setpoint, measurement, feedforward = 0):
        self.setpoint = setpoint

        self.err = setpoint - measurement

        p = self.kp * self.err

        #d = self.kd * (self.err - self.prev_err) / self.dt
        d = -self.kd * (measurement - self.prev_measurement) / self.dt

        self.integrated_err += self.err * self.dt
        i = self.ki * self.integrated_err

        u = p + i + d + feedforward

        u = np.clip(u, self.out_min, self.out_max)

        self.prev_err = self.err
        self.prev_measurement = measurement
        return u

class AttitudePID:
    def __init__(self, dt):

        self.roll_pid = PID(2, 0, 1, dt, -6, 6)
        self.pitch_pid = PID(2, 0, 1, dt, -6, 6)
        self.yaw_pid = PID(4, 0, 1, dt, -np.inf, np.inf)

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
        
        