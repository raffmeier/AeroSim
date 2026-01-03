import numpy as np
from logger import Logger

class PID:
    def __init__(self, dt, kp, ki, kd, out_min, out_max, derivative_on_measurement = False):
        self.dt = dt

        self.setpoint = 0

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.p = 0
        self.i = 0
        self.d = 0

        self.out_min = out_min
        self.out_max = out_max

        self.err = 0
        self.integrated_err = 0
        self.prev_err = 0

        self.derivative_on_measurement = derivative_on_measurement
        self.prev_measurement = 0

    def update(self, setpoint, measurement, feedforward = 0):
        self.setpoint = setpoint
        self.err = setpoint - measurement

        self.p = self.kp * self.err

        if self.derivative_on_measurement == True:
            self.d = -self.kd * (measurement - self.prev_measurement) / self.dt
        else:
            self.d = self.kd * (self.err - self.prev_err) / self.dt

        self.integrated_err += self.err * self.dt
        self.i = self.ki * self.integrated_err

        u_unsat = self.p + self.i + self.d + feedforward

        u_sat = np.clip(u_unsat, self.out_min, self.out_max)

        # simple anti-windup
        if u_unsat != u_sat:
            self.integrated_err -= self.err * self.dt

        self.prev_err = self.err
        self.prev_measurement = measurement
        return u_sat
    
    def log(self, L: Logger, prefix: str):
        L.log_scalar(f"{prefix}p", self.p)
        L.log_scalar(f"{prefix}i", self.i)
        L.log_scalar(f"{prefix}d", self.d)
        L.log_scalar(f"{prefix}setpoint", self.setpoint)
        L.log_scalar(f"{prefix}err", self.err)
        L.log_scalar(f"{prefix}int_err", self.integrated_err)
        
        