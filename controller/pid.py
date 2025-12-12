import numpy as np
class PID:
    def __init__(self, dt, kp, ki, kd, out_min, out_max, derivative_on_measurement = False):
        self.dt = dt

        self.setpoint = 0

        self.kp = kp
        self.ki = ki
        self.kd = kd

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

        p = self.kp * self.err

        if self.derivative_on_measurement == True:
            d = -self.kd * (measurement - self.prev_measurement) / self.dt
        else:
            d = self.kd * (self.err - self.prev_err) / self.dt

        self.integrated_err += self.err * self.dt
        i = self.ki * self.integrated_err

        u_unsat = p + i + d + feedforward

        u_sat = np.clip(u_unsat, self.out_min, self.out_max)

        # simple anti-windup
        if u_unsat != u_sat:
            self.integrated_err -= self.err * self.dt

        self.prev_err = self.err
        self.prev_measurement = measurement
        return u_sat
        
        