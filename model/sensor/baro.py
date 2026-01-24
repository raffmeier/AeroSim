import numpy as np

class BaroParam():
    def __init__(self):
        self.baro_noise_density = 0.001 #mbar
        self.baro_range = 1100 #mbar
        self.sensitivity = 0.0001 #mbar/LSB


class Barometer:
    def __init__(self, params: BaroParam, dt):
        self.params = params
        self.dt = dt
        self.baro_scale = 1.0 / self.params.sensitivity

        self.meas_baro = 0.0

    def step(self, pressure):

        baro_white_noise = np.random.normal(0, self.params.baro_noise_density)

        self.meas_baro = pressure + baro_white_noise
        self.meas_baro = np.clip(self.meas_baro, 0, self.params.baro_range) #clip to sensor range
        self.meas_baro = np.round(self.meas_baro * self.baro_scale) / self.baro_scale #quantization
