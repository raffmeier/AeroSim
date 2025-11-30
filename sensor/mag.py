import numpy as np
import constants

class MagParam():
    def __init__(self):
        self.mag_noise_density = 0.0006 #Gauss / sqrt(Hz)
        self.mag_range = 49.152 #plus and minus Gauss
        self.sensitivity = 0.0015 #Gauss/LSB

        
class Magnetometer:
    def __init__(self, params: MagParam, dt):
        self.params = params
        self.dt = dt
        self.mag_scale = 1.0 / self.params.sensitivity

        self.meas_mag = np.zeros(3)
    
    def step(self, R_wb):

        #Magnetic field
        mag_body = R_wb.T @ constants.HOME_B_NED #transform to body frame

        mag_white_noise = np.random.normal(0, self.params.mag_noise_density, size=3) / np.sqrt(self.dt)

        self.meas_mag = mag_body + mag_white_noise
        self.meas_mag = np.clip(self.meas_mag, -self.params.mag_range, self.params.mag_range) #clip to sensor range
        self.meas_mag = np.round(self.meas_mag * self.mag_scale) / self.mag_scale #quantization
