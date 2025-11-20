import numpy as np

class IMUParam():
    def __init__(self):
        self.gyro_noise_density = np.deg2rad(0.014) #rad/s / sqrt(Hz)
        self.accel_noise_density = 0.00186 #m/s2 / sqrt(Hz)

        self.accel_range = 24 * 9.81 #plus and minus m/s2
        self.accel_bits = 16

        self.gyro_range = np.deg2rad(2000) #plus and minus rad/s
        self.gyro_bits = 16

        self.accel_random_walk = 0.00117 #m/s3 / sqrt(Hz)
        self.gyro_random_walk = np.deg2rad(0.2) #rad/s2 / sqrt(Hz)
        
class IMU:
    def __init__(self, params: IMUParam, dt):
        self.params = params
        self.dt = dt

        self.meas_accel = np.zeros(3)
        self.meas_rates = np.zeros(3)

        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        self.accel_scale = (2**(self.params.accel_bits - 1) - 1) / self.params.accel_range
        self.gyro_scale = (2**(self.params.gyro_bits - 1) - 1) / self.params.gyro_range
    
    def step(self, accel_world, body_rates, R_wb):

        #Linear acceleration
        accel_body = R_wb.T @ accel_world #transform to body frame

        accel_white_noise = np.random.normal(0, self.params.accel_noise_density, size=3) / np.sqrt(self.dt)
        self.accel_bias += np.random.normal(0, self.params.accel_random_walk, size=3) * np.sqrt(self.dt)
        accel_noise = accel_white_noise + self.accel_bias

        meas_accel = accel_body + accel_noise
        meas_accel = np.clip(meas_accel, -self.params.accel_range, self.params.accel_range) #clip to sensor range
        meas_accel = np.round(meas_accel * self.accel_scale) / self.accel_scale #quantization
        self.meas_accel = meas_accel

        #Angular rates
        rates_white_noise = np.random.normal(0, self.params.gyro_noise_density, size=3) / np.sqrt(self.dt)
        self.gyro_bias += np.random.normal(0, self.params.gyro_random_walk, size=3) * np.sqrt(self.dt)
        rates_noise = rates_white_noise + self.gyro_bias

        meas_rates = body_rates + rates_noise
        meas_rates = np.clip(meas_rates, -self.params.gyro_range, self.params.gyro_range) #clip to sensor range
        meas_rates = np.round(meas_rates * self.gyro_scale) / self.gyro_scale #quantization
        self.meas_rates = meas_rates
