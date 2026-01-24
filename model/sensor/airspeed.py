import numpy as np

class PitotDiffPressureParam:
    def __init__(self):
        self.rho = 1.1                    # density
        self.noise_std_hpa = 0.02         # hPa white noise std
        self.range_hpa = 100.0            # hPa sensor full-scale
        self.sensitivity_hpa_per_lsb = 0.01  # hPa / LSB (quantization)


class PitotDiffPressureSensor:
    def __init__(self, params: PitotDiffPressureParam):
        self.params = params
        self.scale = 1.0 / self.params.sensitivity_hpa_per_lsb
        self.diff_pressure = 0.0  # hPa

    def step(self, true_airspeed_mps):

        q_pa = 0.5 * self.params.rho * (true_airspeed_mps ** 2)
        q_hpa = q_pa / 100.0

        q_hpa += np.random.normal(0.0, self.params.noise_std_hpa)

        q_hpa = np.clip(q_hpa, 0.0, self.params.range_hpa)
        q_hpa = np.round(q_hpa * self.scale) / self.scale

        self.diff_pressure = float(q_hpa)
        return self.diff_pressure