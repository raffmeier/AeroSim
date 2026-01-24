import numpy as np
import os
import json
from logger import Logger

class MotorParam():

    def __init__(self, motor_name: str):

        base_dir = os.path.dirname(__file__)
        motor_file = os.path.join(
            base_dir,
            "..",     
            "..",
            "parameter",
            "motor",
            f"{motor_name}.json"
        )

        if not os.path.isfile(motor_file):
            raise FileNotFoundError(
                f"Motor file not found: {motor_file}"
            )

        with open(motor_file, "r") as f:
            data = json.load(f)

        self.R_brake = 1                # Brake resistor, Ohm

        self.J_mot = data["J_mot"]      # Motor inertia, kg*m2
        self.b = data["b"]              # Motor damping constant, kg*m2/s

        self.L = data["L"]              # Motor inductance, H
        self.k_T = data["k_T"]          # Torque constant, Nm/A
        self.k_e = data["k_e"]          # Back EMF constant, Nm/A
        self.R_w0 = data["R_w0"]        # Winding resistance at T_rw0, Ohm
        self.T_rw0 = data["T_Rw0"]      # Temperature for winding resistance value, degC

        self.C_th_w = data["C_th_w"]    # Winding thermal capacitance, J/K
        self.R_th_wh = data["R_th_wh"]  # Thermal resistance winding <-> motor housing, K/W

        self.C_th_h = data["C_th_h"]    # Motor housing thermal capacitance, J/K
        self.R_th_ha = data["R_th_ha"]  # Thermal resistance motor housing <-> ambient, K/W
    
class DCMotor():

    def __init__(self, params: MotorParam, T_amb, J_load, simulate_electrical_dynamics=False):
        # Sim parameters
        self.simulate_electrial_dynamics = simulate_electrical_dynamics

        # Motor parameters
        self.params = params

        # State
        self.state = np.array([0.0, 0.0, float(T_amb), float(T_amb)], dtype=np.float64) # omega (rad/s), RMS motor current (A), winding temperature (degC), motor housing temperature (degC)

        # Extended metrics
        self.voltage = 0                 # RMS terminal voltage V, 0..Vbat
        self.J_tot = self.params.J_mot + J_load
        self.R_tot = self._get_winding_resistance()

    def get_state(self):
        return self.state
    
    def set_state(self, state):
        #If electrical dynamics are simulated -> take current from integrator, otherwise keep algebraic current
        if self.simulate_electrial_dynamics:
            self.state[1] = state[1]
        
        self.state[0] = state[0]
        self.state[2] = state[2]
        self.state[3] = state[3]
    
    def get_state_derivative(self, state, u, V_bat, tau_load, T_amb):

        # Unpack state
        omega = state[0]
        current = state[1]
        temp_winding = state[2]
        temp_housing = state[3]

        if self.simulate_electrial_dynamics == True:
            V = V_bat * u
            current_dot = (V - self.R_tot * current - self.params.k_e * omega) / self.params.L
        else:
            current_dot = 0.0

        omega_dot = (self.params.k_T * current - self.params.b * omega - tau_load) / self.J_tot
        temp_winding_dot = (current** 2 * self.R_tot  - (temp_winding - temp_housing) / self.params.R_th_wh) / self.params.C_th_w
        temp_housing_dot = (omega**2 * self.params.b + (temp_winding - temp_housing) / self.params.R_th_wh - (temp_housing - T_amb) / self.params.R_th_ha) / self.params.C_th_h

        return np.array([omega_dot, current_dot, temp_winding_dot, temp_housing_dot])
    
    def update(self, u, V_bat, is_braking):
        self.voltage = u * V_bat

        R_w = self._get_winding_resistance()

        if is_braking:
            self.R_tot = R_w + self.params.R_brake
            self.voltage = 0.0           # brake mode
        else:
            self.R_tot = R_w             # normal mode

        if not self.simulate_electrial_dynamics:
            current = (self.voltage - self.params.k_e * self.state[0]) / self.R_tot # algebraic current (electrical dynamics + controller are fast)
            self.state[1] = current

    def log(self, L: Logger, prefix: str):
        state = self.state
        L.log_scalar(f"{prefix}omega", state[0])
        L.log_scalar(f"{prefix}current", state[1])
        L.log_scalar(f"{prefix}temp_winding", state[2])
        L.log_scalar(f"{prefix}temp_housing", state[3])
        L.log_scalar(f"{prefix}voltage", self.voltage)
        L.log_scalar(f"{prefix}resistance", self.R_tot)
        L.log_scalar(f"{prefix}resistive_loss", self.R_tot * state[1]**2)
        L.log_scalar(f"{prefix}mechanical_loss", self.params.b * state[0]**2)
    
    def _get_winding_resistance(self):
        return self.params.R_w0 * (1 + 0.00393 * (self.state[2] - self.params.T_rw0))

