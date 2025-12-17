import numpy as np
import os
import json

class MotorParam():

    def __init__(self, motor_name: str):

        base_dir = os.path.dirname(__file__)
        motor_file = os.path.join(
            base_dir,
            "..",          # actuator
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

        self.R_brake = 1             # Brake resistor, Ohm

        self.J_mot = data["J_mot"]   # Motor inertia, kg*m2
        self.b = data["b"]       # Motor damping constant, kg*m2/s

        self.L = data["L"]     # Motor inductance, H
        self.k_T = data["k_T"]            # Torque constant, Nm/A
        self.k_e = data["k_e"]         # Back EMF constant, Nm/A
        self.R_w0 = data["R_w0"]            # Winding resistance at T_rw0, Ohm
        self.T_rw0 = data["T_Rw0"]              # Temperature for winding resistance value, degC

        self.c_thermal = data["C_w"]          # Winding thermal capacitance, J/K
        self.R_th = data["R_w"]               # Thermal resistance, K/W
    
class DCMotor():

    def __init__(self, params: MotorParam, T_amb, J_load, simulate_electrical_dynamics=False):
        # Sim parameters
        self.simulate_electrial_dynamics = simulate_electrical_dynamics

        # Motor parameters
        self.params = params

        # State
        self.omega = 0          # rad/s
        self.current = 0        # A
        self.temp = T_amb       # degC

        #
        self.input_u = 0        # 0..1

        self.J_tot = self.params.J_mot + J_load
        self.R_tot = self._get_winding_resistance()

    def get_state(self):
        return np.array([self.omega, self.current, self.temp])
    
    def set_state(self, state):
        self.omega = state[0]
        self.current = state[1]
        self.temp = state[2]

    def get_state_derivative(self, state, u, V_bat, tau_load, T_amb):

        # Unpack state
        omega = state[0]
        current = state[1]
        temp = state[2]

        if self.simulate_electrial_dynamics == True:
            V = V_bat * u
            current_dot = (V - self.R_tot * current - self.params.k_e * omega) / self.params.L
        else:
            current_dot = 0.0

        omega_dot = (self.params.k_T * current - self.params.b * omega - tau_load) / self.J_tot
        temp_dot = ((current ** 2) * self.R_tot - (temp - T_amb) / self.params.R_th) / self.params.c_thermal

        return np.array([omega_dot, current_dot, temp_dot])
    
    def update(self, u, V_bat, is_braking):
        self.input_u = u

        R_w = self._get_winding_resistance()

        if is_braking:
            self.R_tot = R_w + self.params.R_brake
            self.input_u = 0.0           # brake mode
        else:
            self.R_tot = R_w             # normal mode

        if not self.simulate_electrial_dynamics:
            self.current = (self.input_u * V_bat - self.params.k_e * self.omega) / self.R_tot # algebraic current (electrical dynamics + controller are fast)

    
    def set_state(self, state):
        #If electrical dynamics are simulated -> take current from integrator, otherwise keep algebraic current
        if self.simulate_electrial_dynamics:
            self.current = state[1]
        
        self.omega = state[0]
        self.temp = state[2]

    def log(self):
        log = {
            "omega": self.omega,
            "current": self.current,
            "T_winding": self.temp,
            "input_u": self.input_u,
            "R_tot": self.R_tot
        }
        return log
    
    def _get_winding_resistance(self):
        return self.params.R_w0 * (1 + 0.00393 * (self.temp - self.params.T_rw0))

