import numpy as np

class MotorParam():

    def __init__(self):
        self.R_brake = 1             # Brake resistor, Ohm

        self.J_mot = 3.07 * 10**-6   # Motor inertia, kg*m2
        self.b = 1.65 * 10**-5        # Motor damping constant, kg*m2/s

        self.L = 0.0949 * 10**-3     # Motor inductance, H
        self.k_T = 0.0113            # Torque constant, Nm/A
        self.k_e = self.k_T          # Back EMF constant, Nm/A
        self.R_w0 = 0.080            # Winding resistance at T_rw0, Ohm
        self.T_rw0 = 20              # Temperature for winding resistance value, degC

        self.c_thermal = 1.59          # Winding thermal capacitance, J/K
        self.R_th = 5               # Thermal resistance, K/W
    
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

