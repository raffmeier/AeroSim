import numpy as np
import os
import json
from logger import Logger

class PropellerParam():

    def __init__(self, propeller_name: str):

        base_dir = os.path.dirname(__file__)
        propeller_file = os.path.join(
            base_dir,
            "parameter",
            "propeller",
            f"{propeller_name}.json"
        )

        if not os.path.isfile(propeller_file):
            raise FileNotFoundError(
                f"Propeller file not found: {propeller_file}"
            )

        with open(propeller_file, "r") as f:
            data = json.load(f)

    # --- load parameters ---

        self.J_prop = data["inertia"]
        self.rho = 1.2 #Fix this

        # Static model
        self.kT = data["kT"]
        self.kQ = data["kQ"]

        # Dynamic model
        self.d = data["diameter"]

        self.J0 = data["J0"]

        self.cT2 = data["cT_coeff_2"]
        self.cT1 = data["cT_coeff_1"]
        self.cT0 = data["cT_coeff_0"]

        self.cQ2 = data["cQ_coeff_2"]
        self.cQ1 = data["cQ_coeff_1"]
        self.cQ0 = data["cQ_coeff_0"]

class Propeller():

    def __init__(self, params: PropellerParam, model='dynamic'):
        self.params = params
        self.model = model

        self.thrust = 0.0
        self.torque = 0.0

        self.cT = 0.0
        self.cQ = 0.0
        self.advance_ratio = 0.0
        self.prop_eff = 0.0


    def get_force_torque(self, omega, axial_freestream_velocity=0):
        
        if self.model == 'static':
            return self.get_static_force_torque(omega)
        elif self.model == 'dynamic':
            return self.get_dynamic_force_torque(omega, axial_freestream_velocity)
        else:
            raise NotImplementedError("Invalid propeller model.")
        
    
    def get_static_force_torque(self, omega):
        omega_squared = omega**2

        prop_thrust = self.params.kT * omega_squared
        prop_torque = self.params.kQ * omega_squared

        return prop_thrust, prop_torque
    
    
    def get_dynamic_force_torque(self, omega, v_inf):

        # If prop is not rotating, return 0 thrust and 0 torque
        if omega == 0:
            self.thrust = 0.0
            self.torque = 0.0
            self.advance_ratio = 0.0
            self.prop_eff = 0.0
            return 0.0, 0.0

        omega_revps = omega / (2.0 * np.pi)
        advance_ratio = v_inf / (omega_revps * self.params.d)

        J0 = self.params.J0

        if advance_ratio <= J0:
            self.cT = self.params.cT2 * advance_ratio**2 + self.params.cT1 * advance_ratio + self.params.cT0
            self.cQ = self.params.cQ2 * advance_ratio**2 + self.params.cQ1 * advance_ratio + self.params.cQ0
        else:
            # Value at J0 for fade region
            cT0 = self.params.cT2 * J0**2 + self.params.cT1 * J0 + self.params.cT0
            cQ0 = self.params.cQ2 * J0**2 + self.params.cQ1 * J0 + self.params.cQ0

            # Exponential fade region (J > J0)
            k = 5.0
            fade = np.exp(-k * (advance_ratio - J0))
            self.cT = cT0 * fade
            self.cQ = cQ0 * fade

        # Forces
        self.thrust = self.params.rho * omega_revps**2 * self.params.d**4 * self.cT
        self.torque = self.params.rho * omega_revps**2 * self.params.d**5 * self.cQ

        self.advance_ratio = advance_ratio

        return self.thrust, self.torque
    

    def log(self, L: Logger, prefix: str):
        L.log_scalar(f"{prefix}thrust", self.thrust)
        L.log_scalar(f"{prefix}torque", self.torque)

        if self.model == 'dynamic':
            L.log_scalar(f"{prefix}cT", self.cT)
            L.log_scalar(f"{prefix}cQ", self.cQ)
            L.log_scalar(f"{prefix}advance_ratio", self.advance_ratio)
