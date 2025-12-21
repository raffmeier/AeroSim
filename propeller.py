import numpy as np
import os
import json

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
        self.kT = data["kT"]
        self.kQ = data["kQ"]

class Propeller():

    def __init__(self, params: PropellerParam):
        self.params = params