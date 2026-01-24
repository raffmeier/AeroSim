import numpy as np
from logger import Logger

class Battery():
    def __init__(self, dt):
        self.dt = dt
        self.cells_series = 6
        self.cells_parallel = 1
        self.full_capacity = 5.0 # Ah, Samsung 21700 50S

        self.capacity = self.full_capacity
        self.soc = self.capacity / self.full_capacity
        self.voltage = self.get_cell_voltage(self.full_capacity - self.capacity)
        self.discharge_current = 0.0
    
    def update(self, power):
        # if empty: stop discharge
        if self.capacity <= 0.0:
            self.capacity = 0.0
            self.discharge_current = 0.0
            self.voltage = 0.0
            return
        
        voltage_pack = self.cells_series * self.voltage

        self.discharge_current = power / voltage_pack

        self.capacity -= (self.discharge_current / self.cells_parallel) * (self.dt / 3600)
        self.soc = self.capacity / self.full_capacity

        self.voltage = self.get_cell_voltage(self.full_capacity - self.capacity)

    
    def log(self, L: Logger):
        L.log_scalar('capacity', self.capacity)
        L.log_scalar('V_pack', self.cells_series * self.voltage)
        L.log_scalar('I_discharge', self.discharge_current)
        L.log_scalar('SOC', self.soc)
    
    def get_pack_voltage(self):
        return self.cells_series * self.voltage
    
    def get_cell_voltage(self, cap):
        return -0.00616736*cap**5 + 0.07370887*cap**4 -0.32360096*cap**3 +  0.61528413*cap**2 - 0.63435624*cap + 4.05140775 # Modelled from Samsung 21700 50S 10A discharge curve