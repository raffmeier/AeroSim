import numpy as np
from model.vehicle.vehicle import Vehicle
from integrator import Integrator

class Simulator:

    def __init__(self, veh: Vehicle, integrator: Integrator):
        self.veh = veh
        self.integrator = integrator
    
    def step(self, u, dt, T_amb):

        self.veh.pre_step(u, np.zeros(len(self.veh.get_motor())))
        state = self.veh.get_state()

        state_new = self.integrator.step(self.veh.get_state_derivative, state, u, dt, T_amb)

        self.veh.set_state(state_new)
        self.veh.post_step()