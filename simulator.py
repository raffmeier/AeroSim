import numpy as np
import time
from vehicle import Vehicle
from multicopter import Multicopter
from integrator import Integrator
from integrator import RK4, Euler

class Simulator:

    def __init__(self, veh: Vehicle, integrator: Integrator):
        self.veh = veh
        self.integrator = integrator
    
    def step(self, u, dt, V_bat, T_amb):

        self.veh.pre_step(u, V_bat, np.zeros(4))
        state = self.veh.get_state()

        state_new = self.integrator.step(self.veh.get_state_derivative, state, u, dt, V_bat, T_amb)

        self.veh.set_state(state_new)
        self.veh.post_step()



if __name__ == "__main__":
    veh = Multicopter()
    rk4 = RK4()
    sim = Simulator(veh, rk4)

    u = np.full(4, 1)
    dt = 0.001

    start = time.time()

    for k in range(1000):
        tim = time.time()
        sim.step(u, dt, V_bat=50, T_amb=20)
        print(np.round(time.time() - tim, 5))
        #print(sim.veh.get_state())    
    print("Simulation finished in " + str(time.time() - start))