import numpy as np
from utils import *
from quad import *
from logger import *
from plot import *
from controller.pid import *
import time

if __name__ == '__main__':

    dt = 0.01
    t_sim = 15
    timesteps = int(t_sim/dt)

    QuadParams = QuadcopterParam()
    Quad = QuadcopterDynamcis(QuadParams)
    Log = Logger(timesteps)

    Controller = AttitudePID(dt)

    start_time = time.time()

    for step in range(timesteps):

        angle_setpoint = np.array([0, 0, np.deg2rad(90) * np.sin(step/500)])

        if(step > 400 and step < 800):
            angle_setpoint = np.array([0.2, 0.35, np.deg2rad(90) * np.sin(step/500)])


        input = Controller.update(angle_setpoint, quat_to_euler(Quad.q), np.zeros(3), Quad.params.mass * 9.81)
    
        #input = np.full(4, 0) #TODO get from controller

        Quad.step(input, dt)

        Log.log(Quad, Controller, step, dt)
    
    print("Finished simulation in " + str(np.round(time.time() - start_time, 2)) + "s")
    
    plot(Log)
    Log.save_csv('output/log.csv')

            

            
            
        


