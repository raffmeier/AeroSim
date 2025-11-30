import numpy as np
from matplotlib import pyplot as plt
from logger import *

def plot(Log: Logger):
        #Plot
    timeaxis = Log.time

    fig, axes = plt.subplots(3, 5, figsize=(24,14))

    position_labels = ["x [m]", "y [m]", "z [m]"]
    for i in range(3):
        axes[i, 0].plot(timeaxis, Log.pos[i, :])
        axes[i, 0].set_ylabel(position_labels[i])
        axes[i, 0].grid()
    
    velocity_labels = ["vx [m/s]", "vy [m/s]", "vz [m/s]"]
    for i in range(3):
        axes[i, 1].plot(timeaxis, Log.vel[i, :])
        axes[i, 1].set_ylabel(velocity_labels[i])
        axes[i, 1].grid()

    euler_labels = ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"]
    for i in range(3):
        axes[i, 2].plot(timeaxis, Log.eulerangles[i, :])
        axes[i, 2].plot(timeaxis, Log.angle_setpoint[i, :] * 180 / np.pi, color="orangered")
        axes[i, 2].set_ylabel(euler_labels[i])
        axes[i, 2].grid()
    
    rate_labels = ["Rollrate [deg/s]", "Pitchrate [deg/s]", "Yawrate [deg/s]"]
    for i in range(3):
        axes[i, 3].plot(timeaxis, Log.w[i, :])
        axes[i, 3].set_ylabel(rate_labels[i])
        axes[i, 3].grid()
    
    thrust_labels = ["Motor 1", "Motor 2", "Motor 3", "Motor 4"]
    for i in range(4):
        axes[0, 4].plot(timeaxis, Log.n[i, :], label = thrust_labels[i])
        #axes[0, 4].plot(timeaxis, Log.motor_commands[i, :])
        axes[0, 4].legend()
    axes[0, 4].grid()

    axes[1,4].scatter(Log.pos[0, :], Log.pos[1, :], s=2)
    axes[1,4].set_xlabel("x [m]")
    axes[1,4].set_ylabel("y [m]")

    axes[2,4].remove()


    plt.show()