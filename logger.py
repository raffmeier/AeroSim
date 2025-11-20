import numpy as np
from quad import *
from controller.pid import *
from sensor.imu import IMU
from utils import *
import pandas as pd

class Logger():
    def __init__(self, timesteps):
        #State
        self.pos = np.zeros((3, timesteps))
        self.vel = np.zeros((3, timesteps))
        self.q = np.zeros((4, timesteps))
        self.w = np.zeros((3, timesteps))
        self.n = np.zeros((4, timesteps))

        #Extended metrics
        self.time = np.zeros(timesteps)
        self.eulerangles = np.zeros((3, timesteps))
        self.motor_thrust = np.zeros((4, timesteps))
        self.total_thrust = np.zeros(timesteps)
        self.accel = np.zeros((3, timesteps))

        #Controller
        self.angle_setpoint = np.zeros((3, timesteps))
        self.angle_error = np.zeros((3, timesteps))
        self.integrated_error = np.zeros((3, timesteps))
        self.motor_commands = np.zeros((4, timesteps))

        #IMU
        self.imu_accel = np.zeros((3, timesteps))
        self.imu_rates = np.zeros((3, timesteps))

    def log(self, quad: QuadcopterDynamcis, ctrl: AttitudePID, imu: IMU, step, dt):
        self.pos[:, step] = quad.pos
        self.vel[:, step] = quad.vel
        self.q[:, step] = quad.q
        self.w[:, step] = quad.w * 180 / np.pi
        self.n[:, step] = quad.n

        self.time[step] = dt * step
        self.eulerangles[:, step] = quat_to_euler(quad.q) * 180 / np.pi
        self.motor_thrust[:, step] = quad.params.kT * np.square(quad.n)
        self.total_thrust[step] = np.sum(self.motor_thrust)
        self.accel[:, step] = quad.accel

        self.angle_setpoint[:, step] = np.array([ctrl.roll_pid.setpoint, ctrl.pitch_pid.setpoint, ctrl.yaw_pid.setpoint])
        self.angle_error[:, step] = np.array([ctrl.roll_pid.err, ctrl.pitch_pid.err, ctrl.yaw_pid.err])
        self.integrated_error[:, step] = np.array([ctrl.roll_pid.integrated_err, ctrl.pitch_pid.integrated_err, ctrl.yaw_pid.integrated_err])
        self.motor_commands[:, step] = np.array([ctrl.motor_commands])

        self.imu_accel[:, step] = imu.meas_accel
        self.imu_rates[:, step] = imu.meas_rates * 180 / np.pi

    def save_csv(self, filename):
        df = pd.DataFrame({
            "Time": self.time,
            "x": self.pos[0, :],
            "y": self.pos[1, :],
            "z": self.pos[2, :],
            "vx": self.vel[0, :],
            "vy": self.vel[1, :],
            "vz": self.vel[2, :],
            "roll": self.eulerangles[0, :],
            "pitch": self.eulerangles[1, :],
            "yaw": self.eulerangles[2, :],
            "p": self.w[0, :],
            "q": self.w[1, :],
            "r": self.w[2, :],
            "n1": self.n[0, :],
            "n1_cmd": self.motor_commands[0, :],
            "n2": self.n[1, :],
            "n2_cmd": self.motor_commands[1, :],
            "n3": self.n[2, :],
            "n3_cmd": self.motor_commands[2, :],
            "n4": self.n[3, :],
            "n4_cmd": self.motor_commands[3, :],
            "roll_setpoint": self.angle_setpoint[0, :],
            "pitch_setpoint": self.angle_setpoint[1, :],
            "yaw_setpoint": self.angle_setpoint[2, :],
            "roll_error": self.angle_error[0, :],
            "pitch_error": self.angle_error[1, :],
            "yaw_error": self.angle_error[2, :],
            "roll_integrated_error": self.integrated_error[0, :],
            "pitch_integrated_error": self.integrated_error[1, :],
            "yaw_integrated_error": self.integrated_error[2, :],
            "accel_x": self.accel[0, :],
            "accel_y": self.accel[1, :],
            "accel_z": self.accel[2, :],
        })

        df.to_csv(filename, index=False)