import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import common.constants as constants

class GNSSParam:
    def __init__(self):
        self.pos_std_dev = 0.5  # meters
        self.vel_std_dev = 0.05  # m/s

class GNSS:
    def __init__(self, params: GNSSParam, dt_gnss):
        self.param = params
        self.dt = dt_gnss

        self.meas_lat = 0  # Latitude in degrees
        self.meas_lon = 0  # Longitude in degrees
        self.meas_alt = 0  # Altitude in meters

        self.meas_vel = np.zeros(3)  # NED velocity in m/s
        self.ground_speed = 0  # Ground speed in m/s
        self.course_over_ground = 0  # Course over ground in radians
    
    def step(self, true_pos, true_vel):
        noise_pos = np.random.normal(0, self.param.pos_std_dev, 3)
        noise_vel = np.random.normal(0, self.param.vel_std_dev, 3)

        north = true_pos[0] + noise_pos[0]
        east  = true_pos[1] + noise_pos[1]
        down  = true_pos[2] + noise_pos[2]

        self.meas_lat = constants.HOME_LAT_DEG + north / constants.M_PER_DEG_LAT
        self.meas_lon = constants.HOME_LON_DEG + east  / constants.M_PER_DEG_LON
        self.meas_alt = constants.HOME_ALT_M - down

        vel = true_vel + noise_vel
        self.meas_vel = vel
        self.ground_speed = np.sqrt(vel[0]**2 + vel[1]**2)

        if abs(vel[0]) > 1e-3 or abs(vel[1]) > 1e-3:
            cog_rad = np.arctan2(vel[1], vel[0])
            if cog_rad < 0:
                cog_rad += 2 * np.pi
            self.course_over_ground = cog_rad
        else:
            self.course_over_ground = 0.0