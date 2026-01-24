import numpy as np
from sensor.imu import IMU, IMUParam
from sensor.mag import Magnetometer, MagParam
from sensor.baro import Barometer, BaroParam
from sensor.gnss import GNSS, GNSSParam
from sensor.airspeed import PitotDiffPressureSensor, PitotDiffPressureParam
from common.utils import quat_to_R, getPressure
from vehicle.vehicle import Vehicle
import common.constants as constants

class SensorSuite:
    def __init__(self, sim_dt, imu_freq, mag_freq, baro_freq, gnss_freq):
        sim_freq = 1 / sim_dt

        dt_imu = 1 / imu_freq
        dt_mag = 1 / mag_freq
        dt_baro = 1 / baro_freq
        dt_gnss = 1 / gnss_freq

        self.imu_div = int(round(sim_freq / imu_freq))
        self.mag_div = int(round(sim_freq / mag_freq))
        self.baro_div = int(round(sim_freq / baro_freq))
        self.gnss_div = int(round(sim_freq / gnss_freq))

        imuParams = IMUParam()
        self.imu = IMU(imuParams, dt_imu)

        magParams = MagParam()
        self.mag = Magnetometer(magParams, dt_mag)

        baroParams = BaroParam()
        self.baro = Barometer(baroParams, dt_baro)

        gnssParams = GNSSParam()
        self.gnss = GNSS(gnssParams, dt_gnss)

        airspeedParams = PitotDiffPressureParam()
        self.airspeed_sensor = PitotDiffPressureSensor(airspeedParams)

    def step(self, step, veh: Vehicle):

        state = veh.get_state()
        accel = veh.get_accel()
        airspeed = veh.get_airspeed(state)

        R_wb = quat_to_R(state[6:10])

        if step % self.imu_div == 0:
            self.imu.step(accel, state[10:13], R_wb)
        
        if step % self.mag_div == 0:
            self.mag.step(R_wb)
        
        if step % self.baro_div == 0:
            pressure = getPressure(constants.HOME_ALT_M - state[2], constants.PRESSURE_ASL)
            self.baro.step(pressure)
            self.airspeed_sensor.step(airspeed)
        
        if step % self.gnss_div == 0:
            self.gnss.step(state[0:3], state[3:6])