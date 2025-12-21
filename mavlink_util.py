import numpy as np
from pymavlink import mavutil
from sensor.sensor import SensorSuite
from common.utils import getAltitude
from quad import QuadcopterDynamics
import common.constants as constants

def connectToPX4SITL():
    master = mavutil.mavlink_connection('tcpin:0.0.0.0:4560')
    print("Waiting for heartbeat")
    master.wait_heartbeat()
    print(f"PX4 connected: system={master.target_system}, component={master.target_component}")

    return master

def sendSensorsMessage(master: mavutil.mavlink_connection, sensors: SensorSuite, step, time_us):
    #Sensor message
    updated_bitmask = 0b0001000000000000 # always update temp
    if step % sensors.imu_div == 0:
        updated_bitmask |= 0b0000000000111111 

    if step % sensors.mag_div == 0:
        updated_bitmask |= 0b0000000111000000 
    
    if step % sensors.baro_div == 0:
        updated_bitmask |= 0b0000101000000000
    
    master.mav.hil_sensor_send(
                time_usec =         time_us,
                xacc =              sensors.imu.meas_accel[0],
                yacc =              sensors.imu.meas_accel[1],
                zacc =              sensors.imu.meas_accel[2],
                xgyro =             sensors.imu.meas_rates[0],
                ygyro =             sensors.imu.meas_rates[1],
                zgyro =             sensors.imu.meas_rates[2],
                xmag =              sensors.mag.meas_mag[0],
                ymag =              sensors.mag.meas_mag[1],
                zmag =              sensors.mag.meas_mag[2],
                abs_pressure =      sensors.baro.meas_baro,
                diff_pressure =     0.0,
                pressure_alt =      getAltitude(sensors.baro.meas_baro, constants.PRESSURE_ASL),
                temperature =       20.0,
                fields_updated =    updated_bitmask,
                id =                0
            )
    #GPS message
    if step % sensors.gnss_div == 0:

        master.mav.hil_gps_send(
                time_usec =         time_us,
                fix_type =          3,
                lat =               int(sensors.gnss.meas_lat * 10**7),
                lon =               int(sensors.gnss.meas_lon * 10**7),
                alt =               int(sensors.gnss.meas_alt * 1000),
                eph =               int(sensors.gnss.param.pos_std_dev * 100),
                epv =               int(sensors.gnss.param.vel_std_dev * 100),
                vel =               int(np.sqrt(sensors.gnss.meas_vel[0]**2 + sensors.gnss.meas_vel[1]**2) * 100),
                vn =                int(sensors.gnss.meas_vel[0] * 100),
                ve =                int(sensors.gnss.meas_vel[1] * 100),
                vd =                int(sensors.gnss.meas_vel[2] * 100),
                cog =               int(np.rad2deg(sensors.gnss.course_over_ground) * 100),
                satellites_visible = 15,
                id =                0
            )
        
def receiveActuatorControls(master):
    msg = master.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=False)
    if msg is None:
        return None
    return np.array(msg.controls[0:4], dtype=float)