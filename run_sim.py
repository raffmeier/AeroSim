import argparse
import time
from logger import *
from plot import *
from sensor.sensor import SensorSuite
from mavlink_util import *
from drone_vis import DroneVis
from simulator import Simulator
from controller.motor_velocity_pi import MotorVelocityController, MotorVelocityControllerParam
from integrator import RK4, Euler
from vehicle.multicopter import Multicopter, MulticopterParam
from vehicle.fixedwing import FixedWing, FixedWingParam
from datetime import datetime
import socket
from unity import sendPose

dt_sim = 0.001                                                      # s
t_sim = 10                                                          # s, For offline sim

initial_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])   # 3 position, 3 velocity, 4 attitude quaternion, 3 body rates

veh_type = 'fw'                                                     # Options: fw, mc
veh_name = 'adv_plane'                                              # Vehicle param file name
motor_ctrl_name = 'ctrl_maxon_ecx32_flat_uav'                       # Motor control param file

visulisation = 'none'                                               # Options: none, matplotlib (quad only), unity (send pose to UDP socket)

controller = 'none'                                                 # Options: px4, none

real_time = False                                                   # Running real time (definitely necessary for PX4 SITL), or as fast as possible

do_plot = True                                                      # Display log plots after finished sim

def run_sim():
    
    if visulisation == 'matplotlib':
        vis = DroneVis(update_hz=60)
    elif visulisation == 'unity':
        unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if veh_type == 'mc':
        veh_param = MulticopterParam(veh_name)
        veh = Multicopter(veh_param, initial_state)
        u_size = 4                                  # very bad, should generalize
        motor_prefixes = ["m0", "m1", "m2", "m3"]   # ugly, fix this

    elif veh_type == 'fw':
        veh_param = FixedWingParam(veh_name)
        veh = FixedWing(veh_param, initial_state)
        u_size = 5
        motor_prefixes = ["m0"]
    
    if controller == 'none':
        timesteps = int(t_sim/dt_sim)
    elif controller == 'px4':
        sensors = SensorSuite(dt_sim, imu_freq=250, mag_freq=10, baro_freq=50, gnss_freq=5)
        px4 = connectToPX4SITL()
    
    num_integrator = RK4()
    sim = Simulator(veh, num_integrator)

    ctrl_param = MotorVelocityControllerParam(motor_ctrl_name)
    motor = veh.get_motor()
    motor_count = len(motor)
    motor_ctrl = [MotorVelocityController(ctrl_param, motor[i], dt_sim, command_type='actuator_effort', simulate_electrical_dynamics=False) for i in range(motor_count)]

    logger = Logger()

    control_inputs = np.zeros(u_size)

    sim_running = True
    sim_step = 0

    start_time = time.time()
    print("Starting simulation...")

    try:
        while sim_running:
            now = time.time()
            now_us = int(now * 1e6)
            
            sim.step(control_inputs, dt_sim, T_amb=20)

            if controller == 'px4':
                sensors.step(sim_step, sim.veh)
                sendSensorsMessage(px4, sensors, sim_step, now_us)

                px4_controls = receiveActuatorControls(px4)
                if px4_controls is not None:
                    actuator_effort = px4_controls
            
            elif controller == 'none':
                actuator_effort = np.zeros(u_size)
            
            for i, mctrl in enumerate(motor_ctrl):
                control_inputs[i] = mctrl.update(actuator_effort[i], V_bat=veh.battery.get_pack_voltage())
            
            control_inputs[motor_count:u_size] = actuator_effort[motor_count:u_size]

            # Update visualisation
            if sim_step % 16 == 0:
                if visulisation == 'matplotlib':
                    vis.update(sim.veh.get_state())
                elif visulisation == 'unity':
                    sendPose(veh, unity_sock)
            
            # Logging
            if sim_step % 20 == 0:
                veh.log(logger, sim_step * dt_sim)
                for i, mctrl in enumerate(motor_ctrl):
                    mctrl.log(logger, f"m{i}.")
            
            # Advance sim step
            sim_step += 1

            # Stop offline sim it is at the final timestep
            if controller == 'none' and sim_step==timesteps:
                sim_running = False

            # Reset sim step to avoid overflow
            if sim_step == 10**6:
                sim_step = 0

            if real_time == True:
                # Sleep for remaining time
                elapsed = time.time() - now
                sleeptime = dt_sim - elapsed
                if sleeptime > 0:
                    time.sleep(sleeptime)
                else:
                    print("Warning: Real-time simulation is running slower than real-time! "+ str(elapsed) +"         " + str(sim_step))

    except KeyboardInterrupt:
        print("Simulation stopped by user")

    finally:
        print("Finished simulation in " + str(np.round(time.time() - start_time, 2)) + "s")

        if visulisation == 'matplotlib':
            vis.close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.save_to_csv(f"output/log_{ts}.csv")

        print(f"Saved logfile to output/log_{ts}.csv")
        
        if do_plot == True:
            plot_all(f"output/log_{ts}.csv", veh_type, motor_prefixes)
            plt.show()


if __name__ == "__main__":
    run_sim()