import argparse
import time
from controller.attitude_pid import *
from logger import *
from plot import *
from sensor.sensor import SensorSuite
from mavlink_util import *
from drone_vis import DroneVis
from simulator import Simulator
from controller.motor_velocity_pi import MotorVelocityController, MotorVelocityControllerParam
from actuator.dcmotor import DCMotor, MotorParam
from integrator import Integrator, RK4, Euler
from vehicle.multicopter import Multicopter, MulticopterParam
from datetime import datetime


# =========================
#   SIM MODE 1: OFFLINE Attitude PID
# =========================
def run_offline_attitude_sim():
    dt_sim = 0.002
    t_sim = 20
    timesteps = int(t_sim/dt_sim)

    nmax = 1225

    quad = Multicopter()
    rk4 = RK4()

    sim = Simulator(quad, rk4)

    ctrl_param = MotorVelocityControllerParam('ctrl_maxon_ecx32_flat_uav')
    motor_ctrl = [MotorVelocityController(ctrl_param, quad.motors[i], dt_sim, command_type='actuator_effort', simulate_electrical_dynamics=False) for i in range(4)]

    viz = DroneVis(update_hz=60)

    u = np.zeros(4)
    omega_ref = np.zeros(4)

    Controller = AttitudePID(dt_sim)

    start_time = time.time()

    print("Starting offline attitude PID simulation")
    for step in range(timesteps):
        now = int(time.time() * 1e6)

        # default: zero attitude
        angle_setpoint = np.array([0, 0, 0])

        # example: apply pitch command between certain timesteps
        if(step > 400 and step < 800):
            angle_setpoint = np.array([0.0, 0.35, 0])

        # Dynamics
        sim.step(u, dt_sim, V_bat=50, T_amb=20)

        # controller output -> motor commands
        omega_ref = Controller.update(angle_setpoint, quat_to_euler(quad.get_state()[6:10]), np.zeros(3), quad.rb.mass * 15.81)

        #print(omega_ref)

        for i, mctrl in enumerate(motor_ctrl):
            u[i] = mctrl.update(omega_ref[i], V_bat=50)
        
        print(quad.get_state())
        
        if step % 16 == 0:
                viz.update(sim.veh.get_state())
        
        elapsed = time.time() - now / 1e6
        sleeptime = dt_sim - elapsed
        if sleeptime > 0:
            #print("Info: PX4 SITL simulation is running real-time! "+ str(elapsed) +"         " + str(step))
            time.sleep(sleeptime)
        else:
            #print("Warning: Offline simulation is running slower than real-time! "+ str(elapsed) +"         " + str(step))
            pass


    
    print("Finished simulation in " + str(np.round(time.time() - start_time, 2)) + "s")


# =========================
#   SIM MODE 2: PX4 SITL
# =========================
def run_px4_sitl_sim():
    dt_sim = 0.001 #1 ms simulation step
    step = 0

    imu_freq = 250 #Hz
    mag_freq = 10 #Hz
    baro_freq = 50 #Hz
    gnss_freq = 5 #Hz

    logger = Logger()

    mcp = MulticopterParam('test_quad')
    quad = Multicopter(mcp)
    
    rk4 = RK4()
    sim = Simulator(quad, rk4)

    sensors = SensorSuite(dt_sim, imu_freq, mag_freq, baro_freq, gnss_freq)

    ctrl_param = MotorVelocityControllerParam('ctrl_maxon_ecx32_flat_uav')
    motor_ctrl = [MotorVelocityController(ctrl_param, quad.motors[i], dt_sim, command_type='actuator_effort', simulate_electrical_dynamics=False) for i in range(4)]

    viz = DroneVis(update_hz=60)

    px4 = connectToPX4SITL()

    u = np.zeros(4)
    actuator_effort = np.zeros(4)

    print("Starting PX4 SITL simulation")
    try:
        while True:
            # Time
            now = time.time()
            now_us = int(now * 1e6)

            # Step simulation
            sim.step(u, dt_sim, V_bat=50, T_amb=20)

            # Measure sensor and send to PX4
            sensors.step(step, sim.veh)
            sendSensorsMessage(px4, sensors, step, now_us)

            # Receive control commands from PX4 and send to PI motor controller
            controls = receiveActuatorControls(px4)
            if controls is not None:
                actuator_effort = controls

            for i, mctrl in enumerate(motor_ctrl):
                u[i] = mctrl.update(actuator_effort[i], V_bat=50)
            
            # Update visualisation
            if step % 16 == 0:
                viz.update(sim.veh.get_state())

            # Advance sim step
            step += 1
            if step == 10**6:
                step = 0
            
            if step % 20 == 0:
                # Log multicopter and motor controllers
                quad.log(logger, now)
                for i, mctrl in enumerate(motor_ctrl):
                    mctrl.log(logger, f"m{i}.")

            # Sleep for remaining time
            elapsed = time.time() - now
            sleeptime = dt_sim - elapsed
            if sleeptime > 0:
                time.sleep(sleeptime)
            else:
                print("Warning: PX4 SITL simulation is running slower than real-time! "+ str(elapsed) +"         " + str(step))
                

    except KeyboardInterrupt:
        print("Simulation stopped by user")
    
    finally:
        viz.close()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.save_to_csv(f"output/log_{ts}.csv")
        plot_all(f"output/log_{ts}.csv", motor_prefixes=["m0", "m1", "m2", "m3"])
        plt.show()


# =========================
#            MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Quadcopter simulation launcher")
    parser.add_argument(
        "--mode",
        choices=["offline", "px4"],
        default="offline",
        help="Choose which simulation to run: 'offline' (PID + plots) or 'px4' (PX4 SITL)."
    )

    args = parser.parse_args()

    if args.mode == "offline":
        run_offline_attitude_sim()
    elif args.mode == "px4":
        run_px4_sitl_sim()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")    

if __name__ == "__main__":
    main()