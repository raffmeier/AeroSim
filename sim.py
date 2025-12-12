import argparse
import time
from quad import *
from controller.attitude_pid import *
from logger import *
from plot import *
from sensor.sensor import SensorSuite
from mavlink_util import *
from drone_vis import DroneVis


# =========================
#   SIM MODE 1: OFFLINE Attitude PID
# =========================
def run_offline_attitude_sim():
    dt = 0.01
    t_sim = 20
    timesteps = int(t_sim/dt)

    QuadParams = QuadcopterParam()
    Quad = QuadcopterDynamics(QuadParams)
    Log = Logger(timesteps)

    Controller = AttitudePID(dt)

    start_time = time.time()

    print("Starting offline attitude PID simulation")
    for step in range(timesteps):
        # default: zero attitude
        angle_setpoint = np.array([0, 0, 0])

        # example: apply pitch command between certain timesteps
        if(step > 400 and step < 800):
            angle_setpoint = np.array([0.0, 0.35, 0])

        # controller output -> motor commands
        input = Controller.update(angle_setpoint, quat_to_euler(Quad.q), np.zeros(3), Quad.params.mass * 9.81)

        Quad.step(input, dt)
        Log.log(Quad, Controller, step, dt)
    
    print("Finished simulation in " + str(np.round(time.time() - start_time, 2)) + "s")
    plot(Log)
    Log.save_csv('output/log.csv')


# =========================
#   SIM MODE 2: PX4 SITL
# =========================
def run_px4_sitl_sim():
    dt_sim = 0.001 #1 ms simulation step
    imu_freq = 250 #Hz
    mag_freq = 10 #Hz
    baro_freq = 50 #Hz
    gnss_freq = 5 #Hz

    sensors = SensorSuite(dt_sim, imu_freq, mag_freq, baro_freq, gnss_freq)

    quadParams = QuadcopterParam()
    quad = QuadcopterDynamics(quadParams)

    viz = DroneVis(update_hz=60)

    px4 = connectToPX4SITL()

    ncmd = np.zeros(4)
    step = 0

    print("Starting PX4 SITL simulation")
    try:
        while True:
            now = int(time.time() * 1e6)

            quad.step(ncmd, dt_sim)
            sensors.step(step, quad)

            sendSensorsMessage(px4, sensors, step, now)

            controls = receiveActuatorControls(px4, quad)
            if controls is not None:
                ncmd = controls
            
            if step % 16 == 0:
                viz.update(quat_to_R(quad.q), quad.pos, quad.n)

            step += 1
            if step == 10**6:
                step = 0

            elapsed = time.time() - now / 1e6
            sleeptime = dt_sim - elapsed
            if sleeptime > 0:
                #print("Info: PX4 SITL simulation is running real-time! "+ str(elapsed) +"         " + str(step))
                time.sleep(sleeptime)
            else:
                print("Warning: PX4 SITL simulation is running slower than real-time! "+ str(elapsed) +"         " + str(step))
                

    except KeyboardInterrupt:
        print("Simulation stopped by user")
    
    finally:
        viz.close()


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