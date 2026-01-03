# AeroSim

readme is not up to date

Python-based quadrotor sandbox that can drive PX4 SITL via MAVLink. The repository models the vehicle and sensor dynamics. There is also a simple PID attitude controller for debugging.

- Offline attitude PID demo with logging/plots plus CSV export for analysis.
- Real-time sensor simulation (IMU, magnetometer, barometer, GNSS) with noise, bias random walk, quantization, and rate limits.
- Physics model with 17 states (position, velocity, quaternion, angular rates, four rotor speeds) advanced with 4th-order Runge-Kutta integration.
- MAVLink bridge that drives `HIL_SENSOR` and `HIL_GPS` messages and consumes `HIL_ACTUATOR_CONTROLS` so PX4 SITL can command the virtual quadrotor.

## Quick Start
1. Create a Python 3.9+ environment and install the dependencies:  
   `pip install numpy pandas matplotlib pymavlink`
2. Run the offline attitude demo (generates plots and `output/log.csv`):  
   `python sim.py --mode offline`
3. (Optional) Drive PX4 SITL (requires PX4 sources and QGroundControl paths to be correct in `run_px4sitl.sh`):  
   - Update `PX4_DIR`, `PHYSICS_SIM`, and `QGC` in the script, or run the components manually.  
   - Start PX4 SITL (`make px4_sitl none_iris`).  
   - Launch the physics bridge: `python sim.py --mode px4`.  
   - PX4 will stream actuator commands via `HIL_ACTUATOR_CONTROLS`; the sim responds with matching sensor feeds.


## Frame conventions
- **World frame:** NED (x north, y east, z down). Ground plane is clamped at z = 0.
- **Body frame:** FRD (x forward, y right, z down)

## Physics Model
`QuadcopterDynamics.step()` integrates a 17-dimensional state vector (Position xyz, velocity vx,vy,vz, attitude quaternion qw,qx,qy,qz, body rates p,q,r and rotor speeds n0,n1,n2,n3) with RK4:

- **Thrust & torque:** Individual thrusts are `T = kT * n²`. Moments are resolved using the arm length and the drag torque coefficient `kQ`.
- **Translational forces:** Sum of motor thrust (in body-Z negative direction) and quadratic body drag (`CdA` per axis) transformed into the world frame. Gravity is applied in world coordinates.
- **Rotational dynamics:** Eulers rotation equations: Propeller torques from thrust differentials plus drag torque from spin direction.
- **Motor dynamics:** First-order response with time constant `tau_m` toward commanded rotor speeds with magnitude and slew limits; outputs are saturated to `[0, nmax]`.

## Sensor Models
The `SensorSuite` advances each instrument at its own rate (IMU 250 Hz, mag 10 Hz, baro 50 Hz, GNSS 5 Hz by default):

- **IMU:** Adds white noise and bias random walk to both accelerometer and gyro channels, clips to hardware ranges, and quantizes based on bit depth.
- **Magnetometer:** Uses a fixed NED field vector from `constants.py`, adds Gaussian noise, clips, and quantizes based on sensitivity.
- **Barometer:** Converts down position to pressure via an ISA approximation, adds Gaussian noise, saturates to sensor limits, and quantizes.
- **GNSS:** Perturbs position/velocity with configurable standard deviations before converting to geodetic latitude/longitude/altitude using the Zurich reference and meters-per-degree constants. Ground speed and course-over-ground are computed for MAVLink.

## PID attitude controller
- `AttitudePID` uses decoupled PIDs for roll, pitch, and yaw.
- Offline mode drives the controller with perfect attitude feedback (`quat_to_euler`).

## Logging and Visualization (offline simulation only)
- `Logger` records states, commanded angles, motor speeds, thrust, and accelerations at every simulation step. Call `Logger.save_csv()` to export to `output/log.csv`.
- `plot.py` produces multi-panel Matplotlib plots showing translational states, Euler angles versus setpoints, body rates, rotor speeds, and XY position traces.

## Repository Layout
```
.
├── README.md
├── controller
│   └── pid.py                      # Simple PID attitude controller
├── sensor
│   ├── baro.py                     # Barometer model
│   ├── gnss.py                     # GNSS model
│   ├── imu.py                      # IMU model
│   ├── mag.py                      # Magnetometer model
│   └── sensor.py                   # Handles different sensor rates
├── constants.py                    # Physical constants (some specific to Zurich)
├── logger.py                       # Logger class to log values for the offline sim
├── mavlink_util.py                 # PX4 SITL connection helpers and MAVLink I/O
├── plot.py                         # Matplotlib plots for the offline sim
├── quad.py                         # Quadcopter rigid body dynamics and parameters
├── sim.py                          # entry point, provdies offline and px4 modes
├── utils.py                        # utility functions
└── run_px4sitl.sh                  # convenience launcher that opens QGC, sim and PX4 SITL
```
