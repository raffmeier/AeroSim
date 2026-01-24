import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv(filename):
    with open(filename, "r", newline="") as f:
        r = csv.reader(f)
        keys = next(r)
        cols = {k: [] for k in keys}
        for row in r:
            for k, s in zip(keys, row):
                cols[k].append(float(s) if s != "" else np.nan)
    return cols


def _time_axis(d):
    t = np.asarray(d["t"], dtype=float)
    return t - t[0]


def plot_dashboard(d):
    t = _time_axis(d)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    axs[0, 0].plot(t, d["pos_0"], label="x", color='red')
    axs[0, 0].plot(t, d["pos_1"], label="y", color='green')
    axs[0, 0].plot(t, d["pos_2"], label="z", color='blue')
    axs[0, 0].set_title("Position")
    axs[0, 0].legend(); axs[0, 0].grid(True)

    axs[0, 1].plot(t, d["body_vel_0"], label="body vx", color='red')
    axs[0, 1].plot(t, d["body_vel_1"], label="body vy", color='green')
    axs[0, 1].plot(t, d["body_vel_2"], label="body vz", color='blue')
    axs[0, 1].set_title("Velocity")
    axs[0, 1].legend(); axs[0, 1].grid(True)

    axs[1, 0].plot(t, np.rad2deg(d["eulerangles_0"]), label="roll", color='darkorange')
    axs[1, 0].plot(t, np.rad2deg(d["eulerangles_1"]), label="pitch", color='blueviolet')
    axs[1, 0].plot(t, np.rad2deg(d["eulerangles_2"]), label="yaw", color='deepskyblue')
    axs[1, 0].set_title("Euler angles")
    axs[1, 0].legend(); axs[1, 0].grid(True)

    axs[1, 1].plot(t, np.rad2deg(d["rate_0"]), label="rollrate", color='darkorange')
    axs[1, 1].plot(t, np.rad2deg(d["rate_1"]), label="pitchrate", color='blueviolet')
    axs[1, 1].plot(t, np.rad2deg(d["rate_2"]), label="yawrate", color='deepskyblue')
    axs[1, 1].set_title("Body rates")
    axs[1, 1].legend(); axs[1, 1].grid(True)

    fig.tight_layout()
    plt.show(block=False)
    return fig


def plot_motor_dashboard(d, motor_prefix):
    def arr(name):
        return np.asarray(d[f"{motor_prefix}.{name}"], dtype=float)

    omega = arr("omega")
    omega_ref = arr("omegapid.setpoint")
    current = arr("current")
    current_ref = arr("current_ref")
    voltage = arr("voltage")
    resistance = arr("resistance")
    resistive_loss = arr("resistive_loss")
    mechanical_loss = arr("mechanical_loss")
    temp_winding = arr("temp_winding")
    temp_housing = arr("temp_housing")

    t = _time_axis(d)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    axs = axs.ravel()

    axs[0].set_title(f"{motor_prefix} Speed")
    axs[0].plot(t, omega, label="omega")
    axs[0].plot(t, omega_ref, "k--", label="omega_ref")
    axs[0].grid(); axs[0].legend()

    axs[1].set_title(f"{motor_prefix} Current")
    axs[1].plot(t, current, label="i")
    axs[1].plot(t, current_ref, "k--", label="i_ref")
    axs[1].grid(); axs[1].legend()

    axs[2].set_title(f"{motor_prefix} Voltage")
    axs[2].plot(t, voltage)
    axs[2].grid()

    axs[3].set_title(f"{motor_prefix} Power")
    axs[3].plot(t, voltage * current, label="Electrical")
    axs[3].plot(t, resistive_loss, label="Resistive loss")
    axs[3].plot(t, mechanical_loss, label="Mechanical loss")
    axs[3].plot(t, resistive_loss + mechanical_loss, label="Total loss")
    axs[3].grid(); axs[3].legend()

    axs[4].set_title(f"{motor_prefix} Temp")
    axs[4].plot(t, temp_winding, label="Winding temp", color='orangered')
    axs[4].plot(t, temp_housing, label="Housing temp", color='cornflowerblue')
    axs[4].axhline(y=155, linestyle='--', label='155degC', color='red')
    axs[4].grid(); axs[4].legend()

    axs[5].set_title(f"{motor_prefix} Resistance")
    axs[5].plot(t, resistance)
    axs[5].grid()

    fig.tight_layout()
    plt.show(block=False)
    return fig


def plot_battery_dashboard(d):
    t = _time_axis(d)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    axs[0, 0].plot(t, d["capacity"])
    axs[0, 0].set_title("Remaining capacity")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, d["V_pack"])
    axs[0, 1].set_title("Battery voltage")
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, d["I_discharge"])
    axs[1, 0].set_title("Battery discharge current")
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, d["SOC"])
    axs[1, 1].set_title("State of charge")
    axs[1, 1].grid(True)

    fig.tight_layout()
    plt.show(block=False)
    return fig

def plot_aero_dashboard(d):
    t = _time_axis(d)

    fig, axs = plt.subplots(5, 2, figsize=(12, 7), sharex=True)

    axs[0, 0].plot(t, np.rad2deg(d["alpha"]))
    axs[0, 0].set_title("Angle of attack")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, np.rad2deg(d["beta"]))
    axs[0, 1].set_title("Sideslip angle")
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, d["F_lift"])
    axs[1, 0].set_title("Lift force")
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, d["F_drag"])
    axs[1, 1].set_title("Drag force")
    axs[1, 1].grid(True)

    axs[2, 0].plot(t, d["F_side"])
    axs[2, 0].set_title("Side force")
    axs[2, 0].grid(True)

    axs[2, 1].plot(t, d["roll_mom"])
    axs[2, 1].set_title("Roll moment")
    axs[2, 1].grid(True)

    axs[3, 0].plot(t, d["pitch_mom"])
    axs[3, 0].set_title("Pitch mom")
    axs[3, 0].grid(True)

    axs[3, 1].plot(t, d["yaw_mom"])
    axs[3, 1].set_title("Yaw moment")
    axs[3, 1].grid(True)

    axs[4, 0].plot(t, d["prop_thrust"])
    axs[4, 0].set_title("prop_thrust")
    axs[4, 0].grid(True)

    axs[4, 1].plot(t, d["prop_torque"])
    axs[4, 1].set_title("Yaw prop_torque")
    axs[4, 1].grid(True)


    fig.tight_layout()
    plt.show(block=False)
    return fig


def plot_control_surface_dashboard(d):
    t = _time_axis(d)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    axs[0, 0].plot(t, np.rad2deg(d["cs0.angle"]), label='Angle')
    axs[0, 0].plot(t, np.rad2deg(d["cs0.setpoint"]), color='darkorange', linestyle="--", label='Setpoint')
    axs[0, 0].set_title("Left aileron")
    axs[0, 0].legend(); axs[0, 0].grid(True)

    axs[0, 1].plot(t, np.rad2deg(d["cs1.angle"]), label='Angle')
    axs[0, 1].plot(t, np.rad2deg(d["cs1.setpoint"]), color='darkorange', linestyle="--", label='Setpoint')
    axs[0, 1].set_title("Right aileron")
    axs[0, 1].legend(); axs[0, 1].grid(True)

    axs[1, 0].plot(t, np.rad2deg(d["cs2.angle"]), label='Angle')
    axs[1, 0].plot(t, np.rad2deg(d["cs2.setpoint"]), color='darkorange', linestyle="--", label='Setpoint')
    axs[1, 0].set_title("Elevator")
    axs[1, 0].legend(); axs[1, 0].grid(True)

    axs[1, 1].plot(t, np.rad2deg(d["cs3.angle"]), label='Angle')
    axs[1, 1].plot(t, np.rad2deg(d["cs3.setpoint"]), color='darkorange', linestyle="--", label='Setpoint')
    axs[1, 1].set_title("Rudder")
    axs[1, 1].legend(); axs[1, 1].grid(True)

    fig.tight_layout()
    plt.show(block=False)
    return fig



def plot_all(csv_file, vehicle_tyle, motor_prefixes):
    
    d = load_csv(csv_file)

    figs = []
    figs.append(plot_dashboard(d))

    for p in motor_prefixes:
        figs.append(plot_motor_dashboard(d, p))

    figs.append(plot_battery_dashboard(d))

    if vehicle_tyle=='fw':
        figs.append(plot_aero_dashboard(d))
        figs.append(plot_control_surface_dashboard(d))
    
    return figs
