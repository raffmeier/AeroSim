# drone_vis.py
import multiprocessing as mp
import time
import numpy as np
import queue as queue_mod  # stdlib queue for Empty / Full


def _vis_process(q, arm_length, update_hz):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    L = arm_length
    rotor_centers_body = np.array(
        [
            [L, 0.0, 0.0],   # front
            [-L, 0.0, 0.0],  # back
            [0.0, L, 0.0],   # left
            [0.0, -L, 0.0],  # right
        ]
    ).T  # (3,4)

    arm_x_line, = ax.plot([], [], [], linewidth=3)
    arm_y_line, = ax.plot([], [], [], linewidth=3)
    prop_lines = [ax.plot([], [], [], linewidth=1)[0] for _ in range(4)]

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_box_aspect([1, 1, 0.5])

    update_period = 1.0 / update_hz
    last_draw = time.time()
    prop_angle = 0.0

    # defaults if queue is empty initially
    R = np.eye(3)
    pos = np.zeros(3)

    while plt.fignum_exists(fig.number):
        now = time.time()
        if now - last_draw < update_period:
            time.sleep(0.001)
            continue
        last_draw = now

        # get latest pose (non-blocking, drop intermediate ones)
        try:
            while True:
                R_new, pos_new = q.get_nowait()
                R, pos = R_new, pos_new * np.array([1, 1, -1])
        except queue_mod.Empty:
            pass

        # transform rotor centers to world
        rotor_world = R @ rotor_centers_body + pos.reshape(3, 1)
        arm_x = rotor_world[:, 0:2]
        arm_y = rotor_world[:, 2:4]

        arm_x_line.set_data(arm_x[0, :], arm_x[1, :])
        arm_x_line.set_3d_properties(arm_x[2, :])

        arm_y_line.set_data(arm_y[0, :], arm_y[1, :])
        arm_y_line.set_3d_properties(arm_y[2, :])

        # spinning props
        prop_angle += 80.0 * update_period
        angle_rad = np.deg2rad(prop_angle)
        r_prop = 0.12
        base = np.array([[-r_prop, 0.0, 0.0],
                         [ r_prop, 0.0, 0.0]]).T  # (3,2)
        Rz = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
                [np.sin(angle_rad),  np.cos(angle_rad), 0.0],
                [0.0,                0.0,               1.0],
            ]
        )

        for i in range(4):
            c_body = rotor_centers_body[:, i:i+1]
            prop_body = Rz @ base + c_body
            prop_world = R @ prop_body + pos.reshape(3, 1)
            line = prop_lines[i]
            line.set_data(prop_world[0, :], prop_world[1, :])
            line.set_3d_properties(prop_world[2, :])

        fig.canvas.draw_idle()
        plt.pause(0.001)

    try:
        plt.close(fig)
    except Exception:
        pass


class DroneVis:
    def __init__(self, update_hz: int = 60, arm_length: float = 0.25):
        self.queue = mp.Queue(maxsize=1)
        self.proc = mp.Process(
            target=_vis_process,
            args=(self.queue, arm_length, update_hz),
            daemon=True,
        )
        self.proc.start()

    def update(self, R, pos):
        """Non-blocking: drops old frames if the queue is full."""
        R = np.asarray(R, dtype=float)
        pos = np.asarray(pos, dtype=float)

        # try to clear old item (if any) so we always keep only newest
        try:
            _ = self.queue.get_nowait()
        except queue_mod.Empty:
            pass

        try:
            self.queue.put_nowait((R, pos))
        except queue_mod.Full:
            # if still full, just skip — sim must never block
            pass

    def close(self):
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=1.0)
