import multiprocessing as mp
import time
import numpy as np
import queue as queue_mod
from common.utils import *


def _vis_process(q, arm_length, update_hz):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D 

    plt.ion()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")

    L = arm_length
    rotor_centers_body = np.array([[L, 0, 0],
                                   [0, L, 0],
                                   [-L, 0, 0],
                                   [0, -L, 0]]).T

    # rotor locations
    rotor_scatter = ax.scatter([], [], [], s=30, color="k")

    # quad arms
    arm_x_line, = ax.plot([], [], [], linewidth=3, color='black')
    arm_y_line, = ax.plot([], [], [], linewidth=3, color='black')

    # trail
    trail_duration = 3 # seconds
    trail = []
    trail_scatter = ax.scatter([], [], [], s=10, color="gray", alpha=0.7)

    # body frame axes
    body_frame_quivers = []

    # rotor thrust arrows
    rotor_thrust_quiver = None

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.invert_yaxis()

    update_period = 1.0 / update_hz
    last_draw = time.time()

    R = np.eye(3)
    pos = np.zeros(3)
    w = np.zeros(4)

    max_arrow_length = 5 * L     # max length for thrust arrows
    body_axis_length = 0.5 * L     # length of the FRD body frame axes arrows

    thrust_dir_visual_world = np.array([0.0, 0.0, 1.0])

    F = np.diag([1, 1, -1])  # NED/FRD -> z-up vis frame

    while plt.fignum_exists(fig.number):
        try:
            while True:
                R, pos, w = q.get_nowait()
        except queue_mod.Empty:
            pass

        # convert from NED/FRD to visualization frame
        pos_vis = F @ pos
        R_vis = F @ R @ F

        # transform rotor centers to world (visual) frame
        rotor_world = R_vis @ rotor_centers_body + pos_vis.reshape(3, 1)

        # update rotor positions
        rotor_scatter._offsets3d = (
            rotor_world[0, :],
            rotor_world[1, :],
            rotor_world[2, :],
        )

        # update arms
        arm_x = rotor_world[:, [0, 2]]   # +x to -x
        arm_y = rotor_world[:, [1, 3]]   # +y to -y

        arm_x_line.set_data(arm_x[0, :], arm_x[1, :])
        arm_x_line.set_3d_properties(arm_x[2, :])

        arm_y_line.set_data(arm_y[0, :], arm_y[1, :])
        arm_y_line.set_3d_properties(arm_y[2, :])

        # update trail
        now = time.time()
        trail.append((now, pos_vis.copy()))

        cutoff = now - trail_duration
        trail = [(t, p) for (t, p) in trail if t >= cutoff]

        if len(trail) > 0:
            pts = np.stack([p for (t, p) in trail], axis=1)
            trail_scatter._offsets3d = (pts[0, :], pts[1, :], pts[2, :])
        else:
            trail_scatter._offsets3d = ([], [], [])

        # update body frame axes
        for qv in body_frame_quivers:
            qv.remove()
        body_frame_quivers = []

        origin = pos_vis
        axes = [
            (R_vis[:, 0], "r"),
            (R_vis[:, 1], "g"),
            (-R_vis[:, 2], "b")
        ]

        for axis_vec, color in axes:
            u, v, w_axis = body_axis_length * axis_vec
            qv = ax.quiver(
                origin[0], origin[1], origin[2],
                u, v, w_axis,
                length=1.0, normalize=False, color=color
            )
            body_frame_quivers.append(qv)

        # update rotor thrust arrows 
        if rotor_thrust_quiver is not None:
            rotor_thrust_quiver.remove()

        omega_norm = w / 1100
        arrow_lengths = max_arrow_length * omega_norm  # (4,)
        thrust_vecs = thrust_dir_visual_world.reshape(3, 1) * arrow_lengths.reshape(1, 4)

        xs_r = rotor_world[0, :]
        ys_r = rotor_world[1, :]
        zs_r = rotor_world[2, :]

        us_r = thrust_vecs[0, :]
        vs_r = thrust_vecs[1, :]
        ws_r = thrust_vecs[2, :]

        rotor_thrust_quiver = ax.quiver(
            xs_r, ys_r, zs_r,
            us_r, vs_r, ws_r,
            length=1.0, normalize=False, color = 'dodgerblue'
        )

        now = time.time()
        if now - last_draw >= update_period:
            fig.canvas.draw_idle()
            plt.pause(0.001)
            last_draw = now

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

    def update(self, state):
        try:
            q = state[6:10]
            R = quat_to_R(q)
            pos = state[0:3]
            w = state[13::4]
            
            self.queue.put_nowait((R, pos, w))
        except queue_mod.Full:
            # drop frame if visualizer is behind
            pass

    def close(self):
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=1.0)
