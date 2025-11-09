import numpy as np

def quat_multiply(q1, q2):
    # [w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_to_R(q):
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ])

    return R

def quat_to_euler(q):
    qw, qx, qy, qz = q

    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = -np.pi / 2 + 2 * np.arctan2(np.sqrt(1 + 2 * (qw * qy - qx * qz)), np.sqrt(1 - 2*(qw * qy - qx * qz)))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    return np.array([roll, pitch, yaw])