import numpy as np

def quat_to_R(q): #Quaternion to Rotation Matrix: Body to World
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

def omega_matrix_from_q(omega):
            p, q, r = omega
            return np.array([
                [0.0, -p, -q, -r],
                [p,  0.0,  r, -q],
                [q, -r,  0.0,  p],
                [r,  q, -p,  0.0],
            ])

def getPressure(altitude, ref_pressure):
    return ref_pressure * (1 - (2.25577e-5 * altitude))**5.25588

def getAltitude(pressure, ref_pressure):
    return (1 - (pressure / ref_pressure)**(1/5.25588)) / 2.25577e-5