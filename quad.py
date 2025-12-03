import numpy as np
from utils import *
import constants

class QuadcopterParam():
    def __init__(self):
        # Quadcopter physical parameters (Iris quadcopter)
        self.mass = 1.5
        self.inertia = np.diag(np.array([0.0348, 0.0459, 0.0977]))
        self.arm = 0.23
        self.kT = 8.54858 * 10**-6
        self.kQ = 5.129148 * 10**-7
        self.CdA = np.array([0.02, 0.02, 0.03])

        self.tau_m = 0.01
        self.nmax = 1100 # max 1100 rad/s = 10'000 rpm
        self.dn_max = 520 # max 520 rad/s² = 5000 rpm/s²

        self.ground_linear_friction = 5
        self.ground_angular_friction = 5

        self.rho = 1.225
        

class QuadcopterDynamics():
    def __init__(self, params: QuadcopterParam):
        self.params = params
        self.inv_inertia = np.linalg.inv(self.params.inertia)
        
        #State
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.array([1, 0, 0, 0])
        self.w = np.zeros(3)
        self.n = np.full(4, 0)

        #Additional
        self.accel = np.zeros(3)
    
    def state_dynamics(self, state, ncmd):

            #Unpack state
            pos = state[0:3]
            vel = state[3:6]
            q = state[6:10]
            w = state[10:13]
            n = state[13:17]

            state_dot = np.zeros(17)

            state_dot[0:3] = vel # Kinematics

            # 2) Linear dynamics
            R = quat_to_R(q)
            T = self.params.kT * np.square(n)
            T_total = np.array([0, 0, -np.sum(T)]) # Total thrust in body frame

            vel_body = R.T @ vel # Linear velocity in body frame
            Fd = -0.5 * self.params.rho * self.params.CdA * vel_body * np.abs(vel_body) # Quadratic drag force in body frame

            state_dot[3:6] = 1/self.params.mass * (R @ (T_total + Fd) + constants.GRAVITY_NED) # NSL in world frame

            state_dot[6:10] = 0.5 * omega_matrix_from_q(w) @ q # Quaternion dynamics
        
            torque = np.array([self.params.arm * (-T[1] + T[3]), self.params.arm * (T[0] - T[2]), self.params.kQ * (-n[0]**2 + n[1]**2 - n[2]**2 + n[3]**2)]) # Rotational dynamics
            
            state_dot[10:13] = self.inv_inertia @ (torque - np.cross(w, self.params.inertia @ w)) 

            # 5) Motor dynamics
            dn = (ncmd - n) / self.params.tau_m
            state_dot[13:17] = np.clip(dn, -self.params.dn_max, self.params.dn_max) # Motor dynamics
            
            # 6) Ground contact
            if(pos[2] >= 0): # On ground
                # Apply friction terms to linear and angular velocities
                state_dot[3:5] -= self.params.ground_linear_friction * vel[0:2]
                state_dot[10:13] -= self.params.ground_angular_friction * w

            return state_dot
    
    def rk4(self, S, n_cmd, dt):
            k1 = self.state_dynamics(S, n_cmd)
            k2 = self.state_dynamics(S + 0.5*dt*k1, n_cmd)
            k3 = self.state_dynamics(S + 0.5*dt*k2, n_cmd)
            k4 = self.state_dynamics(S + dt*k3,     n_cmd)
            return S + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    def step(self, ncmd, dt):

        S = np.zeros(17)
        S[0:3]   = self.pos
        S[3:6]   = self.vel
        S[6:10]  = self.q
        S[10:13] = self.w
        S[13:17] = self.n

        S = self.rk4(S, ncmd, dt)

        q = S[6:10]
        S[6:10] = q / np.linalg.norm(q)

        self.pos = S[0:3]
        self.vel = S[3:6]
        self.q = S[6:10]
        self.w = S[10:13]
        self.n = np.clip(S[13:17], 0, self.params.nmax)

        state_dot_final = self.state_dynamics(S, ncmd)
        self.accel = state_dot_final[3:6]

        # On ground --> clip vertical position, velocity, acceleration
        if(self.pos[2] >= 0):
            self.pos[2] = 0
            self.vel[2] = 0
            self.accel[2] = 0