import numpy as np
from utils import *
from matplotlib import pyplot as plt

class QuadcopterParam():

    def __init__(self):
        self.mass = 1
        self.inertia = np.diag(np.array([0.015, 0.015, 0.03]))
        self.arm = 0.17
        self.kT = 1.9 * 10**-6
        self.kQ = 2.6 ** 10**-7

class QuadcopterDynamcis():

    def __init__(self, params: QuadcopterParam):
        self.params = params
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.array([1, 0, 0, 0])
        self.w = np.zeros(3)
        self.n = np.zeros(4)
    
    def step(self, ncmd, dt):

        def omega_matrix(omega):
            p, q, r = omega
            return np.array([
                [0.0, -p, -q, -r],
                [p,  0.0,  r, -q],
                [q, -r,  0.0,  p],
                [r,  q, -p,  0.0],
            ])

        def state_dynamics(state, ncmd):

            state_dot = np.zeros(17)

            pos = state[0:3]
            vel = state[3:6]
            q = state[6:10]
            w = state[10:13]
            n = state[13:17]

            state_dot[0:3] = vel

            T = self.params.kT * np.square(n)
            T_total = np.array([0, 0, np.sum(T)])
            Fg = np.array([0, 0, 9.81 * self.params.mass])

            state_dot[3:6] = 1/self.params.mass * (-T_total - Fg)

            state_dot[6:10] = 0.5 * omega_matrix(w) @ q
            
            torque = np.array([self.params.arm * (-T[1] + T[3]), self.params.arm * (T[0] - T[2]), self.params.kQ * (n[0]**2 - n[1]**2 + n[2]**2 - n[3]**2)])
            
            state_dot[10:13] = np.linalg.inv(self.params.inertia) @ (torque - np.cross(w, self.params.inertia @ w))

            state_dot[13:17] = np.zeros(4)

            return state_dot
        
        def rk(S, n_cmd, dt):
            k1 = state_dynamics(S, n_cmd)
            k2 = state_dynamics(S + 0.5*dt*k1, n_cmd)
            k3 = state_dynamics(S + 0.5*dt*k2, n_cmd)
            k4 = state_dynamics(S + dt*k3,     n_cmd)
            return S + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        S = np.zeros(17)
        S[0:3]   = self.pos
        S[3:6]   = self.vel
        S[6:10]  = self.q
        S[10:13] = self.w
        S[13:17] = self.n
        S = rk(S, ncmd, dt) 

        S = rk(S, np.zeros(4), dt)

        q = S[6:10]
        S[6:10] = q / np.linalg.norm(q)

        self.pos = S[0:3]
        self.vel = S[3:6]
        self.q = S[6:10]
        self.w = S[10:13]
        self.n = S[13:17]

if __name__ == '__main__':

    QuadParams = QuadcopterParam()
    Quad = QuadcopterDynamcis(QuadParams)

    dt = 0.01
    t_sim = 10

    timesteps = int(t_sim/dt)

    log = {'pos':[], 'vel':[], 'q':[], 'omega':[], 'n':[], 'T':[]}


    for step in range(timesteps):
        Quad.step(np.zeros(4), dt)

        # log
        log['pos'].append(Quad.pos.copy())
        log['vel'].append(Quad.vel.copy())
        log['q'].append(Quad.q.copy())
        log['omega'].append(Quad.w.copy())
        log['n'].append(Quad.n.copy())
    
    #Plot

    timeaxis = np.arange(timesteps)
    p = np.vstack(log['pos'])

    plt.plot(timeaxis, p[:,2])
    plt.show()
        
            

            
            
        


