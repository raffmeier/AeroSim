class Integrator():
    def step():
          raise NotImplementedError()
class Euler(Integrator):
    def step(self, f, state, u, dt, V_bat, T_amb):
            return state + dt * f(state, u, V_bat, T_amb)

class RK4(Integrator):
    def step(self, f, state, u, dt, V_bat, T_amb):
            k1 = f(state, u, V_bat, T_amb)
            k2 = f(state + 0.5*dt*k1, u, V_bat, T_amb)
            k3 = f(state + 0.5*dt*k2, u, V_bat, T_amb)
            k4 = f(state + dt*k3, u, V_bat, T_amb)
            return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)