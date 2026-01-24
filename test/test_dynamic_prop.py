import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from model.propeller import Propeller, PropellerParam

pp = PropellerParam('mejzlik_26.5x16.2_2B')
prop = Propeller(pp)

omegas = np.linspace(0, 350, 51)         # rad/s
v_infs = np.linspace(-5, 25, 51)          # m/s

O, V = np.meshgrid(omegas, v_infs)

T = np.zeros_like(O, dtype=float)
Q = np.zeros_like(O, dtype=float)
advr = np.zeros_like(O, dtype=float)

# --- Evaluate over full grid ---
for i in range(O.shape[0]):
    for j in range(O.shape[1]):
        omega = float(O[i, j])
        vinf  = float(V[i, j])

        t, q = prop.get_dynamic_force_torque(omega, vinf)
        T[i, j] = t
        Q[i, j] = q
        advr[i, j] = prop.advance_ratio

fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax1.plot_surface(O, V, T)
ax1.set_xlabel("omega [rad/s]")
ax1.set_ylabel("axial freestream velocity [m/s]")
ax1.set_zlabel("thrust [N]")
ax1.set_title("Thrust surface")

ax2 = fig.add_subplot(2, 3, 2, projection="3d")
ax2.plot_surface(O, V, Q)
ax2.set_xlabel("omega [rad/s]")
ax2.set_ylabel("axial freestream velocity [m/s]")
ax2.set_zlabel("torque [Nm]")
ax2.set_title("Torque surface")

ax3 = fig.add_subplot(2, 3, 3, projection="3d")
ax3.plot_surface(O, V, advr)
ax3.set_xlabel("omega [rad/s]")
ax3.set_ylabel("axial freestream velocity [m/s]")
ax3.set_zlabel("advance ratio")
ax3.set_title("Advance ratio surface")

v_slices = [-5.0, 0, 5.0, 10.0, 15.0, 20, 25]

ax4 = fig.add_subplot(2, 3, 4)

for v in v_slices:
    T_line = np.zeros_like(omegas, dtype=float)

    for k, om in enumerate(omegas):
        t, _ = prop.get_dynamic_force_torque(float(om), float(v))
        T_line[k] = t

    ax4.plot(omegas, T_line, label=f"V∞={v:g} m/s")
ax4.set_xlabel("omega [rad/s]")
ax4.set_ylabel("thrust [N]")
ax4.set_title("Thrust vs omega")
ax4.grid(True)
ax4.legend()

ax5 = fig.add_subplot(2, 3, 5)
for v in v_slices:
    Q_line = np.zeros_like(omegas, dtype=float)

    for k, om in enumerate(omegas):
        _, q = prop.get_dynamic_force_torque(float(om), float(v))
        Q_line[k] = q

    ax5.plot(omegas, Q_line, label=f"V∞={v:g} m/s")
ax5.set_xlabel("omega [rad/s]")
ax5.set_ylabel("torque [Nm]")
ax5.set_title("Torque vs omega")
ax5.grid(True)
ax5.legend()


plt.tight_layout()
plt.show()

