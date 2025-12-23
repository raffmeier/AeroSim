import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from propeller import Propeller, PropellerParam

pp = PropellerParam('mejzlik_26.5x16.2_2B')
prop = Propeller(pp)

omegas = np.linspace(0, 350, 51)         # rad/s
v_infs = np.linspace(0, 25, 51)          # m/s

# Grid
O, V = np.meshgrid(omegas, v_infs, indexing="xy")

# Evaluate over full grid
T, Q, advr, prop_eff = prop.get_dynamic_force_torque(O, V)

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

v_slices = [0, 5.0, 10.0, 15.0, 20, 25]

ax4 = fig.add_subplot(2, 3, 4)
for v in v_slices:
    V_line = np.full_like(omegas, v)
    T_line, _, _, _ = prop.get_dynamic_force_torque(omegas, V_line)
    ax4.plot(omegas, T_line, label=f"V∞={v:g} m/s")
ax4.set_xlabel("omega [rad/s]")
ax4.set_ylabel("thrust [N]")
ax4.set_title("Thrust vs omega")
ax4.grid(True)
ax4.legend()

ax5 = fig.add_subplot(2, 3, 5)
for v in v_slices:
    V_line = np.full_like(omegas, v)
    _, Q_line, _, _ = prop.get_dynamic_force_torque(omegas, V_line)
    ax5.plot(omegas, Q_line, label=f"V∞={v:g} m/s")
ax5.set_xlabel("omega [rad/s]")
ax5.set_ylabel("torque [Nm]")
ax5.set_title("Torque vs omega")
ax5.grid(True)
ax5.legend()


plt.tight_layout()
plt.show()

