from plot import *

file = "log_20251221_201357"

plot_all("output/"+file+".csv", motor_prefixes=["m0", "m1", "m2", "m3"])
plt.show()