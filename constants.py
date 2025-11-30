import numpy as np

#Zurich position
HOME_LAT_DEG = 47.376888
HOME_LON_DEG = 8.541694
HOME_ALT_M = 500.0
M_PER_DEG_LAT = 111319.9
M_PER_DEG_LON = 75382.8

HOME_B_NED = np.array([0.2149, 0.0133, 0.43]) #Gauss

#Pressure at sea level
PRESSURE_ASL = 1013.25 #mbar

#Gravity
GRAVITY_NED = np.array([0, 0, 9.81]) #m/s2