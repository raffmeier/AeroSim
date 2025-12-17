#!/usr/bin/env bash

PX4_DIR="$HOME/Projects/PX4-Autopilot"
PHYSICS_SIM="$HOME/Projects/AeroSim/run_sim.py"
QGC="$HOME/QGC/QGroundControl-x86_64.AppImage"

# First tab: QGC
gnome-terminal \
  --tab \
  --title="QGC" \
  -- bash -ic "echo '[QGC] starting...'; $QGC; echo; echo '[QGC] exited'; exec bash" &

sleep 0.1

# Second tab: Physics sim
gnome-terminal \
  --tab \
  --title="Physics Sim" \
  -- bash -ic "echo '[Physics Sim] starting...'; python3 \"$PHYSICS_SIM\" --mode px4; echo; echo '[Physics Sim] exited'; exec bash" &

sleep 0.1

# Third tab: PX4 SITL
gnome-terminal \
  --tab \
  --title="PX4 SITL" \
  -- bash -ic "cd \"$PX4_DIR\" && echo '[PX4] starting SITL...'; make px4_sitl none_iris; echo; echo '[PX4] exited'; exec bash" &
