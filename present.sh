#!/bin/bash
echo "Launching ARM-S Full Presentation Mode..."

# Activate environment
source /home/ferrarikazu/ARM-S/srl_env/bin/activate

# Terminal 1 — MuJoCo Live Viewer (the actual robot simulation)
gnome-terminal --title="ARM-S Live Simulation" -- bash -c \
  "source /home/ferrarikazu/ARM-S/srl_env/bin/activate && cd /home/ferrarikazu/ARM-S && python launch_viewer.py --policy reach; exec bash" &

sleep 2

# Terminal 2 — Main System Dashboard
gnome-terminal --title="ARM-S Main Dashboard" -- bash -c \
  "source /home/ferrarikazu/ARM-S/srl_env/bin/activate && cd /home/ferrarikazu/ARM-S && python dashboard.py; exec bash" &

sleep 2

# Terminal 3 — Controllers Dashboard  
gnome-terminal --title="ARM-S Controllers" -- bash -c \
  "source /home/ferrarikazu/ARM-S/srl_env/bin/activate && cd /home/ferrarikazu/ARM-S && python dashboard_controllers.py; exec bash" &

echo "All 3 components launched. Press Ctrl+C to stop all."
wait
