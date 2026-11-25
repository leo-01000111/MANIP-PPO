#!/bin/bash
# Final policy (PPO with q). Target: XYZ="[x, y, z]" (default below). Orientation not used (position task). The simulator must already be running.
# Usage: ~/ppo_demo_final.sh [record_seconds]   -> with an argument, screenshots are recorded into /tmp/frames
pkill -f ppo_reach_node.py; sleep 1
source ~/rosenv.sh; cd ~/MANIP-PPO
ros2 topic pub --times 40 --rate 20 /main_control/command sensor_msgs/msg/JointState "{position: [0.0, -1.0472, 0.2094, 0.0, 0.0, 0.0]}" > /dev/null
sleep 1
if [ -n "$1" ]; then python3 ~/jsrec.py /tmp/js.csv "$1" & fi
timeout 15 python3 ros2/ppo_reach_node.py --ros-args -p policy:=policy_final.npz \
    -p target_xyz:="${XYZ:-[-0.726, -1.448, 0.749]}" -p target_tu:="[0.9423584377, -2.852157604, 0.1075927581]"
wait
