"""Replay a recorded joint trajectory in RViz pose by pose and take a screenshot of each pose.
python3 replay.py js.csv OUT_DIR FPS [EXTRA_END_SECONDS]
The pose is published continuously (the lab bridge falls back to the sliders 0.5 s after the last command)."""
import os
import shutil
import sys
import threading
import time

import numpy as np
import rclpy
from PIL import ImageGrab
from sensor_msgs.msg import JointState

csv, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
extra = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
d = np.loadtxt(csv, delimiter=",")
t, q = d[:, 0] - d[0, 0], d[:, 1:7]
moving = np.where(np.abs(np.diff(q, axis=0)).sum(axis=1) > 1e-4)[0]
t0, t1 = t[moving[0]] - 0.3, t[moving[-1]] + extra
times = np.arange(t0, t1, 1.0 / fps)
poses = [q[min(np.searchsorted(t, s), len(q) - 1)] for s in times]

rclpy.init()
node = rclpy.create_node("replay")
pub = node.create_publisher(JointState, "/main_control/command", 10)
current = {"q": poses[0]}
stop = False


def publisher():
    while not stop:
        m = JointState()
        m.position = [float(v) for v in current["q"]]
        pub.publish(m)
        time.sleep(0.05)


th = threading.Thread(target=publisher, daemon=True)
th.start()
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
time.sleep(1.5)
for k, p in enumerate(poses):
    current["q"] = p
    time.sleep(0.4)                                   # let the bridge and RViz settle
    ImageGrab.grab(xdisplay=":0").save(os.path.join(out, f"f{k:03d}.png"))
stop = True
print("frames", len(poses))
node.destroy_node()
rclpy.shutdown()
