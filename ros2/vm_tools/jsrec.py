"""Record /joint_states (time, q1..q6) to a csv for SECONDS.  python3 jsrec.py OUT.csv SECONDS"""
import sys
import time

import rclpy
from sensor_msgs.msg import JointState

out, secs = sys.argv[1], float(sys.argv[2])
rclpy.init()
node = rclpy.create_node("jsrec")
rows = []
names = ["joint_a%d" % i for i in range(1, 7)]


def cb(m):
    if all(n in m.name for n in names):
        rows.append([time.time()] + [m.position[m.name.index(n)] for n in names])


node.create_subscription(JointState, "/joint_states", cb, 50)
t0 = time.time()
while time.time() - t0 < secs:
    rclpy.spin_once(node, timeout_sec=0.05)
with open(out, "w") as f:
    for r in rows:
        f.write(",".join(f"{v:.6f}" for v in r) + "\n")
print("rows", len(rows))
node.destroy_node()
rclpy.shutdown()
