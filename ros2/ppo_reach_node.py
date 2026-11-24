#!/usr/bin/env python3
"""ROS 2 node: run the trained PPO reaching policy (numpy only) on the lab simulator (ecn_manip, KR16).

Every 0.05 s: read q (joint states) and the tool pose (TF base_link -> tool0, the measured pose; with
pose_source:=model it uses src/kr16.py fMe(q), which then plays the role of the sensor), build the
observation with the same build_obs as in training, get the action, send the motion command, and stop
when the target is reached or the timeout is over.

ASSUMED (not verified, taken from the lab notes): joint states on /joint_states, command on
/main_control/command (sensor_msgs/JointState), TF frames base_link and tool0, joints joint_a1..joint_a6.
All of them are parameters. Run:  python3 ppo_reach_node.py --ros-args -p policy:=policy.npz
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import ppo_core


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PPOReach(Node):
    def __init__(self):
        super().__init__("ppo_reach")
        P = self.declare_parameter
        P("policy", "policy.npz")
        P("src_dir", "")
        P("target_xyz", ppo_core.TARGET_A[:3])
        P("target_tu", ppo_core.TARGET_A[3:])     # orientation of the target as theta*u (rad)
        P("joint_state_topic", "/joint_states")
        P("command_topic", "/main_control/command")
        P("command_type", "position")             # "position" (q + qdot*dt) or "velocity" (qdot)
        P("joint_names", ["joint_a%d" % i for i in range(1, 7)])
        P("pose_source", "tf")                    # "tf" (measured) or "model" (kr16.fMe(q))
        P("base_frame", "base_link")
        P("tool_frame", "tool0")
        P("pos_tol", 0.01)
        P("rot_tol_deg", 5.0)                     # used only if check_rot is true
        P("check_rot", False)
        P("timeout", 10.0)                        # s
        P("dry_run", False)                       # compute and log, send nothing

        def g(n):
            return self.get_parameter(n).value

        ppo_core.add_src_to_path(g("src_dir"))
        from kr16 import KR16
        self.world = KR16()
        M_star = ppo_core.pose_matrix(g("target_xyz"), g("target_tu"))
        self.ctl = ppo_core.PPOController(g("policy"), M_star)
        self.names = list(g("joint_names"))
        self.cmd_type, self.pose_source = g("command_type"), g("pose_source")
        self.pos_tol, self.rot_tol = g("pos_tol"), np.deg2rad(g("rot_tol_deg"))
        self.check_rot, self.timeout, self.dry = g("check_rot"), g("timeout"), g("dry_run")
        self.base, self.tool = g("base_frame"), g("tool_frame")
        self.q = None
        self.t0 = None
        self.sub = self.create_subscription(JointState, g("joint_state_topic"), self.on_js, 10)
        self.pub = self.create_publisher(JointState, g("command_topic"), 10)
        if self.pose_source == "tf":
            from tf2_ros import Buffer, TransformListener
            self.tfbuf = Buffer()
            self.tfl = TransformListener(self.tfbuf, self)
        self.timer = self.create_timer(ppo_core.DT, self.tick)
        self.get_logger().info("obs_q=%s multiscale=%s history=%d target=%s" % (
            self.ctl.pol.obs_q, self.ctl.pol.multiscale_obs, self.ctl.pol.history, np.round(M_star[:3, 3], 3)))

    def on_js(self, msg):
        if msg.name and set(self.names) <= set(msg.name):
            idx = [list(msg.name).index(n) for n in self.names]
            self.q = np.array([msg.position[i] for i in idx])
        elif len(msg.position) >= 6:
            self.q = np.array(msg.position[:6])      # no names: assume the joint order

    def tool_pose(self):
        if self.pose_source == "tf":
            try:
                t = self.tfbuf.lookup_transform(self.base, self.tool, rclpy.time.Time())
            except Exception as e:  # noqa
                self.get_logger().warn("no TF %s -> %s yet: %s" % (self.base, self.tool, e),
                                       throttle_duration_sec=2.0)
                return None
            tr, r = t.transform.translation, t.transform.rotation
            M = np.eye(4)
            M[:3, :3] = quat_to_R(r.x, r.y, r.z, r.w)
            M[:3, 3] = [tr.x, tr.y, tr.z]
            return M
        return self.world.fMe(self.q)               # model: plays the role of the measured pose

    def send(self, q, qdot):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = self.names
        if self.cmd_type == "velocity":
            m.velocity = [float(v) for v in qdot]
        else:
            m.position = [float(v) for v in self.ctl.next_q(q, qdot)]
        if not self.dry:
            self.pub.publish(m)

    def tick(self):
        if self.q is None:
            self.get_logger().info("waiting for joint states ...", throttle_duration_sec=2.0)
            return
        M = self.tool_pose()
        if M is None:
            return
        if self.t0 is None:
            self.t0 = time.time()
        e_p, e_r = self.ctl.errors(M)
        done = e_p < self.pos_tol and (not self.check_rot or e_r < self.rot_tol)
        timed_out = time.time() - self.t0 > self.timeout
        if done or timed_out:
            if self.cmd_type == "velocity":
                self.send(self.q, np.zeros(6))
            self.get_logger().info("%s: position error %.4f m, orientation error %.2f deg, time %.1f s" % (
                "REACHED" if done else "TIMEOUT", e_p, np.rad2deg(e_r), time.time() - self.t0))
            self.timer.cancel()
            raise SystemExit
        qdot = self.ctl.step(self.q, M)
        self.send(self.q, qdot)
        self.get_logger().info("err %.3f m  q1..q3 %s" % (e_p, np.round(self.q[:3], 2)),
                               throttle_duration_sec=0.5)


def main():
    rclpy.init()
    node = PPOReach()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    if rclpy.ok():                 # Ctrl+C may already have shut the context down
        rclpy.shutdown()


if __name__ == "__main__":
    main()
