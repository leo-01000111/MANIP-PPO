# Run the PPO policy in the lab simulator (ROS 2 VM)

The node `ppo_reach_node.py` replaces `control.cpp`. It reads the joints and the tool pose,
asks the numpy policy for an action, and sends a joint command every 0.05 s.
No torch is needed. Only numpy, rclpy and tf2_ros (all in the ROS 2 install).

## Verified in the school VM (2026-11-24, Ubuntu 24.04, ROS 2 Jazzy)
- joint states: `/joint_states` (published by the lab's `arm_bridge.py`)
- command: `/main_control/command` (sensor_msgs/JointState; the bridge accepts positions or velocities)
- tool pose: TF `base_link` -> `tool0` agrees exactly with `src/kr16.py` (position and rotation)
- joint names: `joint_a1` ... `joint_a6`
- **Use `command_type:=position` (the default).** It matches training (q_next = q + qdot * dt).
  With `velocity`, the bridge keeps the last velocity until the next command; the tool gets within
  2-7 cm and then oscillates.
- Result: from the slider start q = [0, -1.05, 0.21, 0, 0, 0] to target
  `target_xyz:="[0.4212, 0.428, 1.0862]"`, `target_tu:="[-1.2667, 0.7953, 0.4718]"`:
  REACHED, 5.8 mm in 3.0 s. Lab target A is one of the targets this policy misses (about half are missed).
- If the lab GUI crashes with a TkAgg error when launched from a script, `export MPLBACKEND=QtAgg` first
  (the VM's `.bashrc` does this in a normal terminal).

All of them are parameters. Step 3 checks them.

## Steps (on your Windows PC)
1. Train and export: `python src/export_numpy.py runs/<name>/final_model.zip policy.npz`
2. Test without ROS: `python ros2/offline_check.py --policy policy.npz`
   It must print a final position error below 0.01 m.

## Steps (in the VM)
1. Copy the whole repo folders `src/` and `ros2/` and the file `policy.npz` to the VM, for example into `~/ppo/`.
   Keep `src/` and `ros2/` side by side.
2. Open terminal 1. Start the simulator:
   `ros2ws` then `ros2 launch ecn_manip simulation_launch.py robot:=kr16`
3. Open terminal 2. Check the names:
   - `ros2 topic list` (is there `/joint_states` and `/main_control/command`?)
   - `ros2 topic info /main_control/command` (type and who listens)
   - `ros2 topic echo /joint_states --once` (joint names)
   - `ros2 run tf2_ros tf2_echo base_link tool0`
4. Stop the `control.cpp` program (red square in Qt Creator). Two programs must not send commands at the same time.
5. Dry run (sends nothing):
   `cd ~/ppo && python3 ros2/ppo_reach_node.py --ros-args -p policy:=policy.npz -p dry_run:=true`
   You should see "err ... m" lines. Stop with Ctrl+C.
6. Real run:
   `python3 ros2/ppo_reach_node.py --ros-args -p policy:=policy.npz`
7. Start from a pose that is not far from a joint limit (use the manual joint sliders first).

## What you should see in RViz
The robot moves smoothly toward the `target` frame (lab target A by default).
The terminal prints the error every 0.5 s, then `REACHED: position error 0.00x m ...`.
If it prints `TIMEOUT`, the final error is still logged.

## Useful parameters (`-p name:=value`)
- `target_xyz:="[0.8, -1.18, 0.49]"` and `target_tu:="[0.94, -2.85, 0.11]"` (target position, orientation as theta*u)
- `command_type:=velocity` sends joint velocities instead of positions
- `pose_source:=model` uses `src/kr16.py` as the pose sensor (if TF does not work)
- `check_rot:=true` also needs 5 degrees of orientation (only for a policy trained on the full pose)
- `timeout:=15.0`, `pos_tol:=0.01`, `src_dir:=/path/to/src`
- `command_topic`, `joint_state_topic`, `joint_names`, `base_frame`, `tool_frame`

## If something is wrong
- "waiting for joint states": wrong `joint_state_topic`. Pass the right one.
- "no TF": check the frame names in step 3, or use `pose_source:=model`.
- Robot does not move: wrong `command_topic`, wrong message type, or `control.cpp` still running.
  The simulator bridge (`arm_bridge.py`) falls back to the manual sliders if no command arrives for 0.5 s.
- Robot moves the wrong way: try `command_type:=velocity`, and check the joint order.
- Send me: the output of the four commands in step 3 and the first 20 lines of the node.

## If the topics cannot be driven from Python
Run the policy offline and replay the result from `control.cpp`:
`python ros2/offline_check.py --policy policy.npz --csv traj.csv`
The CSV has `t,q1..q6,pos_err` every 0.05 s. In `control.cpp`, read it once into a vector, and in
one exercise block call `robot->setJointPosition(qCommand)` with row `int((t - t0) / 0.05)`.
This is the same call the position exercises already use.
