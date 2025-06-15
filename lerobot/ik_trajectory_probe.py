#!/usr/bin/env python
#
# ik_trajectory_probe.py: A single-file simulation for debugging SO101 End-Effector IK
# by running predefined trajectories and reporting aggregated errors.
#

import logging
import os
import sys
import time
from queue import Queue
from typing import Any, Generator

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

# --- Dependencies Check ---
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# SECTION 1: KINEMATICS (Pasted from kinematics.py)
# This is the user-provided kinematics code.
# ==============================================================================
# Note: The full kinematics code is included here but truncated for brevity in this view.
# Assume the full, correct code from the previous step is pasted here.

def skew_symmetric(w: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def screw_axis_to_transform(s: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    screw_axis_rot = s[:3]
    screw_axis_trans = s[3:]
    if np.allclose(screw_axis_rot, 0):
        transform = np.eye(4)
        transform[:3, 3] = screw_axis_trans * theta
    elif np.linalg.norm(screw_axis_rot) == 1:
        w_hat = skew_symmetric(screw_axis_rot)
        rot_mat = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
        t = (np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat @ w_hat) @ screw_axis_trans
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    return transform

def pose_difference_se3(pose1: NDArray[np.float32], pose2: NDArray[np.float32]) -> NDArray[np.float32]:
    rot1, rot2 = pose1[:3, :3], pose2[:3, :3]
    translation_diff = pose1[:3, 3] - pose2[:3, 3]
    rot_diff = Rotation.from_matrix(rot1 @ rot2.T)
    return np.concatenate([translation_diff, rot_diff.as_rotvec()])

def se3_error(target_pose: NDArray[np.float32], current_pose: NDArray[np.float32]) -> NDArray[np.float32]:
    pos_error = target_pose[:3, 3] - current_pose[:3, 3]
    rot_error_mat = target_pose[:3, :3] @ current_pose[:3, :3].T
    return np.concatenate([pos_error, Rotation.from_matrix(rot_error_mat).as_rotvec()])

class RobotKinematics:
    ROBOT_MEASUREMENTS = { "so_new_calibration": { "gripper": [0.33, 0.0, 0.285], "wrist": [0.30, 0.0, 0.267], "forearm": [0.25, 0.0, 0.266], "humerus": [0.06, 0.0, 0.264], "shoulder": [0.0, 0.0, 0.238], "base": [0.0, 0.0, 0.12], } }
    def __init__(self, robot_type: str = "so_new_calibration"):
        if robot_type not in self.ROBOT_MEASUREMENTS:
            robot_type = "so_new_calibration"
        self.robot_type = robot_type
        self.measurements = self.ROBOT_MEASUREMENTS[robot_type]
        self._setup_transforms()

    def _create_translation_matrix(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> NDArray[np.float32]:
        return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    def _setup_transforms(self):
        m = self.measurements
        self.gripper_X0 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.wrist_X0 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.base_X0 = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.S_BG = np.array([1, 0, 0, 0, m["gripper"][2], -m["gripper"][1]], dtype=np.float32)
        self.X_GoGt = self._create_translation_matrix(x=0.12)
        self.X_BoGo = self._create_translation_matrix(x=m["gripper"][0], y=m["gripper"][1], z=m["gripper"][2])
        self.S_BR = np.array([0, 1, 0, -m["wrist"][2], 0, m["wrist"][0]], dtype=np.float32)
        self.X_BR = self._create_translation_matrix(x=m["wrist"][0], y=m["wrist"][1], z=m["wrist"][2])
        self.S_BF = np.array([0, 1, 0, -m["forearm"][2], 0, m["forearm"][0]], dtype=np.float32)
        self.X_BF = self._create_translation_matrix(x=m["forearm"][0], y=m["forearm"][1], z=m["forearm"][2])
        self.S_BH = np.array([0, -1, 0, m["humerus"][2], 0, -m["humerus"][0]], dtype=np.float32)
        self.X_BH = self._create_translation_matrix(x=m["humerus"][0], y=m["humerus"][1], z=m["humerus"][2])
        self.S_BS = np.array([0, 0, -1, 0, 0, 0], dtype=np.float32)
        self.X_BS = self._create_translation_matrix(x=m["shoulder"][0], y=m["shoulder"][1], z=m["shoulder"][2])
        self.X_WoBo = self._create_translation_matrix(x=m["base"][0], y=m["base"][1], z=m["base"][2])

    def forward_kinematics(self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip") -> NDArray[np.float32]:
        if len(robot_pos_deg) < 5: raise ValueError(f"Expected at least 5 joint angles, got {len(robot_pos_deg)}")
        thetas = np.deg2rad(robot_pos_deg)
        t = self.X_WoBo @ screw_axis_to_transform(self.S_BS, thetas[0])
        t = t @ screw_axis_to_transform(self.S_BH, -thetas[1])
        t = t @ screw_axis_to_transform(self.S_BF, thetas[2])
        t = t @ screw_axis_to_transform(self.S_BR, thetas[3])
        t = t @ screw_axis_to_transform(self.S_BG, thetas[4])
        return t @ self.X_GoGt @ self.X_BoGo @ self.gripper_X0

    def compute_jacobian(self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip") -> NDArray[np.float32]:
        eps = 1e-5
        if len(robot_pos_deg) < 6: robot_pos_deg_6 = np.append(robot_pos_deg, 0.0)
        else: robot_pos_deg_6 = robot_pos_deg
        arm_joints = robot_pos_deg_6[:-1]
        num_joints = len(arm_joints)
        jac = np.zeros(shape=(6, num_joints))
        for i in range(num_joints):
            delta = np.zeros(num_joints)
            delta[i] = eps / 2
            jac[:, i] = pose_difference_se3(self.forward_kinematics(arm_joints + delta, frame), self.forward_kinematics(arm_joints - delta, frame)) / eps
        return jac

    def compute_positional_jacobian(self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip") -> NDArray[np.float32]:
        eps = 1e-5
        if len(robot_pos_deg) < 6: robot_pos_deg_6 = np.append(robot_pos_deg, 0.0)
        else: robot_pos_deg_6 = robot_pos_deg
        arm_joints = robot_pos_deg_6[:-1]
        num_joints = len(arm_joints)
        jac = np.zeros(shape=(3, num_joints))
        for i in range(num_joints):
            delta = np.zeros(num_joints)
            delta[i] = eps / 2
            jac[:, i] = (self.forward_kinematics(arm_joints + delta, frame)[:3, 3] - self.forward_kinematics(arm_joints - delta, frame)[:3, 3]) / eps
        return jac

    def ik(self, current_joint_pos: NDArray[np.float32], desired_ee_pose: NDArray[np.float32], **kwargs) -> NDArray[np.float32]:
        max_iter = kwargs.get("max_iterations", 10)
        lr = kwargs.get("learning_rate", 0.5)
        tol = kwargs.get("tolerance", 1e-4)
        pos_only = kwargs.get("position_only", True)
        frame = kwargs.get("frame", "gripper_tip")
        if len(current_joint_pos) < 6: state_6 = np.append(current_joint_pos, 0.0)
        else: state_6 = current_joint_pos.copy()
        for _ in range(max_iter):
            current_ee_pose = self.forward_kinematics(state_6[:-1], frame)
            if pos_only:
                error = desired_ee_pose[:3, 3] - current_ee_pose[:3, 3]
                if np.linalg.norm(error) < tol: break
                jac = self.compute_positional_jacobian(state_6, frame)
            else:
                error = se3_error(desired_ee_pose, current_ee_pose)
                if np.linalg.norm(error) < tol: break
                jac = self.compute_jacobian(state_6, frame)
            damping = 0.01
            jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping**2 * np.eye(jac.shape[0]))
            state_6[:-1] += lr * (jac_pinv @ error)
        return state_6

# ==============================================================================
# SECTION 2: MOCKS AND BASE CLASSES (Unchanged)
# ==============================================================================
class MockBaseConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class KeyboardTeleopConfig(MockBaseConfig): pass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig): pass
class SO101FollowerConfig(MockBaseConfig): pass
class SO101FollowerEndEffectorConfig(SO101FollowerConfig): pass
class DeviceNotConnectedError(Exception): pass
class DeviceAlreadyConnectedError(Exception): pass
class Teleoperator:
    def __init__(self, config): pass
class Robot:
    def __init__(self, config): pass
class MockFeetechMotorsBus:
    def __init__(self, port, motors, calibration=None):
        self.motors = motors
        self.is_connected = False
        self._motor_positions = {"shoulder_pan": 0.0, "shoulder_lift": 20.0, "elbow_flex": 90.0, "wrist_flex": -90.0, "wrist_roll": 0.0, "gripper": 50.0}
    def connect(self): self.is_connected = True
    def disconnect(self, disable_torque=True): self.is_connected = False
    def sync_read(self, register: str) -> dict[str, float]: return self._motor_positions.copy() if register == "Present_Position" else {}
    def sync_write(self, register: str, values: dict[str, float]):
        if register == "Goal_Position":
            for motor, value in values.items(): self._motor_positions[motor] = value

# ==============================================================================
# SECTION 3: REFACTORED ROBOT AND TELEOP CLASSES
# ==============================================================================
# --- Teleop classes (largely unchanged) ---
class KeyboardTeleop(Teleoperator):
    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
    @property
    def is_connected(self) -> bool: return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()
    def connect(self) -> None:
        if self.is_connected: raise DeviceAlreadyConnectedError()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
    def _on_press(self, key): self.event_queue.put((key.char if hasattr(key, "char") else key, True))
    def _on_release(self, key):
        key_val = key.char if hasattr(key, "char") else key
        self.event_queue.put((key_val, False))
        if key == keyboard.Key.esc: self.disconnect()
    def _drain_pressed_keys(self):
        while not self.event_queue.empty(): self.current_pressed[self.event_queue.get_nowait()[0]] = self.event_queue.get_nowait()[1]
    def disconnect(self) -> None:
        if self.is_connected: self.listener.stop()

class KeyboardEndEffectorTeleop(KeyboardTeleop):
    def get_action(self) -> dict[str, Any]:
        if not self.is_connected: raise DeviceNotConnectedError()
        self._drain_pressed_keys()
        deltas = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "delta_pitch": 0.0, "delta_roll": 0.0, "gripper": 1}
        for key, val in self.current_pressed.items():
            if not val: continue
            if key == keyboard.Key.up: deltas["delta_x"] = 1.0
            elif key == keyboard.Key.down: deltas["delta_x"] = -1.0
            elif key == keyboard.Key.left: deltas["delta_y"] = 1.0
            elif key == keyboard.Key.right: deltas["delta_y"] = -1.0
            elif key == keyboard.Key.shift: deltas["delta_z"] = -1.0
            elif key == keyboard.Key.shift_r: deltas["delta_z"] = 1.0
            elif key == "i": deltas["delta_pitch"] = 1.0
            elif key == "k": deltas["delta_pitch"] = -1.0
            elif key == "j": deltas["delta_roll"] = 1.0
            elif key == "l": deltas["delta_roll"] = -1.0
            elif key == keyboard.Key.ctrl_r: deltas["gripper"] = 2
            elif key == keyboard.Key.ctrl_l: deltas["gripper"] = 0
        return deltas

# --- Robot base class (unchanged) ---
class SO101Follower(Robot):
    name = "so101_follower"
    def __init__(self, config: SO101FollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = MockFeetechMotorsBus(port=config.port, motors=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])
    @property
    def is_connected(self) -> bool: return self.bus.is_connected
    def connect(self, calibrate: bool = True): self.bus.connect()
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{m}.pos": v for m, v in goal_pos.items()}
    def disconnect(self): self.bus.disconnect(self.config.disable_torque_on_disconnect)


# --- REFACTORED End Effector Class ---
EE_FRAME = "gripper_tip"
class SO101FollowerEndEffector(SO101Follower):
    name = "so101_follower_end_effector"

    def __init__(self, config: SO101FollowerEndEffectorConfig):
        super().__init__(config)
        self.kinematics = RobotKinematics(robot_type="so_new_calibration")
        self.end_effector_bounds = config.end_effector_bounds
        self.current_ee_pos = None
        self.current_joint_pos = None
        self.reset()

    def reset(self):
        """Resets the robot's internal state to its starting hardware state."""
        pos_dict = self.bus.sync_read("Present_Position")
        self.current_joint_pos = np.array([pos_dict[m] for m in self.bus.motors])
        self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos[:5], frame=EE_FRAME)
        logging.info("Robot state has been reset.")
        print(f"   Initial Joints (deg): {np.array2string(self.current_joint_pos, precision=2)}")
        print(f"   Initial EE Pose (m):\n{np.array2string(self.current_ee_pos, precision=3, suppress_small=True)}")

    def _move_to_pose(self, desired_ee_pos: np.ndarray, position_only: bool = False, verbose=False):
        """Core internal method to calculate IK and move the robot to a target pose."""
        if self.current_joint_pos is None: self.reset()

        # Apply safety bounds to the target
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(desired_ee_pos[:3, 3], self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Calculate IK
        dummy_joint_state_6 = np.append(self.current_joint_pos[:5], 0.0)
        target_joints_6dof = self.kinematics.ik(
            dummy_joint_state_6, desired_ee_pos, position_only=position_only,
            max_iterations=20, learning_rate=0.4, tolerance=1e-4
        )
        target_arm_joints = np.clip(target_joints_6dof[:5], -180.0, 180.0)

        # Verification using FK
        recalculated_ee_pos = self.kinematics.forward_kinematics(target_arm_joints, frame=EE_FRAME)

        # Calculate errors
        pos_error = np.linalg.norm(desired_ee_pos[:3, 3] - recalculated_ee_pos[:3, 3])
        rot_error_mat = desired_ee_pos[:3, :3] @ recalculated_ee_pos[:3, :3].T
        rot_error_angle = np.rad2deg(np.arccos(np.clip((np.trace(rot_error_mat) - 1) / 2, -1.0, 1.0)))
        
        # Update internal state for the next step
        # Using the *recalculated* pose is crucial to prevent error accumulation.
        self.current_ee_pos = recalculated_ee_pos.copy()
        self.current_joint_pos[:5] = target_arm_joints

        # Send action to mock hardware (arm only, gripper is handled separately)
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joint_action = {f"{name}.pos": target_arm_joints[i] for i, name in enumerate(motor_names)}
        super().send_action(joint_action)

        if verbose:
            print(f"   Target Pose:\n{np.array2string(desired_ee_pos, precision=3)}")
            print(f"   Recalculated Pose:\n{np.array2string(recalculated_ee_pos, precision=3)}")
            print(f"   Position Error: {pos_error*1000:.3f} mm, Rotation Error: {rot_error_angle:.3f} deg")

        return {"pos_error": pos_error, "rot_error": rot_error_angle}
    
    def send_action(self, action: dict[str, Any]):
        """Interprets delta actions from keyboard and calls the core move method."""
        if self.current_ee_pos is None: self.reset()

        # Build desired pose from deltas
        desired_pose = self.current_ee_pos.copy()
        # Translation
        desired_pose[:3, 3] += np.array([
            action["delta_x"] * self.config.end_effector_step_sizes["x"],
            action["delta_y"] * self.config.end_effector_step_sizes["y"],
            action["delta_z"] * self.config.end_effector_step_sizes["z"],
        ])
        # Rotation
        delta_pitch = action["delta_pitch"] * self.config.end_effector_step_sizes["pitch"]
        delta_roll = action["delta_roll"] * self.config.end_effector_step_sizes["roll"]
        delta_rot = Rotation.from_euler('yx', [delta_pitch, delta_roll]).as_matrix()
        desired_pose[:3, :3] = desired_pose[:3, :3] @ delta_rot

        # Handle Gripper
        gripper_delta = action["gripper"] - 1
        self.current_joint_pos[-1] = np.clip(self.current_joint_pos[-1] + gripper_delta * self.config.gripper_step_size, 5, self.config.max_gripper_pos)
        super().send_action({"gripper.pos": self.current_joint_pos[-1]})

        # Call the core IK and move method
        self._move_to_pose(desired_pose, position_only=False, verbose=True)

# ==============================================================================
# SECTION 4: TRAJECTORY GENERATION AND TESTING
# ==============================================================================

def generate_linear_trajectory(start_pose: np.ndarray, end_pose: np.ndarray, num_steps: int) -> Generator[np.ndarray, None, None]:
    """Generates a smooth trajectory from a start to an end pose."""
    start_pos = start_pose[:3, 3]
    end_pos = end_pose[:3, 3]
    
    # Use Scipy's Slerp for smooth rotation interpolation
    key_rots = Rotation.from_matrix([start_pose[:3, :3], end_pose[:3, :3]])
    slerp = Slerp([0, 1], key_rots)
    
    for t in np.linspace(0, 1, num_steps):
        # Linear interpolation for position (lerp)
        interp_pos = start_pos + t * (end_pos - start_pos)
        # Spherical linear interpolation for rotation (slerp)
        interp_rot_mat = slerp(t).as_matrix()
        
        # Assemble the 4x4 pose matrix
        next_pose = np.eye(4)
        next_pose[:3, :3] = interp_rot_mat
        next_pose[:3, 3] = interp_pos
        yield next_pose

def run_trajectory_test(robot: SO101FollowerEndEffector, trajectory: Generator[np.ndarray, None, None], test_name: str, position_only=False):
    """Executes a trajectory and reports the aggregated error statistics."""
    print("\n" + "="*80)
    print(f"STARTING TEST: {test_name}")
    print("="*80)
    
    robot.reset()
    pos_errors_mm = []
    rot_errors_deg = []
    
    for i, target_pose in enumerate(trajectory):
        print(f"Step {i+1}:")
        errors = robot._move_to_pose(target_pose, position_only=position_only, verbose=True)
        pos_errors_mm.append(errors["pos_error"] * 1000)
        rot_errors_deg.append(errors["rot_error"])
        time.sleep(0.02) # Simulate control loop delay

    # --- Print Summary ---
    print("\n" + "-"*80)
    print(f"TRAJECTORY TEST SUMMARY: {test_name}")
    print(f"Position Error (mm):")
    print(f"  Mean: {np.mean(pos_errors_mm):.3f}")
    print(f"  Std Dev: {np.std(pos_errors_mm):.3f}")
    print(f"  Max: {np.max(pos_errors_mm):.3f}")
    print(f"Rotation Error (deg):")
    print(f"  Mean: {np.mean(rot_errors_deg):.3f}")
    print(f"  Std Dev: {np.std(rot_errors_deg):.3f}")
    print(f"  Max: {np.max(rot_errors_deg):.3f}")
    print("-" * 80)

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================

def main_trajectory_test(robot: SO101FollowerEndEffector):
    """Defines and runs a series of trajectory tests."""
    
    # --- Test Case 1: Simple straight line move forward ---
    start_pose = robot.current_ee_pos.copy()
    end_pose_1 = start_pose.copy()
    end_pose_1[0, 3] += 0.15 # Move 15cm in the local X direction (forward)
    trajectory_1 = generate_linear_trajectory(start_pose, end_pose_1, num_steps=50)
    run_trajectory_test(robot, trajectory_1, "Move 15cm Forward (X-axis)")

    # --- Test Case 2: Move up and rotate wrist ---
    start_pose = robot.current_ee_pos.copy() # Start from where the last test ended
    end_pose_2 = start_pose.copy()
    end_pose_2[2, 3] += 0.10 # Move 10cm up (Z-axis)
    # Add a 90-degree roll (rotation around the local X-axis of the end effector)
    roll_90_deg = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    end_pose_2[:3, :3] = end_pose_2[:3, :3] @ roll_90_deg
    trajectory_2 = generate_linear_trajectory(start_pose, end_pose_2, num_steps=50)
    run_trajectory_test(robot, trajectory_2, "Move 10cm Up (Z-axis) with 90-deg Roll")

    # --- Test Case 3: Trace a square in the XY plane ---
    p1 = robot.current_ee_pos.copy()
    p2, p3, p4 = p1.copy(), p1.copy(), p1.copy()
    p2[0, 3] += 0.1 # +10cm in X
    p3[0, 3] += 0.1; p3[1, 3] += 0.1 # +10cm in X, +10cm in Y
    p4[1, 3] += 0.1 # +10cm in Y
    
    # Chain the generators together
    square_traj = (pose for segment in [
        generate_linear_trajectory(p1, p2, 25),
        generate_linear_trajectory(p2, p3, 25),
        generate_linear_trajectory(p3, p4, 25),
        generate_linear_trajectory(p4, p1, 25),
    ] for pose in segment)
    run_trajectory_test(robot, square_traj, "Trace a 10cm x 10cm Square (XY Plane)")


def main_keyboard_test(robot: SO101FollowerEndEffector, teleop_config: KeyboardEndEffectorTeleopConfig):
    """Runs the interactive keyboard test."""
    if not PYNPUT_AVAILABLE:
        logging.error("pynput is not available. Cannot run keyboard test.")
        return

    teleop = KeyboardEndEffectorTeleop(teleop_config)
    teleop.connect()
    robot.reset()

    print("\n--- Keyboard Simulation Ready ---")
    print("Control the simulated robot and watch the verbose output.")
    print("  ESC: Quit")
    
    while teleop.is_connected:
        action = teleop.get_action()
        if any(v != 0.0 for k, v in action.items() if k != 'gripper') or action.get('gripper') != 1:
            robot.send_action(action)
        time.sleep(0.05)
    
    teleop.disconnect()


if __name__ == "__main__":
    # --- Shared Config ---
    robot_config = SO101FollowerEndEffectorConfig(
        port="MOCK_PORT", disable_torque_on_disconnect=True,
        end_effector_step_sizes={"x": 0.005, "y": 0.005, "z": 0.005, "pitch": np.deg2rad(1.5), "roll": np.deg2rad(1.5)},
        end_effector_bounds={"min": np.array([-0.5, -0.5, 0.01]), "max": np.array([0.5, 0.5, 0.6])},
        gripper_step_size=5, max_gripper_pos=90,
    )
    teleop_config = KeyboardEndEffectorTeleopConfig(config=robot_config) # Pass robot config to teleop
    
    # --- Instantiate Robot ---
    robot = SO101FollowerEndEffector(config=robot_config)
    robot.connect()

    # --- CHOOSE YOUR MODE ---
    # To run the trajectory tests, use this:
    main_trajectory_test(robot)

    # To run the interactive keyboard test, comment out the line above and uncomment the line below:
    # main_keyboard_test(robot, teleop_config)
    
    robot.disconnect()
    logging.info("Simulation finished.")