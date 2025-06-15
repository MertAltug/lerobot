#!/usr/bin/env python
#
# ik_probe.py: A single-file simulation for debugging SO101 End-Effector IK
#              using the actual provided RobotKinematics.
#

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

# --- Check and Import Dependencies ---
try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError:
    print("ERROR: NumPy not found. Please install it: pip install numpy")
    sys.exit(1)

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    print("ERROR: SciPy not found. The kinematics code requires it. Please install it: pip install scipy")
    sys.exit(1)

try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        raise ImportError("No display.")
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
    print("ERROR: pynput not found or no display. Please install it: pip install pynput")
    sys.exit(1)
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    print(f"ERROR: Could not import pynput: {e}")
    sys.exit(1)


# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# SECTION 1: KINEMATICS (Pasted from kinematics.py)
# This section contains the actual kinematics calculations provided.
# ==============================================================================

def skew_symmetric(w: NDArray[np.float32]) -> NDArray[np.float32]:
    """Creates the skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues_rotation(w: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    """Computes the rotation matrix using Rodrigues' formula."""
    w_hat = skew_symmetric(w)
    return np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat


def screw_axis_to_transform(s: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    """Converts a screw axis to a 4x4 transformation matrix."""
    screw_axis_rot = s[:3]
    screw_axis_trans = s[3:]

    # Pure translation
    if np.allclose(screw_axis_rot, 0) and np.linalg.norm(screw_axis_trans) == 1:
        transform = np.eye(4)
        transform[:3, 3] = screw_axis_trans * theta

    # Rotation (and potentially translation)
    elif np.linalg.norm(screw_axis_rot) == 1:
        w_hat = skew_symmetric(screw_axis_rot)
        rot_mat = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
        t = (
            np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat @ w_hat
        ) @ screw_axis_trans
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    return transform


def pose_difference_se3(pose1: NDArray[np.float32], pose2: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculates the SE(3) difference between two 4x4 homogeneous transformation matrices."""
    rot1 = pose1[:3, :3]
    rot2 = pose2[:3, :3]
    translation_diff = pose1[:3, 3] - pose2[:3, 3]
    rot_diff = Rotation.from_matrix(rot1 @ rot2.T)
    rotation_diff = rot_diff.as_rotvec()  # Axis-angle representation
    return np.concatenate([translation_diff, rotation_diff])


def se3_error(target_pose: NDArray[np.float32], current_pose: NDArray[np.float32]) -> NDArray[np.float32]:
    pos_error = target_pose[:3, 3] - current_pose[:3, 3]
    rot_target = target_pose[:3, :3]
    rot_current = current_pose[:3, :3]
    rot_error_mat = rot_target @ rot_current.T
    rot_error = Rotation.from_matrix(rot_error_mat).as_rotvec()
    return np.concatenate([pos_error, rot_error])


class RobotKinematics:
    """Robot kinematics class supporting multiple robot models."""

    ROBOT_MEASUREMENTS = {
        # ... other robot measurements omitted for brevity ...
        "so_new_calibration": {
            "gripper": [0.33, 0.0, 0.285],
            "wrist": [0.30, 0.0, 0.267],
            "forearm": [0.25, 0.0, 0.266],
            "humerus": [0.06, 0.0, 0.264],
            "shoulder": [0.0, 0.0, 0.238],
            "base": [0.0, 0.0, 0.12],
        },
    }

    def __init__(self, robot_type: str = "so_new_calibration"):
        if robot_type not in self.ROBOT_MEASUREMENTS:
            # Use a default if the specific one isn't found
            logging.warning(f"Robot type '{robot_type}' not found. Defaulting to 'so_new_calibration'.")
            robot_type = "so_new_calibration"
        
        self.robot_type = robot_type
        self.measurements = self.ROBOT_MEASUREMENTS[robot_type]
        self._setup_transforms()
        logging.info(f"Initialized RobotKinematics with '{robot_type}' measurements.")

    def _create_translation_matrix(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> NDArray[np.float32]:
        return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    def _setup_transforms(self):
        """Setup all transformation matrices and screw axes for the robot."""
        # Gripper orientation
        self.gripper_X0 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        # Wrist orientation
        self.wrist_X0 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # Base orientation
        self.base_X0 = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)

        # --- Setup Screw Axes and 0-position transforms based on measurements ---
        m = self.measurements
        # Gripper
        self.S_BG = np.array([1, 0, 0, 0, m["gripper"][2], -m["gripper"][1]], dtype=np.float32)
        self.X_GoGc = self._create_translation_matrix(x=0.07)
        self.X_GoGt = self._create_translation_matrix(x=0.12)
        self.X_BoGo = self._create_translation_matrix(x=m["gripper"][0], y=m["gripper"][1], z=m["gripper"][2])

        # Wrist
        self.S_BR = np.array([0, 1, 0, -m["wrist"][2], 0, m["wrist"][0]], dtype=np.float32)
        self.X_RoRc = self._create_translation_matrix(x=0.0035, y=-0.002)
        self.X_BR = self._create_translation_matrix(x=m["wrist"][0], y=m["wrist"][1], z=m["wrist"][2])

        # Forearm
        self.S_BF = np.array([0, 1, 0, -m["forearm"][2], 0, m["forearm"][0]], dtype=np.float32)
        self.X_ForearmFc = self._create_translation_matrix(x=0.036)
        self.X_BF = self._create_translation_matrix(x=m["forearm"][0], y=m["forearm"][1], z=m["forearm"][2])

        # Humerus
        self.S_BH = np.array([0, -1, 0, m["humerus"][2], 0, -m["humerus"][0]], dtype=np.float32)
        self.X_HoHc = self._create_translation_matrix(x=0.0475)
        self.X_BH = self._create_translation_matrix(x=m["humerus"][0], y=m["humerus"][1], z=m["humerus"][2])

        # Shoulder
        self.S_BS = np.array([0, 0, -1, 0, 0, 0], dtype=np.float32)
        self.X_SoSc = self._create_translation_matrix(x=-0.017, z=0.0235)
        self.X_BS = self._create_translation_matrix(x=m["shoulder"][0], y=m["shoulder"][1], z=m["shoulder"][2])

        # Base
        self.X_BoBc = self._create_translation_matrix(y=0.015)
        self.X_WoBo = self._create_translation_matrix(x=m["base"][0], y=m["base"][1], z=m["base"][2])

        # Pre-compute gripper post-multiplication matrix
        self._fk_gripper_post = self.X_GoGc @ self.X_BoGo @ self.gripper_X0

    def forward_kinematics(self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip") -> NDArray[np.float32]:
        """Generic forward kinematics."""
        frame = frame.lower()
        
        # Ensure input is at least 5 elements for the arm joints
        if len(robot_pos_deg) < 5:
             raise ValueError(f"Expected at least 5 joint angles, got {len(robot_pos_deg)}")

        robot_pos_rad = np.deg2rad(robot_pos_deg)

        # Extract joint angles (note the sign convention for shoulder lift).
        theta_shoulder_pan = robot_pos_rad[0]
        theta_shoulder_lift = -robot_pos_rad[1]
        theta_elbow_flex = robot_pos_rad[2]
        theta_wrist_flex = robot_pos_rad[3]
        theta_wrist_roll = robot_pos_rad[4]

        # Start with the world-to-base transform; incrementally add successive links.
        transformation_matrix = self.X_WoBo @ screw_axis_to_transform(self.S_BS, theta_shoulder_pan)
        if frame == "shoulder":
            return transformation_matrix @ self.X_SoSc @ self.X_BS

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(self.S_BH, theta_shoulder_lift)
        if frame == "humerus":
            return transformation_matrix @ self.X_HoHc @ self.X_BH

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(self.S_BF, theta_elbow_flex)
        if frame == "forearm":
            return transformation_matrix @ self.X_ForearmFc @ self.X_BF

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(self.S_BR, theta_wrist_flex)
        if frame == "wrist":
            return transformation_matrix @ self.X_RoRc @ self.X_BR @ self.wrist_X0

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(self.S_BG, theta_wrist_roll)
        if frame == "gripper":
            return transformation_matrix @ self._fk_gripper_post
        else:  # frame == "gripper_tip" or default
            return transformation_matrix @ self.X_GoGt @ self.X_BoGo @ self.gripper_X0

    def compute_jacobian(self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip") -> NDArray[np.float32]:
        """Finite differences to compute the Jacobian."""
        eps = 1e-5 # Increased epsilon for better numerical stability
        
        # BUG/FEATURE: The original code expects a 6-element input and slices it to 5 using [:-1].
        # We must adhere to this if we use the original IK.
        if len(robot_pos_deg) < 6:
            # If only 5 are provided, append a dummy to match expectation.
            robot_pos_deg_6 = np.append(robot_pos_deg, 0.0)
        else:
            robot_pos_deg_6 = robot_pos_deg

        arm_joints_deg = robot_pos_deg_6[:-1] # Slices to 5 elements
        num_joints = len(arm_joints_deg)
        jac = np.zeros(shape=(6, num_joints))
        delta = np.zeros(num_joints, dtype=np.float64)

        for el_ix in range(num_joints):
            delta[:] = 0
            delta[el_ix] = eps / 2
            # Calculate forward and backward poses
            pose_plus = self.forward_kinematics(arm_joints_deg + delta, frame)
            pose_minus = self.forward_kinematics(arm_joints_deg - delta, frame)
            
            # Calculate difference
            sdot = pose_difference_se3(pose_plus, pose_minus) / eps
            jac[:, el_ix] = sdot
        return jac

    def compute_positional_jacobian(self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip") -> NDArray[np.float32]:
        """Finite differences to compute the positional Jacobian."""
        eps = 1e-5

        # Handle the 6-element input expectation
        if len(robot_pos_deg) < 6:
            robot_pos_deg_6 = np.append(robot_pos_deg, 0.0)
        else:
            robot_pos_deg_6 = robot_pos_deg
            
        arm_joints_deg = robot_pos_deg_6[:-1] # Slices to 5 elements
        num_joints = len(arm_joints_deg)
        jac = np.zeros(shape=(3, num_joints))
        delta = np.zeros(num_joints, dtype=np.float64)

        for el_ix in range(num_joints):
            delta[:] = 0
            delta[el_ix] = eps / 2
            # Calculate forward and backward positions
            pos_plus = self.forward_kinematics(arm_joints_deg + delta, frame)[:3, 3]
            pos_minus = self.forward_kinematics(arm_joints_deg - delta, frame)[:3, 3]

            sdot = (pos_plus - pos_minus) / eps
            jac[:, el_ix] = sdot
        return jac

    def ik(
        self,
        current_joint_pos: NDArray[np.float32],
        desired_ee_pose: NDArray[np.float32],
        position_only: bool = True,
        frame: str = "gripper_tip",
        max_iterations: int = 10, # Increased default iterations from 5 to 10
        learning_rate: float = 0.5, # Reduced learning rate from 1 to 0.5 for stability
        tolerance: float = 1e-3, # Added tolerance for exit condition
    ) -> NDArray[np.float32]:
        """Inverse kinematics using gradient descent."""
        
        # Ensure input matches the 6-element expectation of the Jacobian functions
        if len(current_joint_pos) < 6:
            current_joint_state_6 = np.append(current_joint_pos, 0.0)
        else:
            current_joint_state_6 = current_joint_pos.copy()

        for _ in range(max_iterations):
            # FK requires 5 elements, Jacobian computation expects 6 (which it slices to 5 internally)
            current_ee_pose = self.forward_kinematics(current_joint_state_6[:-1], frame)
            
            if not position_only:
                error = se3_error(desired_ee_pose, current_ee_pose)
                jac = self.compute_jacobian(current_joint_state_6, frame)
                error_norm = np.linalg.norm(error)
            else:
                error = desired_ee_pose[:3, 3] - current_ee_pose[:3, 3]
                jac = self.compute_positional_jacobian(current_joint_state_6, frame)
                error_norm = np.linalg.norm(error)

            # Check if we are close enough
            if error_norm < tolerance:
                break

            # Compute joint updates using pseudo-inverse (handles singularities better than direct inverse)
            try:
                # Use a small damping factor (DLS - Damped Least Squares) for better stability
                damping = 0.01
                jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping**2 * np.eye(jac.shape[0]))
                # Alternatively, use numpy's built-in pseudo-inverse:
                # jac_pinv = np.linalg.pinv(jac)
            except np.linalg.LinAlgError:
                # If inversion fails (near singularity), stop and return current best guess
                logging.warning("Jacobian inversion failed (near singularity). Returning current joint state.")
                return current_joint_state_6

            delta_angles = jac_pinv @ error
            
            # Update the first 5 joints of the 6-element array
            current_joint_state_6[:-1] += learning_rate * delta_angles

        return current_joint_state_6


# ==============================================================================
# SECTION 2: MOCKS AND BASE CLASSES
# This section contains dummy classes to replace hardware dependencies.
# ==============================================================================

# --- Mock Config Classes ---
class MockBaseConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
class KeyboardTeleopConfig(MockBaseConfig): pass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig): pass
class SO101FollowerConfig(MockBaseConfig): pass
class SO101FollowerEndEffectorConfig(SO101FollowerConfig): pass

# --- Mock Robot/Teleoperator Base Classes and Errors ---
class DeviceNotConnectedError(Exception): pass
class DeviceAlreadyConnectedError(Exception): pass
class Teleoperator:
    def __init__(self, config): pass
class Robot:
    def __init__(self, config): pass

# --- Mock FeetechMotorsBus (Hardware Abstraction) ---
class MockFeetechMotorsBus:
    """A mock motor bus that simulates the robot's hardware."""
    def __init__(self, port, motors, calibration=None):
        self.port = port
        self.motors = motors
        self.is_connected = False
        # Initialize motor positions to a safe, reachable starting pose (in degrees)
        self._motor_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 20.0,   # Slightly lifted
            "elbow_flex": 90.0,      # 90 degrees
            "wrist_flex": -90.0,     # Straight wrist
            "wrist_roll": 0.0,
            "gripper": 50.0,         # 0-100 range
        }
        logging.info(f"Initialized MockFeetechMotorsBus with start positions: {self._motor_positions}")

    def connect(self):
        self.is_connected = True
    def disconnect(self, disable_torque=True):
        self.is_connected = False
    def sync_read(self, register: str) -> dict[str, float]:
        if register == "Present_Position":
            return self._motor_positions.copy()
        return {}
    def sync_write(self, register: str, values: dict[str, float]):
        if register == "Goal_Position":
            for motor, value in values.items():
                if motor in self._motor_positions:
                    self._motor_positions[motor] = value


# ==============================================================================
# SECTION 3: TELEOPERATION AND ROBOT CLASSES (With Probing Logic)
# The code from your files, adapted to use the mocks and kinematics, with
# probing added to the EndEffector class.
# ==============================================================================

# --- From teleop_keyboard.py ---
class KeyboardTeleop(Teleoperator):
    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    def connect(self) -> None:
        if self.is_connected: raise DeviceAlreadyConnectedError()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        logging.info("Keyboard listener connected.")

    def _on_press(self, key):
        key_val = key.char if hasattr(key, "char") else key
        self.event_queue.put((key_val, True))

    def _on_release(self, key):
        key_val = key.char if hasattr(key, "char") else key
        self.event_queue.put((key_val, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def disconnect(self) -> None:
        if self.is_connected and self.listener is not None:
            self.listener.stop()

class KeyboardEndEffectorTeleop(KeyboardTeleop):
    def get_action(self) -> dict[str, Any]:
        if not self.is_connected: raise DeviceNotConnectedError()
        self._drain_pressed_keys()
        delta_x, delta_y, delta_z, delta_pitch, delta_roll = 0.0, 0.0, 0.0, 0.0, 0.0
        gripper_action = 1

        # Check continuous pressed keys
        for key, val in self.current_pressed.items():
            if not val: continue
            
            if key == keyboard.Key.up: delta_x = 1.0
            elif key == keyboard.Key.down: delta_x = -1.0
            elif key == keyboard.Key.left: delta_y = 1.0
            elif key == keyboard.Key.right: delta_y = -1.0
            elif key == keyboard.Key.shift: delta_z = -1.0
            elif key == keyboard.Key.shift_r: delta_z = 1.0
            elif key == "i": delta_pitch = 1.0
            elif key == "k": delta_pitch = -1.0
            elif key == "j": delta_roll = 1.0
            elif key == "l": delta_roll = -1.0
            elif key == keyboard.Key.ctrl_r: gripper_action = 2 # open
            elif key == keyboard.Key.ctrl_l: gripper_action = 0 # close
        
        action_dict = {
            "delta_x": delta_x, "delta_y": delta_y, "delta_z": delta_z,
            "delta_pitch": delta_pitch, "delta_roll": delta_roll, "gripper": gripper_action
        }
        return action_dict

# --- From so101_follower.py ---
class SO101Follower(Robot):
    name = "so101_follower"
    def __init__(self, config: SO101FollowerConfig):
        super().__init__(config)
        self.config = config
        # Use the Mock bus
        self.bus = MockFeetechMotorsBus(
            port=self.config.port,
            motors=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        )

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected: raise DeviceAlreadyConnectedError()
        self.bus.connect()
        logging.info(f"Simulated {self.name} connected.")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected: raise DeviceNotConnectedError()
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected: raise DeviceNotConnectedError()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        logging.info(f"Simulated {self.name} disconnected.")

# --- From so101_follower_end_effector.py (with Probing Logic) ---
EE_FRAME = "gripper_tip"

class SO101FollowerEndEffector(SO101Follower):
    name = "so101_follower_end_effector"

    def __init__(self, config: SO101FollowerEndEffectorConfig):
        super().__init__(config)
        self.config = config
        
        # Use the real kinematics
        self.kinematics = RobotKinematics(robot_type="so_new_calibration")
        
        self.end_effector_bounds = self.config.end_effector_bounds
        self.current_ee_pos = None
        self.current_joint_pos = None

    def reset(self):
        self.current_ee_pos = None
        self.current_joint_pos = None
        logging.info("Internal state reset. Will re-read from mock hardware.")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        # --- Initialize state on first run (Read from Mock Hardware) ---
        if self.current_joint_pos is None:
            current_joint_pos_dict = self.bus.sync_read("Present_Position")
            # Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
            self.current_joint_pos = np.array([
                current_joint_pos_dict["shoulder_pan"],
                current_joint_pos_dict["shoulder_lift"],
                current_joint_pos_dict["elbow_flex"],
                current_joint_pos_dict["wrist_flex"],
                current_joint_pos_dict["wrist_roll"],
                current_joint_pos_dict["gripper"]
            ])
        
        if self.current_ee_pos is None:
            # FK needs the 5 arm joints
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos[:5], frame=EE_FRAME)

        # ----------------------------------------------------------------------
        # PROBE REPORT - Diagnostic output to check IK/FK consistency
        # ----------------------------------------------------------------------
        print("\n" + "="*80)
        print(f"[{time.strftime('%H:%M:%S')}] PROBE REPORT")
        print("="*80)
        
        # PROBE 1: INPUT ACTION
        print(f"1. INPUT: Keyboard delta action\n   {action}")

        # PROBE 2: CURRENT STATE (Before calculation)
        print(f"\n2. STATE (Before):")
        print(f"   Current Joints (deg): {np.array2string(self.current_joint_pos, precision=3)}")
        print(f"   Current EE Pose (m):\n{np.array2string(self.current_ee_pos, precision=4, suppress_small=True)}")

        # --- Original action processing logic ---
        # Calculate desired translation
        delta_ee_pos = np.array([
            action["delta_x"] * self.config.end_effector_step_sizes["x"],
            action["delta_y"] * self.config.end_effector_step_sizes["y"],
            action["delta_z"] * self.config.end_effector_step_sizes["z"],
        ], dtype=np.float32)
        desired_ee_translation = self.current_ee_pos[:3, 3] + delta_ee_pos

        # Calculate desired rotation
        delta_pitch = action["delta_pitch"] * self.config.end_effector_step_sizes["pitch"]
        delta_roll = action["delta_roll"] * self.config.end_effector_step_sizes["roll"]
        cp, sp = np.cos(delta_pitch), np.sin(delta_pitch)
        cr, sr = np.cos(delta_roll), np.sin(delta_roll)
        rot_y_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rot_x_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        delta_rot = rot_x_roll @ rot_y_pitch
        desired_ee_rot = self.current_ee_pos[:3, :3] @ delta_rot

        # Assemble desired pose
        desired_ee_pos = np.eye(4)
        desired_ee_pos[:3, :3] = desired_ee_rot
        desired_ee_pos[:3, 3] = desired_ee_translation

        # Apply bounds
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # PROBE 3: TARGET POSE
        print(f"\n3. TARGET: Desired EE Pose after applying deltas and bounds")
        print(f"   Desired EE Pose (m):\n{np.array2string(desired_ee_pos, precision=4, suppress_small=True)}")

        # --- Inverse Kinematics Calculation ---
        # WORKAROUND: The kinematics.py IK functions expect a 6-element array as input, which they slice [:-1] internally.
        # So we pass the 5 arm joints + a dummy value (like the current gripper position or 0).
        current_arm_joints_5 = self.current_joint_pos[:5]
        dummy_joint_state_6 = np.append(current_arm_joints_5, 0.0)

        target_joint_values_6dof = self.kinematics.ik(
            dummy_joint_state_6, 
            desired_ee_pos, 
            position_only=False, # Try to match position AND rotation
            frame=EE_FRAME,
            max_iterations=20, # Give the solver more time to converge
            learning_rate=0.2, # Use a slightly slower learning rate for stability
            tolerance=1e-4
        )
        
        # Extract the 5 arm joints
        target_joint_values_in_degrees = target_joint_values_6dof[:5]
        target_joint_values_in_degrees = np.clip(target_joint_values_in_degrees, -180.0, 180.0)

        # PROBE 4: IK RESULT
        print(f"\n4. IK RESULT: Target joint angles from IK solver (Arm only)")
        print(f"   Target Joints (deg): {np.array2string(target_joint_values_in_degrees, precision=3)}")
        joint_change = np.linalg.norm(target_joint_values_in_degrees - current_arm_joints_5)
        print(f"   Total Joint Change (deg L2 norm): {joint_change:.3f}")


        # PROBE 5: VERIFICATION (Forward Kinematics Check)
        # Calculate where these new joints *actually* put the end effector
        recalculated_ee_pos = self.kinematics.forward_kinematics(target_joint_values_in_degrees, frame=EE_FRAME)
        print(f"\n5. VERIFICATION: EE Pose recalculated from IK result (using FK)")
        print(f"   Recalculated EE Pose (m):\n{np.array2string(recalculated_ee_pos, precision=4, suppress_small=True)}")

        # PROBE 6: ERROR CALCULATION (Target vs Actual)
        pos_error_vec = desired_ee_pos[:3, 3] - recalculated_ee_pos[:3, 3]
        pos_error_norm = np.linalg.norm(pos_error_vec)
        
        # Calculate rotational error angle
        try:
            rot_error_mat = desired_ee_pos[:3, :3] @ recalculated_ee_pos[:3, :3].T
            rot_error_trace = np.trace(rot_error_mat)
            # Angle = arccos((trace(R) - 1) / 2)
            rot_error_angle = np.rad2deg(np.arccos(np.clip((rot_error_trace - 1) / 2, -1.0, 1.0)))
        except Exception:
            rot_error_angle = float('nan')

        print(f"\n6. ERROR (TARGET vs VERIFICATION):")
        print(f"   Position Error Vector (mm): {np.array2string(pos_error_vec*1000, precision=2)}")
        print(f"   Position Error Norm: {pos_error_norm*1000:.3f} mm")
        print(f"   Rotation Error Angle: {rot_error_angle:.3f} degrees")
        print("="*80 + "\n")
        
        # --- Prepare joint space action for the 5 arm motors ---
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joint_action = {
            f"{name}.pos": target_joint_values_in_degrees[i]
            for i, name in enumerate(motor_names)
        }

        # Handle gripper action
        gripper_delta_action = action["gripper"] # 0=close, 1=stay, 2=open
        new_gripper_pos = np.clip(
            self.current_joint_pos[-1] + (gripper_delta_action - 1) * self.config.gripper_step_size,
            5, self.config.max_gripper_pos
        )
        joint_action["gripper.pos"] = new_gripper_pos

        # --- Update internal state for the next step ---
        # ! Important: Should we update with the desired pose or the recalculated pose?
        # Original code used desired_ee_pos. If IK is inaccurate, errors will accumulate.
        # If we use recalculated_ee_pos, the robot will try to correct the IK error on the next step.
        self.current_ee_pos = recalculated_ee_pos.copy() # Using recalculated is usually safer
        # self.current_ee_pos = desired_ee_pos.copy() # Original behavior

        self.current_joint_pos = np.append(target_joint_values_in_degrees, new_gripper_pos)
        
        # Send joint space action to parent class (Mock hardware)
        return super().send_action(joint_action)


# ==============================================================================
# SECTION 4: MAIN SIMULATION LOOP
# This block sets up and runs the simulation.
# ==============================================================================

if __name__ == "__main__":
    # --- Create configuration objects ---
    teleop_config = KeyboardEndEffectorTeleopConfig()
    
    robot_config = SO101FollowerEndEffectorConfig(
        port="MOCK_PORT",
        disable_torque_on_disconnect=True,
        # Define step sizes for end-effector control
        end_effector_step_sizes={
            "x": 0.005, "y": 0.005, "z": 0.005, # 5 mm translation steps
            "pitch": np.deg2rad(1.5), "roll": np.deg2rad(1.5) # 1.5 degree rotation steps
        },
        # Define safety bounds for the end-effector (in meters)
        end_effector_bounds={
            "min": np.array([-0.5, -0.5, 0.01]),
            "max": np.array([0.5, 0.5, 0.6]),
        },
        gripper_step_size=5,
        max_gripper_pos=90,
    )

    # --- Instantiate the simulated robot and teleop controller ---
    robot = SO101FollowerEndEffector(config=robot_config)
    teleop = KeyboardEndEffectorTeleop(config=teleop_config)

    try:
        # --- Connect devices ---
        robot.connect()
        teleop.connect()
        robot.reset() # Initialize state by reading from the mock hardware

        print("\n--- Simulation Ready ---")
        print("Using Real Kinematics. Monitor the PROBE REPORT for errors.")
        print("Control the simulated robot with the keyboard:")
        print("  - Arrow Keys: Move in X/Y plane")
        print("  - L/R Shift: Move in Z (up/down)")
        print("  - i/k: Pitch")
        print("  - j/l: Roll")
        print("  - L/R Ctrl: Close/Open Gripper")
        print("  - ESC: Quit")
        print("--------------------------\n")
        
        # --- Main loop ---
        while teleop.is_connected:
            # 1. Get action from keyboard
            action = teleop.get_action()

            # 2. Only send action if a key is pressed (value != 0 or gripper != 1)
            if any(v != 0.0 for k, v in action.items() if k != 'gripper') or action.get('gripper') != 1:
                 robot.send_action(action)
            
            # 3. Control loop frequency
            time.sleep(0.05) # 20 Hz loop

    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt, shutting down.")
    except Exception as e:
        logging.error(f"An exception occurred: {e}", exc_info=True)
    finally:
        # --- Graceful shutdown ---
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        logging.info("Simulation finished.")