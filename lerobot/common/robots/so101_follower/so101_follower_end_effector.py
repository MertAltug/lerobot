# lerobot/common/robots/so101/so101_follower_end_effector.py

#!/usr/bin/env python

import logging
from typing import Any

import numpy as np

from lerobot.common.errors import DeviceNotConnectedError
from lerobot.common.model.kinematics import RobotKinematics
from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus

from .config_so101_follower_end_effector import SO101FollowerEndEffectorConfig
from .so101_follower import SO101Follower

logger = logging.getLogger(__name__)
EE_FRAME = "gripper_tip"


class SO101FollowerEndEffector(SO101Follower):
    """
    SO101Follower robot with end-effector space control.
    Inherits from SO101Follower and transforms end-effector actions to joint space.
    """

    config_class = SO101FollowerEndEffectorConfig
    name = "so101_follower_end_effector"

    def __init__(self, config: SO101FollowerEndEffectorConfig):
        # Initialize the parent SO101Follower class
        super().__init__(config)

        # Override the bus from the parent to ensure we use DEGREES, as IK outputs degrees.
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.config = config

        # ################################## WARNING ###################################
        # The kinematics.py file does not have a specific "so101" calibration.
        # We are using "so_new_calibration" as a placeholder.
        # This might work well, but for optimal precision, you may need to measure
        # your SO101 arm and create a new entry in RobotKinematics.ROBOT_MEASUREMENTS.
        # ############################################################################
        self.kinematics = RobotKinematics(robot_type="so_new_calibration")

        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        # Internal state for optimization
        self.current_ee_pos = None
        self.current_joint_pos = None

    @property
    def action_features(self) -> dict[str, Any]:
        """Define action features for end-effector control."""
        return {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "delta_x": 0,
                "delta_y": 1,
                "delta_z": 2,
                "delta_pitch": 3,
                "delta_roll": 4,
                "gripper": 5,
            },
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Default to zero action if keys are missing
        required_keys = ["delta_x", "delta_y", "delta_z"]
        if not self.config.position_only:
            required_keys.append("delta_pitch")
            required_keys.append("delta_roll")

        if not (isinstance(action, dict) and all(k in action for k in required_keys)):
            logger.warning(
                "Expected action keys %s, got %s",
                required_keys,
                list(action.keys()) if isinstance(action, dict) else action,
            )
            action = {key: 0.0 for key in required_keys}
            action["gripper"] = 1.0  # stay

        if "gripper" not in action:
            action["gripper"] = 1.0  # Default to "stay"

        if self.current_joint_pos is None:
            current_joint_pos_dict = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([current_joint_pos_dict[name] for name in self.bus.motors])

        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos[:5], frame=EE_FRAME)

        # Apply deltas to current end-effector pose
        # 1. Translation
        delta_ee_pos = np.array(
            [
                action["delta_x"] * self.config.end_effector_step_sizes["x"],
                action["delta_y"] * self.config.end_effector_step_sizes["y"],
                action["delta_z"] * self.config.end_effector_step_sizes["z"],
            ],
            dtype=np.float32,
        )
        desired_ee_translation = self.current_ee_pos[:3, 3] + delta_ee_pos

        # 2. Rotation
        desired_ee_rot = self.current_ee_pos[:3, :3]
        if not self.config.position_only:
            delta_pitch = action["delta_pitch"] * self.config.end_effector_step_sizes["pitch"]
            delta_roll = action["delta_roll"] * self.config.end_effector_step_sizes["roll"]

            # Create rotation matrices for pitch (Y) and roll (X) in the EE frame
            cp = np.cos(delta_pitch)
            sp = np.sin(delta_pitch)
            cr = np.cos(delta_roll)
            sr = np.sin(delta_roll)

            rot_y_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            rot_x_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

            # Post-multiply to apply intrinsic rotations: first roll, then pitch.
            delta_rot = rot_x_roll @ rot_y_pitch
            desired_ee_rot = self.current_ee_pos[:3, :3] @ delta_rot

        # Assemble the desired pose
        desired_ee_pos = np.eye(4)
        if not self.config.position_only:
            desired_ee_pos[:3, :3] = desired_ee_rot
        desired_ee_pos[:3, 3] = desired_ee_translation

        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # WORKAROUND: The kinematics.py `compute_jacobian` has a bug where it slices `[:-1]`,
        # reducing the joint array size from 5 to 4 and causing an IndexError.
        # We pass a 6-element array so that after slicing it becomes a 5-element array,
        # which is what `forward_kinematics` expects.
        current_arm_joints = self.current_joint_pos[:5]
        dummy_joint_state = np.append(current_arm_joints, 0.0)

        # Calculate inverse kinematics for the full pose
        target_joint_values_6dof = self.kinematics.ik(
            dummy_joint_state, desired_ee_pos, position_only=self.config.position_only, frame=EE_FRAME,max_iterations=20 , learning_rate=0.5
        )

        # We only care about the first 5 joint values from the result
        target_joint_values_in_degrees = target_joint_values_6dof[:5]

        target_joint_values_in_degrees = np.clip(target_joint_values_in_degrees, -180.0, 180.0)

        # Prepare joint space action for the 5 arm motors
        joint_action = {
            f"{key}.pos": target_joint_values_in_degrees[i]
            for i, key in enumerate(list(self.bus.motors.keys())[:-1])
        }

        # Handle gripper action
        gripper_delta_action = action["gripper"]
        joint_action["gripper.pos"] = np.clip(
            self.current_joint_pos[-1] + (gripper_delta_action - 1) * self.config.gripper_step_size,
            5,
            self.config.max_gripper_pos,
        )

        # Update internal state for the next step
        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = np.append(target_joint_values_in_degrees, joint_action["gripper.pos"])

        # Send joint space action to parent class (SO101Follower) for hardware execution
        return super().send_action(joint_action)

    def reset(self):
        """Reset the internal state, forcing a re-read from hardware on the next action."""
        self.current_ee_pos = None
        self.current_joint_pos = None
