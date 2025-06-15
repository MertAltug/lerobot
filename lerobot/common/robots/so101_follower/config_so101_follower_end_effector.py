# lerobot/common/robots/so101/config_so101_follower_end_effector.py

#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.common.robots.config import RobotConfig
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig


@RobotConfig.register_subclass("so101_follower_end_effector")
@dataclass
class SO101FollowerEndEffectorConfig(SO101FollowerConfig):
    """
    Configuration for the SO-101 Follower Arm with end-effector control.
    """

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-0.5, -0.5, -0.5],  # min x, y, z
            "max": [0.5, 0.5, 0.5],  # max x, y, z
        }
    )

    # Step sizes for each axis of the end-effector.
    # This controls the speed of the teleoperation.
    # Positional steps are in meters, rotational steps are in radians.
    end_effector_step_sizes: dict = field(
        default_factory=lambda: {"x": 0.5, "y": 0.5, "z": 0.5, "pitch": 1, "roll": 1}
    )

    # Maximum gripper position for clipping.
    # This value might need to be tuned based on your physical gripper's range.
    max_gripper_pos: float = 100.0

    # Step size for gripper actions, enabling incremental control.
    gripper_step_size: float = 10.0
