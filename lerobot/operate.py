import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import draccus
import os
import numpy as np
import rerun as rr

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

from lerobot.common.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower , SO101FollowerEndEffector ,SO101FollowerEndEffectorConfig
from lerobot.common.teleoperators.keyboard import KeyboardEndEffectorTeleop , KeyboardEndEffectorTeleopConfig, ArucoEndEffectorTeleopConfig, ArucoEndEffectorTeleop

robot_config = SO101FollowerEndEffectorConfig(
    port="/dev/ttyACM0",
    id="follower_0_ee1",
)

# teleop_config = KeyboardEndEffectorTeleopConfig(
#     id="teleop_keyboard_ee",
# )

teleop_config = ArucoEndEffectorTeleopConfig(
    id="teleop_aruco_ee",
    camera_calibration_folder = os.getcwd() + "/calibration_values/"
)

robot = SO101FollowerEndEffector(robot_config)
teleop_device = ArucoEndEffectorTeleop(teleop_config)
# teleop_device = KeyboardEndEffectorTeleop(teleop_config)

# robot.connect(calibrate=False)
# robot.calibrate()
# robot.disconnect()

robot.connect()

print("Robot Connected...")

teleop_device.connect()
print("Teleoperator connected...")


def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if display_data:
            observation = robot.get_observation()
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation_{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation_{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action_{act}", rr.Scalar(val))

        robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)

try:
    teleop_loop(teleop_device, robot, 30)
except KeyboardInterrupt:
    pass
finally:
    robot.disconnect()
    teleop_device.disconnect()

# while True:
#     action = teleop_device.get_action()
#     robot.send_action(action)
