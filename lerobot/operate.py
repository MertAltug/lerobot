from lerobot.common.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower , SO101FollowerEndEffector ,SO101FollowerEndEffectorConfig
from lerobot.common.teleoperators.keyboard import KeyboardEndEffectorTeleop , KeyboardEndEffectorTeleopConfig

robot_config = SO101FollowerEndEffectorConfig(
    port="/dev/ttyACM0",
    id="follower_0_ee1",
)

teleop_config = KeyboardEndEffectorTeleopConfig(
    id="teleop_keyboard_ee",
)

robot = SO101FollowerEndEffector(robot_config)
teleop_device = KeyboardEndEffectorTeleop(teleop_config)

# robot.connect(calibrate=False)
# robot.calibrate()
# robot.disconnect()

robot.connect()

teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)