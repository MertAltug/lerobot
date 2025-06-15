#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig

from cv2 import aruco


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    # TODO(Steven): Consider setting in here the keys that we want to capture/listen
    mock: bool = False


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("aruco_ee")
@dataclass
class ArucoEndEffectorTeleopConfig(KeyboardTeleopConfig):
    # Set some defaults, based on our specific setup, probably needs changing
    use_gripper: bool = True
    camera: int = 0
    camera_calibration_folder: str = "./calibration_values"
    aruco_dictionary: int = aruco.DICT_6X6_50
    marker_size: float = 0.0675
    main_marker: int = 0
    backup_marker: int = 1
    bk_offset: float = 0.085
    position_only: bool = False

