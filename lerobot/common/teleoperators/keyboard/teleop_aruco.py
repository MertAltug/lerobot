import cv2
import logging
import numpy as np
import os
import sys
import time

from cv2 import aruco
from queue import Queue
from typing import Any

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_keyboard import ArucoEndEffectorTeleopConfig, KeyboardTeleopConfig
from .teleop_keyboard import KeyboardTeleop


PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")

class ArucoEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = ArucoEndEffectorTeleopConfig
    name = "aruco_ee"

    def __init__(self, config: ArucoEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
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
        else:
            return {
                "dtype": "float32",
                "shape": (5,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_pitch": 3, "delta_roll": 4},
            }

    def get_capture_frame(self):
        ret, frame = self.capture.read()
        if frame is None:
            raise ConnectionError(
                "Failed to capture camera frame."
            )

        return frame

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

        self.capture = cv2.VideoCapture(self.config.camera)
        self.dictionary = aruco.getPredefinedDictionary(self.config.aruco_dictionary)
        self.detector_params = aruco.DetectorParameters()

        try:
            self.camera_matrix = np.load(self.config.camera_calibration_folder + "camera_matrix.npy")
            self.dist_coeffs = np.load(self.config.camera_calibration_folder + "dist_coeffs.npy")
        except:
            error = "Unable to find the camera calibration Values! Directory is " + os.getcwd()
            raise ConnectionError(error)

        ret = -1
        while ret == -1:
            # Tries to get an initial estimation of the position in a loop
            initial_frame = self.get_capture_frame()
            ret, self.current_state, self.current_reference = self.calculate_state(initial_frame)

        state_size = self.current_state.shape[0]
        buffer_size = 10
        self.delta_buffer = np.zeros((buffer_size, state_size))
        self.position_buffer = np.zeros((buffer_size, state_size))
        for i in range(buffer_size):
            self.position_buffer[i] = self.current_state.copy()

        self.display_window = "Aruco Tracker"
        cv2.namedWindow(self.display_window, cv2.WINDOW_GUI_NORMAL)


    def _on_press(self, key):
        key_val = key.char if hasattr(key, "char") else key
        self.event_queue.put((key_val, True))

    def _on_release(self, key):
        key_val = key.char if hasattr(key, "char") else key
        self.event_queue.put((key_val, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    # The following functions extract the deltas from the images
    
    # Get coordinates of aruco marker in image
    def get_marker_corners(self, frame, markerId):    
        corners, ids, _ = aruco.detectMarkers(frame, self.dictionary, parameters=self.detector_params)

        if ids is None:
            return None
    
        if markerId not in ids:
            return None

        index = np.where(ids == markerId)
        i = index[0][0]
        s_corners = corners[i][0]
    
        return s_corners

    # Estimate the Translation And Rotation vectors from camera
    def estimate_pose(self, frame, markerId):
        marker_corners = self.get_marker_corners(frame, markerId)

        if marker_corners is None:
            return -1, None, None
    
        # Object coordinate system:
        objectPoints = np.array([
            (-self.config.marker_size/2, self.config.marker_size/2, 0),
            (self.config.marker_size/2, self.config.marker_size/2, 0),
            (self.config.marker_size/2, -self.config.marker_size/2, 0),
            (-self.config.marker_size/2, -self.config.marker_size/2, 0)
        ])

        retval, rvec, tvec = cv2.solvePnP(
            objectPoints,
            marker_corners,
            self.camera_matrix,
            self.dist_coeffs
        )

        return retval, rvec, tvec    


    # Translate rotation matrix to angles around X, Y and Z axis
    def rotation_matrix_to_angles(self, rot_mat):
        beta = np.arcsin(-rot_mat[2][0])
        alpha = np.arcsin(
            rot_mat[1][0] / np.cos(beta)
        )
        gamma = np.arcsin(
            rot_mat[2][1] / np.cos(beta)
        )

        return alpha, beta, gamma


    # Get the wrist flex and wrist roll angle the rotation vector
    def wrist_angles(self, rvec, backup_marker=False):
        rot_mat, _ = cv2.Rodrigues(rvec)

        alpha, beta, gamma = self.rotation_matrix_to_angles(rot_mat)
    
        # flex = gamma
        flex = beta
        roll = alpha

        if backup_marker:
            roll = beta
            flex = alpha

        return flex, roll
 
    def get_rotated_vector(self, source, rvec):
        rot_mat, _ = cv2.Rodrigues(rvec)

        rotated_vector = np.matmul(rot_mat, source)

        return rotated_vector

    # Compute current State [x, y, z, flex, roll] based on the frame and camera parameters
    def calculate_state(self, frame):
        retval, rvec, tvec = self.estimate_pose(frame, self.config.main_marker)

        reference_used = self.config.main_marker
        backup_translation = False
        if retval == -1:
            # This indicates the marker was not found, in this case, try the backup marker
            retval, rvec, tvec = self.estimate_pose(frame, self.config.backup_marker)
            backup_translation = True
            reference_used = self.config.backup_marker

        if retval == -1:
            return -1, np.array([0,0,0,0,0]), -1

        position = tvec[:,0]

        # Correct position acording to the plane offset to the point of rotation
        normal = self.get_rotated_vector(np.array([0,0,1]), rvec)
        position = position - (self.config.bk_offset/2)*normal

        x = position[0]
        y = position[1]*(-1)
        z = position[2]*(-1)

        flex, roll = self.wrist_angles(rvec, backup_translation)
    
        if backup_translation:
            flex = flex - np.pi/2

        state = np.array([x, y, z, flex, roll])

        return 1, state, reference_used

    # Update function for deltas, this uses the last N deltas to create a momentum term,
    # Which should increase movement stability
    def smoothed_position_deltas(self, frame, last_position, last_marker_reference, past_deltas, eta = 0.5):
        retval, new_position, new_reference = self.calculate_state(frame)

        delta = np.zeros_like(new_position)
        if new_reference == last_marker_reference and retval != -1:
            delta = new_position - last_position

        n_deltas = past_deltas.shape[0]
        new_delta_vector = np.zeros_like(past_deltas)
        new_delta_vector[1:n_deltas] = past_deltas[0:n_deltas-1]
        new_delta_vector = eta*new_delta_vector

        new_delta_vector[0] = delta

        sum_of_weights = (1 - eta**n_deltas) / (1 - eta)

        weighted_delta = np.sum(new_delta_vector, axis=0) / sum_of_weights

        return weighted_delta, new_position, new_delta_vector, new_reference    

    def delta_from_position_buffer(self, frame, position_buffer, last_marker_reference, eta=0.5):
        retval, new_position, new_reference = self.calculate_state(frame)

        last_position = position_buffer[0].copy()
        if new_reference != last_marker_reference or retval == -1:
            new_position = last_position

        buffer_size = position_buffer.shape[0]
        new_buffer = np.zeros_like(position_buffer)
        new_buffer[1:buffer_size] = position_buffer[0:buffer_size-1]

        new_buffer = eta*new_buffer

        sum_of_weights = ((1 - eta**buffer_size) / (1 - eta))

        weighted_position = (new_position + new_buffer.sum()) / sum_of_weights

        delta = weighted_position - last_position
        new_buffer[0] = weighted_position

        return delta, new_buffer, new_reference

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ArucoTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        delta_pitch = 0.0
        delta_roll = 0.0
        gripper_action = 1  # default gripper action is to stay
        
        frame = self.get_capture_frame()
        aruco_delta, new_position, new_delta_buffer, new_reference_frame = self.smoothed_position_deltas(
            frame, self.current_state, self.current_reference, self.delta_buffer, eta=0.8
        )

        # aruco_delta, new_buffer, new_reference_frame = self.delta_from_position_buffer(frame, self.position_buffer, self.current_reference, eta=0.9)

        corner_points = self.get_marker_corners(frame, self.config.main_marker)
        corner_points_bk = self.get_marker_corners(frame, self.config.backup_marker)

        
        if corner_points is not None:
            corners = corner_points.astype(int)
            for pt in corners:
                frame = cv2.circle(frame, pt, 3, (0,0,255), -1)
        if corner_points_bk is not None:
            corners = corner_points_bk.astype(int)
            for pt in corners:
                frame = cv2.circle(frame, pt, 3, (0,255,0), -1)

        cv2.imshow(self.display_window, frame)
        cv2.waitKey(100)


        # Update the teleoperator state:
        self.current_state = new_position
        self.delta_buffer = new_delta_buffer
        # self.position_buffer = new_buffer
        self.current_reference = new_reference_frame

        # X, Y and Z are rotated in relation to the calculated values
        eps = 1e-3
        delta_x = aruco_delta[2] if abs(aruco_delta[2]) > eps else 0
        delta_y = aruco_delta[0] if abs(aruco_delta[0]) > eps else 0
        delta_z = aruco_delta[1] if abs(aruco_delta[1]) > eps else 0

        # The angles need to be converted from radians to degrees
        delta_pitch = aruco_delta[3]
        delta_roll = aruco_delta[4]

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            # Translations
            # Gripper
            if key == keyboard.Key.up:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.down:
                gripper_action = 0 if val else 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()
        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        if not self.config.position_only:
            action_dict["delta_pitch"] = delta_pitch
            action_dict["delta_roll"] = delta_roll

        return action_dict
