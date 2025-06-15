#!/usr/bin/env python
#
# ik_visual_probe_fixed.py: A single-file simulation for debugging SO101 End-Effector IK
# by running predefined trajectories, plotting the results, and reporting errors.
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

# --- Check and Import Dependencies ---
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# SECTION 1: KINEMATICS (Corrected and verified)
# ==============================================================================
def skew_symmetric(w: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def screw_axis_to_transform(s: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    """Converts a screw axis to a 4x4 transformation matrix."""
    screw_axis_rot = s[:3]
    screw_axis_trans = s[3:]
    transform = np.eye(4)

    if np.allclose(screw_axis_rot, 0) and np.linalg.norm(screw_axis_trans) == 1:
        transform[:3, 3] = screw_axis_trans * theta
    elif np.linalg.norm(screw_axis_rot) == 1:
        w_hat = skew_symmetric(screw_axis_rot)
        rot_mat = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
        t = (np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat @ w_hat) @ screw_axis_trans
        transform[:3, :3] = rot_mat
        transform[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    # THE FIX: This must be outside the if/elif block to always execute.
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
    ROBOT_MEASUREMENTS = {"so_new_calibration": {"gripper": [0.33,0.0,0.285],"wrist": [0.30,0.0,0.267],"forearm": [0.25,0.0,0.266],"humerus": [0.06,0.0,0.264],"shoulder": [0.0,0.0,0.238],"base": [0.0,0.0,0.12]}}
    def __init__(self, robot_type: str = "so_new_calibration"):
        self.measurements = self.ROBOT_MEASUREMENTS.get(robot_type, self.ROBOT_MEASUREMENTS["so_new_calibration"])
        self._setup_transforms()
    def _create_translation_matrix(self, x=0., y=0., z=0.): return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])
    def _setup_transforms(self):
        m=self.measurements; self.gripper_X0=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],dtype=np.float32)
        self.S_BG=np.array([1,0,0,0,m["gripper"][2],-m["gripper"][1]],dtype=np.float32); self.X_GoGt=self._create_translation_matrix(x=0.12); self.X_BoGo=self._create_translation_matrix(x=m["gripper"][0],y=m["gripper"][1],z=m["gripper"][2])
        self.S_BR=np.array([0,1,0,-m["wrist"][2],0,m["wrist"][0]],dtype=np.float32); self.S_BF=np.array([0,1,0,-m["forearm"][2],0,m["forearm"][0]],dtype=np.float32)
        self.S_BH=np.array([0,-1,0,m["humerus"][2],0,-m["humerus"][0]],dtype=np.float32); self.S_BS=np.array([0,0,-1,0,0,0],dtype=np.float32)
        self.X_WoBo=self._create_translation_matrix(x=m["base"][0],y=m["base"][1],z=m["base"][2])
    def forward_kinematics(self, pos_deg: NDArray[np.float32], frame:str="gripper_tip") -> NDArray[np.float32]:
        thetas=np.deg2rad(pos_deg)
        t = self.X_WoBo @ screw_axis_to_transform(self.S_BS, thetas[0])
        t = t @ screw_axis_to_transform(self.S_BH, -thetas[1])
        t = t @ screw_axis_to_transform(self.S_BF, thetas[2])
        t = t @ screw_axis_to_transform(self.S_BR, thetas[3])
        t = t @ screw_axis_to_transform(self.S_BG, thetas[4])
        return t @ self.X_GoGt @ self.X_BoGo @ self.gripper_X0
    def _compute_jacobian_internal(self, pos_deg: NDArray[np.float32], frame: str, positional: bool) -> NDArray[np.float32]:
        eps=1e-5; joints_6d=np.append(pos_deg,0.0) if len(pos_deg)<6 else pos_deg; arm_joints=joints_6d[:-1]; n_j=len(arm_joints); jac=np.zeros((3 if positional else 6,n_j))
        for i in range(n_j):
            delta=np.zeros(n_j); delta[i]=eps/2
            p1=self.forward_kinematics(arm_joints+delta,frame); p2=self.forward_kinematics(arm_joints-delta,frame)
            jac[:,i] = (p1[:3,3] - p2[:3,3])/eps if positional else pose_difference_se3(p1,p2)/eps
        return jac
    def compute_jacobian(self,pos_deg,frame="gripper_tip"): return self._compute_jacobian_internal(pos_deg,frame,False)
    def compute_positional_jacobian(self,pos_deg,frame="gripper_tip"): return self._compute_jacobian_internal(pos_deg,frame,True)
    def ik(self, joint_pos: NDArray[np.float32], desired_pose: NDArray[np.float32], **kwargs) -> NDArray[np.float32]:
        p_only=kwargs.get("position_only",True); frame=kwargs.get("frame","gripper_tip"); n_iter=kwargs.get("max_iterations",10); lr=kwargs.get("learning_rate",0.5); tol=kwargs.get("tolerance",1e-4)
        state_6d=np.append(joint_pos,0.0) if len(joint_pos)<6 else joint_pos.copy()
        for _ in range(n_iter):
            current_pose=self.forward_kinematics(state_6d[:-1],frame); error=desired_pose[:3,3]-current_pose[:3,3] if p_only else se3_error(desired_pose,current_pose)
            if np.linalg.norm(error)<tol: break
            jac=self.compute_positional_jacobian(state_6d,frame) if p_only else self.compute_jacobian(state_6d,frame)
            damp=0.01; jac_pinv=jac.T@np.linalg.inv(jac@jac.T+damp**2*np.eye(jac.shape[0])); state_6d[:-1]+=lr*(jac_pinv@error)
        return state_6d

# ==============================================================================
# SECTION 2: MOCKS AND BASE CLASSES
# ==============================================================================
class MockBaseConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class SO101FollowerEndEffectorConfig(MockBaseConfig): pass
class SO101Follower(object):
    def __init__(self, config: MockBaseConfig): self.config=config; self.bus=MockFeetechMotorsBus(port=config.port)
    def connect(self): self.bus.connect()
    def send_action(self, action:dict[str,Any]): self.bus.sync_write("Goal_Position",{k.removesuffix(".pos"):v for k,v in action.items() if k.endswith(".pos")})
    def disconnect(self): self.bus.disconnect(self.config.disable_torque_on_disconnect)
class MockFeetechMotorsBus:
    def __init__(self, port): self.is_connected=False; self._motor_positions={"shoulder_pan":0.0,"shoulder_lift":20.0,"elbow_flex":90.0,"wrist_flex":-90.0,"wrist_roll":0.0,"gripper":50.0}
    def connect(self): self.is_connected=True
    def disconnect(self, disable_torque=True): self.is_connected=False
    def sync_read(self, register:str): return self._motor_positions.copy() if register=="Present_Position" else {}
    def sync_write(self, register:str, values:dict[str,float]):
        if register=="Goal_Position": [self._motor_positions.update({m:v}) for m,v in values.items()]

# ==============================================================================
# SECTION 3: ROBOT CLASS
# ==============================================================================
EE_FRAME = "gripper_tip"
class SO101FollowerEndEffector(SO101Follower):
    def __init__(self, config: SO101FollowerEndEffectorConfig):
        super().__init__(config)
        self.kinematics=RobotKinematics(robot_type="so_new_calibration")
        self.end_effector_bounds=config.end_effector_bounds
        self.reset()
    def reset(self):
        pos_dict=self.bus.sync_read("Present_Position"); motor_names=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
        self.current_joint_pos=np.array([pos_dict[m] for m in motor_names])
        self.current_ee_pos=self.kinematics.forward_kinematics(self.current_joint_pos[:5], frame=EE_FRAME)
        logging.info("Robot state has been reset to start pose.")
    def _move_to_pose(self, desired_ee_pos:np.ndarray, position_only:bool=False, verbose=False):
        if self.end_effector_bounds: desired_ee_pos[:3,3]=np.clip(desired_ee_pos[:3,3],self.end_effector_bounds["min"],self.end_effector_bounds["max"])
        target_joints_6dof=self.kinematics.ik(np.append(self.current_joint_pos[:5],0.0), desired_ee_pos, position_only=position_only, max_iterations=20, learning_rate=0.4, tolerance=1e-4)
        target_arm_joints=np.clip(target_joints_6dof[:5],-180.0,180.0)
        recalculated_ee_pos=self.kinematics.forward_kinematics(target_arm_joints, frame=EE_FRAME)
        pos_error=np.linalg.norm(desired_ee_pos[:3,3]-recalculated_ee_pos[:3,3])
        rot_error_mat=desired_ee_pos[:3,:3]@recalculated_ee_pos[:3,:3].T
        rot_error_angle=np.rad2deg(np.arccos(np.clip((np.trace(rot_error_mat)-1)/2,-1.0,1.0)))
        self.current_ee_pos=recalculated_ee_pos.copy(); self.current_joint_pos[:5]=target_arm_joints
        motor_names=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
        self.send_action({f"{name}.pos":target_arm_joints[i] for i,name in enumerate(motor_names)})
        if verbose: print(f"  -> Pos Error: {pos_error*1000:.3f} mm, Rot Error: {rot_error_angle:.3f} deg")
        return {"pos_error":pos_error, "rot_error":rot_error_angle}

# ==============================================================================
# SECTION 4: TRAJECTORY GENERATION, TESTING, AND PLOTTING
# ==============================================================================
def generate_linear_trajectory(start_pose, end_pose, num_steps):
    key_rots=Rotation.from_matrix([start_pose[:3,:3],end_pose[:3,:3]]); slerp=Slerp([0,1],key_rots)
    for t in np.linspace(0,1,num_steps):
        next_pose=np.eye(4)
        next_pose[:3,3]=start_pose[:3,3]+t*(end_pose[:3,3]-start_pose[:3,3])
        next_pose[:3,:3]=slerp(t).as_matrix(); yield next_pose

def plot_trajectory_3d(desired_points, actual_points, title):
    if not MATPLOTLIB_AVAILABLE: return
    desired=np.array(desired_points); actual=np.array(actual_points)
    fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection='3d')
    ax.plot(desired[:,0],desired[:,1],desired[:,2],'b--',label='Desired Path',zorder=1)
    ax.plot(actual[:,0],actual[:,1],actual[:,2],'r-',label='Actual Path (from IK)',zorder=2)
    ax.scatter(desired[0,0],desired[0,1],desired[0,2],c='green',s=120,marker='o',label='Start',zorder=3)
    ax.scatter(desired[-1,0],desired[-1,1],desired[-1,2],c='purple',s=120,marker='X',label='End',zorder=3)
    ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
    ax.set_title(title);ax.legend();ax.axis('equal');plt.show()

def run_trajectory_test(robot:SO101FollowerEndEffector, trajectory:Generator, test_name:str):
    print("\n"+"="*80+f"\nSTARTING TEST: {test_name}\n"+"="*80)
    robot.reset(); pos_errors,rot_errors,desired_pts,actual_pts=[],[],[],[]
    for i,target_pose in enumerate(trajectory):
        print(f"Step {i+1}:",end=""); desired_pts.append(target_pose[:3,3])
        errors=robot._move_to_pose(target_pose,position_only=False,verbose=True)
        actual_pts.append(robot.current_ee_pos[:3,3])
        pos_errors.append(errors["pos_error"]*1000); rot_errors.append(errors["rot_error"]); time.sleep(0.01)
    print("\n"+"-"*80+f"\nTRAJECTORY TEST SUMMARY: {test_name}")
    print(f"Position Error (mm): Mean={np.mean(pos_errors):.3f}, Std={np.std(pos_errors):.3f}, Max={np.max(pos_errors):.3f}")
    print(f"Rotation Error (deg): Mean={np.mean(rot_errors):.3f}, Std={np.std(rot_errors):.3f}, Max={np.max(rot_errors):.3f}")
    print("-" * 80); plot_trajectory_3d(desired_pts,actual_pts,f"Trajectory Analysis: {test_name}")

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    if not MATPLOTLIB_AVAILABLE: logging.warning("Matplotlib not found. Plotting will be disabled. 'pip install matplotlib'.")

    robot_config = SO101FollowerEndEffectorConfig(
        port="MOCK_PORT", disable_torque_on_disconnect=True,
        end_effector_bounds={"min":np.array([-0.5,-0.5,0.01]), "max":np.array([0.5,0.5,0.6])},
    )
    robot = SO101FollowerEndEffector(config=robot_config); robot.connect()

    # Test Case 1: Move 15cm Forward
    start_pose = robot.current_ee_pos.copy(); end_pose_1 = start_pose.copy(); end_pose_1[0, 3] += 0.15
    run_trajectory_test(robot, generate_linear_trajectory(start_pose,end_pose_1,50), "Move 15cm Forward (Local X-axis)")

    # Test Case 2: Move 10cm Up with 90-deg Roll
    start_pose=robot.current_ee_pos.copy(); end_pose_2=start_pose.copy(); end_pose_2[2,3]+=0.10
    end_pose_2[:3,:3]=end_pose_2[:3,:3]@Rotation.from_euler('x',90,degrees=True).as_matrix()
    run_trajectory_test(robot, generate_linear_trajectory(start_pose,end_pose_2,50), "Move 10cm Up (Z-axis) with 90-deg Roll")

    # Test Case 3: Trace a 10cm x 10cm Square
    p1=robot.current_ee_pos.copy(); p2,p3,p4=p1.copy(),p1.copy(),p1.copy()
    p2[0,3]+=0.1; p3[0,3]+=0.1; p3[1,3]+=0.1; p4[1,3]+=0.1
    square_traj=(p for seg in [generate_linear_trajectory(p1,p2,25), generate_linear_trajectory(p2,p3,25), generate_linear_trajectory(p3,p4,25), generate_linear_trajectory(p4,p1,25)] for p in seg)
    run_trajectory_test(robot, square_traj, "Trace a 10cm x 10cm Square (World XY Plane)")
    
    robot.disconnect(); logging.info("Simulation finished.")

if __name__ == "__main__":
    main()