"""
Custom DDS message types for teleoperation
Using cyclonedds.idl to define messages
"""

from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, int16


@dataclass
class Float64Array(IdlStruct):
    """DDS message type for float64 arrays - similar to ROS Float64MultiArray"""
    data: sequence[float]


@dataclass  
class ArmPose(IdlStruct):
    """4x4 transformation matrix as 16 float values"""
    data: sequence[float]  # Should be length 16


@dataclass
class HandPose(IdlStruct):
    """75 DOF hand pose"""
    data: sequence[float]  # Should be length 75


@dataclass
class TeleopData(IdlStruct):
    """Synchronized teleoperation data - all poses in one message"""
    timestamp: float  # Timestamp in seconds
    left_arm_pose: sequence[float]   # 16 floats (4x4 matrix)
    right_arm_pose: sequence[float]  # 16 floats (4x4 matrix)
    left_hand_pose: sequence[float]  # 75 floats (hand DOF)
    right_hand_pose: sequence[float] # 75 floats (hand DOF)


@dataclass
class TeleopDataWB(IdlStruct):
    """Synchronized whole-body teleoperation data - includes torso commands"""
    timestamp: float  # Timestamp in seconds
    left_arm_pose: sequence[float]   # 16 floats (4x4 matrix)
    right_arm_pose: sequence[float]  # 16 floats (4x4 matrix)
    left_hand_pose: sequence[float]  # 75 floats (hand DOF)
    right_hand_pose: sequence[float] # 75 floats (hand DOF)
    torso_rpy: sequence[float]       # 3 floats (roll, pitch, yaw)
    torso_height: float              # Torso height in meters
    velocity_commands: sequence[float]  # 3 floats (lin_vel_x, lin_vel_y, ang_vel_z)


@dataclass
class ResetCommand(IdlStruct):
    """Reset command for robot - triggers go_to_reset()"""
    timestamp: float  # Timestamp in seconds
    reset_requested: bool  # True to request reset, False otherwise


@dataclass
class EEFState(IdlStruct):
    """End-effector hand state data."""
    timestamp: float  # Timestamp in seconds
    joint_positions: sequence[float]  # Joint positions
    finger_forces: sequence[float]  # Finger forces
    hand_link_poses: sequence[float]  # Hand link poses

@dataclass
class G1RobotState(IdlStruct):
    """G1 robot state data"""
    timestamp: float
    # global poses in world frame (defined by feet flat pose for manipulation, assuming both feet are flat on ground)
    torso_pose: sequence[float]  # xyz+rpy
    left_end_effector_pose: sequence[float]  # xyz+rpy
    right_end_effector_pose: sequence[float]  # xyz+rpy
    pelvis_pose: sequence[float]  # xyz+rpy
    left_foot_pose: sequence[float]  # xyz+rpy
    right_foot_pose: sequence[float]  # xyz+rpy
    left_end_effector_to_torso_pose: sequence[float]  # xyz+rpy
    right_end_effector_to_torso_pose: sequence[float]  # xyz+rpy
    joint_positions: sequence[float]  # 29 joints
    joint_velocities: sequence[float]  # 29 joints
    velocity_commands: sequence[float]  # 3 floats (lin_vel_x, lin_vel_y, ang_vel_z)
    # imu state (rpy's reference frame here is defined by unitree's state estimator)
    imu_quaternion: sequence[float]  
    imu_accelerometer: sequence[float]  
    imu_gyroscope: sequence[float]
    imu_rpy: sequence[float]

@dataclass
class InspireTouchState(IdlStruct):
    timestamp: float
    fingerone_tip_touch_l: sequence[int16, 9]
    fingerone_top_touch_l: sequence[int16, 96]
    fingerone_palm_touch_l: sequence[int16, 80]
    fingertwo_tip_touch_l: sequence[int16, 9]
    fingertwo_top_touch_l: sequence[int16, 96]
    fingertwo_palm_touch_l: sequence[int16, 80]
    fingerthree_tip_touch_l: sequence[int16, 9]
    fingerthree_top_touch_l: sequence[int16, 96]
    fingerthree_palm_touch_l: sequence[int16, 80]
    fingerfour_tip_touch_l: sequence[int16, 9]
    fingerfour_top_touch_l: sequence[int16, 96]
    fingerfour_palm_touch_l: sequence[int16, 80]
    fingerfive_tip_touch_l: sequence[int16, 9]
    fingerfive_top_touch_l: sequence[int16, 96]
    fingerfive_middle_touch_l: sequence[int16, 9]
    fingerfive_palm_touch_l: sequence[int16, 96]
    palm_touch_l: sequence[int16, 112]
    fingerone_tip_touch_r: sequence[int16, 9]
    fingerone_top_touch_r: sequence[int16, 96]
    fingerone_palm_touch_r: sequence[int16, 80]
    fingertwo_tip_touch_r: sequence[int16, 9]
    fingertwo_top_touch_r: sequence[int16, 96]
    fingertwo_palm_touch_r: sequence[int16, 80]
    fingerthree_tip_touch_r: sequence[int16, 9]
    fingerthree_top_touch_r: sequence[int16, 96]
    fingerthree_palm_touch_r: sequence[int16, 80]
    fingerfour_tip_touch_r: sequence[int16, 9]
    fingerfour_top_touch_r: sequence[int16, 96]
    fingerfour_palm_touch_r: sequence[int16, 80]
    fingerfive_tip_touch_r: sequence[int16, 9]
    fingerfive_top_touch_r: sequence[int16, 96]
    fingerfive_middle_touch_r: sequence[int16, 9]
    fingerfive_palm_touch_r: sequence[int16, 96]
    palm_touch_r: sequence[int16, 112]
