#!/usr/bin/env python3
"""
G1 Whole Body Controller - handles both arm control and lower body control
Combines arm IK control with policy-based lower body control
"""

import numpy as np
import threading
import time
from enum import IntEnum
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as EstimatorState
from unitree_sdk2py.utils.crc import CRC
from robot_arm_ik_g1 import G1_29_ArmIK
from robot_state_estimator_g1 import G1_29_KinematicStateEstimator

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

import os
import sys
import struct
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
DEFAULT_WBC_CONFIG_PATH = os.path.abspath(os.path.join(
    current_dir, 'g1_wbc_policy', 'g1_student_htd.yaml'
))

# For lower body control
from g1_wbc_policy.rotation_helper import get_gravity_orientation
import yaml

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
kTopicOdomModeState = "rt/odommodestate"
kTopicLFOdomModeState = "rt/lf/odommodestate"

G1_29_Num_Motors = 29

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class IMUState:
    def __init__(self):
        self.quaternion = None
        self.gyroscope = None
        self.accelerometer = None
        self.rpy = None
        self.temperature = None

class OdomState:
    def __init__(self):
        self.position = None
        self.velocity = None
        self.yaw_speed = None
        self.quaternion = None
        self.rpy = None

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]
        self.imu_state = IMUState()

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class RemoteController:
    def __init__(self, lin_vel_x_range, lin_vel_y_range, ang_vel_z_range):
        self.button = [0] * 16
        self.lx = 0
        self.rx = 0
        self.ry = 0
        self.ly = 0
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range
        self.lin_vel_x = 0
        self.lin_vel_y = 0
        self.ang_vel_z = 0

    def set(self, data):
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]

        if self.lin_vel_x_range[0] <= 0 and self.lin_vel_x_range[1] >= 0:
            if self.ly >= 0:
                self.lin_vel_x = np.clip(self.ly * self.lin_vel_x_range[1], 0, self.lin_vel_x_range[1])
            else:
                self.lin_vel_x =np.clip(-self.ly * self.lin_vel_x_range[0], self.lin_vel_x_range[0], 0)
        if self.lin_vel_y_range[0] <= 0 and self.lin_vel_y_range[1] >= 0:
            if self.lx >= 0:
                self.lin_vel_y = np.clip(self.lx * self.lin_vel_y_range[0], self.lin_vel_y_range[0], 0)
            else:
                self.lin_vel_y = np.clip(-self.lx * self.lin_vel_y_range[1], 0, self.lin_vel_y_range[1])
        if self.ang_vel_z_range[0] <= 0 and self.ang_vel_z_range[1] >= 0:
            if self.rx >= 0:
                self.ang_vel_z = np.clip(self.rx * self.ang_vel_z_range[0], self.ang_vel_z_range[0], 0)
            else:
                self.ang_vel_z = np.clip(-self.rx * self.ang_vel_z_range[1], 0, self.ang_vel_z_range[1])


class Config:
    """Simplified config for whole body control"""
    def __init__(self, file_path):
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_control_dt = config["control_dt"]
            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]
            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.policy_path = config["policy_path"]
            legacy_config = "lower_joint2motor_idx" in config
            lower_prefix = "lower_" if legacy_config else ""
            # Upstream paths are relative to deploy/; legacy paths use the YAML directory.
            self.policy_base_dir = os.path.dirname(os.path.abspath(file_path))
            if not legacy_config:
                self.policy_base_dir = os.path.dirname(self.policy_base_dir)

            self.lower_joint2motor_idx = config[f"{lower_prefix}joint2motor_idx"]
            self.arm_joint2motor_idx = config["arm_joint2motor_idx"]
            self.lower_kps = config[f"{lower_prefix}kps"]
            self.lower_kds = config[f"{lower_prefix}kds"]
            self.arm_kps = config["arm_kps"]
            self.arm_kds = config["arm_kds"]
            self.default_lower_joint_pos = np.array(config[f"{lower_prefix}default_joint_pos"], dtype=np.float32)
            self.default_arm_joint_pos = np.array(config["arm_default_joint_pos"], dtype=np.float32)
            self.waist_pitch_limits = np.array(config["waist_pitch_limits"], dtype=np.float32)
            
            if "torso_idx" in config:
                self.torso_idx = config["torso_idx"]
            
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.command_scale = np.array(config["command_scale"], dtype=np.float32)
            
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            
            self.history_length = config["history_length"]
            self.command_range = config["command_range"]


class G1_29_WholeBodyController:
    """Controller for G1 whole body - arms (IK) + lower body (policy)"""
    
    def __init__(self, config_path=None, urdf_path=None, urdf_dir=None):
        logger_mp.info("Initialize G1_29_WholeBodyController...")
        self.arm_ik = G1_29_ArmIK(urdf_path=urdf_path, urdf_dir=urdf_dir)
        self.state_estimator = G1_29_KinematicStateEstimator(urdf_path=urdf_path, urdf_dir=urdf_dir)
        if config_path is None:
            config_path = DEFAULT_WBC_CONFIG_PATH
        
        # Verify config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger_mp.info(f"Loading config from: {config_path}")
        
        # load config
        self.config = Config(config_path)
        # initialize targets
        self.q_arm_target = np.zeros(14)
        self.tauff_arm_target = np.zeros(14)
        self.q_lower_target = np.zeros(15)
        # load lower body control policy
        self.device = torch.device('cpu')
        policy_full_path = os.path.abspath(os.path.join(self.config.policy_base_dir, self.config.policy_path))
        if not os.path.exists(policy_full_path):
            raise FileNotFoundError(
                f"WBC checkpoint not found: {policy_full_path}. "
                "Check policy_path in the controller YAML."
            )
        logger_mp.info(f"Loading policy from: {policy_full_path}")
        self.policy = torch.jit.load(policy_full_path, map_location=self.device).eval()
        logger_mp.info("Policy loaded successfully")
        
        self.lower_joint_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        self.lower_joint_vel = np.zeros(self.config.num_actions, dtype=np.float32)
        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.current_obs = np.zeros(self.config.num_obs, dtype=np.float32)
        self.current_obs_history = np.zeros((self.config.history_length, self.config.num_obs), dtype=np.float32)
        self.torso_rpy_cmd = np.zeros(3, dtype=np.float32)
        self.torso_height_cmd = 0.75  # Default height
        self.arm_velocity_limit = 80.0
        self.control_dt = 1.0 / 500.0
        self._speed_gradual_max = False
        self._gradual_start_time = None
        self.remote_controller = RemoteController(self.config.command_range['lin_vel_x'], self.config.command_range['lin_vel_y'], self.config.command_range['ang_vel_z'])
        # initialize DDS communication
        ChannelFactoryInitialize(0)
        # use debug mode
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.odommodestate_subscriber = ChannelSubscriber(kTopicOdomModeState, EstimatorState)
        self.odommodestate_subscriber.Init()
        self.lf_odommodestate_subscriber = ChannelSubscriber(kTopicLFOdomModeState, EstimatorState)
        self.lf_odommodestate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()
        self.odomstate_buffer = DataBuffer()
        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()
        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_29_WholeBodyController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_WholeBodyController] Subscribe dds ok.")
        # initialize lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()
        size = len(self.msg.motor_cmd)
        for i in range(size):
            self.msg.motor_cmd[i].mode = 1
            self.msg.motor_cmd[i].q = 0
            self.msg.motor_cmd[i].dq = 0
            self.msg.motor_cmd[i].kp = 0
            self.msg.motor_cmd[i].kd = 0
            self.msg.motor_cmd[i].tau = 0
        
        # initialize publish thread
        self.ctrl_lock = threading.Lock()
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        # move to default position and set motor gains
        self.move_to_default_pos()
        
        self.lin_vel_x_cmd = 0.0
        self.lin_vel_y_cmd = 0.0
        self.ang_vel_z_cmd = 0.0
        
        # Warm up policy
        logger_mp.info("Warming up policy...")
        for _ in range(50):
            with torch.inference_mode():
                obs = self.current_obs_history.reshape(1, -1).astype(np.float32)
                self.policy(torch.from_numpy(obs).to(self.device))
        logger_mp.info("Policy warmed up!\n")
        logger_mp.info("Initialize G1_29_WholeBodyController OK!\n")
    
    def _subscribe_motor_state(self):
        while True:
            lowstate_msg = self.lowstate_subscriber.Read()
            if lowstate_msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = lowstate_msg.motor_state[id].q
                    lowstate.motor_state[id].dq = lowstate_msg.motor_state[id].dq
                lowstate.imu_state.quaternion = lowstate_msg.imu_state.quaternion
                lowstate.imu_state.gyroscope = lowstate_msg.imu_state.gyroscope
                lowstate.imu_state.accelerometer = lowstate_msg.imu_state.accelerometer
                lowstate.imu_state.rpy = lowstate_msg.imu_state.rpy
                self.lowstate_buffer.SetData(lowstate)
                self.remote_controller.set(lowstate_msg.wireless_remote)
            
            odommodestate_msg = self.odommodestate_subscriber.Read()
            if odommodestate_msg is not None:
                odomstate = OdomState()
                odomstate.quaternion = odommodestate_msg.imu_state.quaternion
                odomstate.rpy = odommodestate_msg.imu_state.rpy
                odomstate.position = odommodestate_msg.position
                odomstate.velocity = odommodestate_msg.velocity
                odomstate.yaw_speed = odommodestate_msg.yaw_speed
                self.odomstate_buffer.SetData(odomstate)
            time.sleep(0.001)
    
    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target
    
    def _ctrl_motor_state(self):
        """Publishing thread at 500Hz"""
        while True:
            start_time = time.time()
            with self.ctrl_lock:
                arm_q_target = self.q_arm_target
                arm_tauff_target = self.tauff_arm_target
                lower_body_q_target = self.q_lower_target
            # clip arm velocity
            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)
            # set arm commands
            for idx, arm_motor_idx in enumerate(self.config.arm_joint2motor_idx):
                self.msg.motor_cmd[arm_motor_idx].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[arm_motor_idx].dq = 0
                self.msg.motor_cmd[arm_motor_idx].tau = arm_tauff_target[idx]
            # set lower body commands
            for idx, lower_motor_idx in enumerate(self.config.lower_joint2motor_idx):
                self.msg.motor_cmd[lower_motor_idx].q = lower_body_q_target[idx]
                self.msg.motor_cmd[lower_motor_idx].dq = 0
                self.msg.motor_cmd[lower_motor_idx].tau = 0
            # publish command
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            # speed gradual max
            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 40.0 + (40.0 * min(1.0, t_elapsed / 5.0))
            # sleep
            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_whole_body_at_reset(self):
        torso_height = 0.75
        torso_rpy = np.zeros(3)
        self._update_lower_body_state(torso_height, torso_rpy)

    def ctrl_whole_body_from_cmds(self, left_arm_pose, right_arm_pose, torso_height, torso_rpy, velocity_commands=None):
        current_lr_arm_q = self.get_current_dual_arm_q()
        current_lr_arm_dq = self.get_current_dual_arm_dq()
        arm_q_target, arm_tauff_target = self.arm_ik.solve_ik(
            left_arm_pose,
            right_arm_pose,
            current_lr_arm_q,
            current_lr_arm_dq
        )
        with self.ctrl_lock:
            self.q_arm_target = arm_q_target
            self.tauff_arm_target = arm_tauff_target

        self._update_lower_body_state(torso_height, torso_rpy, velocity_commands)
    
    def _update_lower_body_state(self, torso_height_cmd, torso_rpy_cmd, velocity_commands=None):
        """Update lower body state and compute policy action.
        
        Args:
            velocity_commands: optional 3D array [lin_vel_x, lin_vel_y, ang_vel_z].
                When provided (not None), used for locomotion control.
                Joystick always takes priority when operator gives non-zero input.
                When None, falls back to joystick (remote_controller).
        """
        lowstate = self.lowstate_buffer.GetData()
        # Extract current joint positions and velocities
        for i in range(len(self.config.lower_joint2motor_idx)):
            self.lower_joint_pos[i] = lowstate.motor_state[self.config.lower_joint2motor_idx[i]].q
            self.lower_joint_vel[i] = lowstate.motor_state[self.config.lower_joint2motor_idx[i]].dq
        # Read IMU data
        quat = lowstate.imu_state.quaternion
        ang_vel = np.array([lowstate.imu_state.gyroscope], dtype=np.float32)
        
        # Compute gravity orientation
        gravity_orientation = get_gravity_orientation(quat)
        # print(f"gravity_orientation: {gravity_orientation}")
        # The policy observation contract uses positions relative to the default pose.
        lower_joint_pos = (
            self.lower_joint_pos - self.config.default_lower_joint_pos
        ) * self.config.dof_pos_scale
        lower_joint_vel = self.lower_joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        # Joystick always takes priority when operator gives non-zero input
        joy_active = (self.remote_controller.lin_vel_x != 0 or
                      self.remote_controller.lin_vel_y != 0 or
                      self.remote_controller.ang_vel_z != 0)
        if joy_active or velocity_commands is None:
            self.lin_vel_x_cmd = self.remote_controller.lin_vel_x
            self.lin_vel_y_cmd = self.remote_controller.lin_vel_y
            self.ang_vel_z_cmd = self.remote_controller.ang_vel_z
        else:
            self.lin_vel_x_cmd = float(velocity_commands[0])
            self.lin_vel_y_cmd = float(velocity_commands[1])
            self.ang_vel_z_cmd = float(velocity_commands[2])
        command = np.array([
            self.lin_vel_x_cmd,  # lin_vel_x (no translation for teleoperation)
            self.lin_vel_y_cmd,  # lin_vel_y (no translation for teleoperation)
            self.ang_vel_z_cmd,  # ang_vel_z (no rotation for teleoperation)
            torso_height_cmd,  # height
            torso_rpy_cmd[0],  # body_roll
            torso_rpy_cmd[1],  # body_pitch
            torso_rpy_cmd[2],  # body_yaw
        ], dtype=np.float32)
        
        # Clip command to valid range
        clip_min = np.array([
            self.config.command_range["lin_vel_x"][0],
            self.config.command_range["lin_vel_y"][0],
            self.config.command_range["ang_vel_z"][0],
            self.config.command_range["height"][0],
            self.config.command_range["body_roll"][0],
            self.config.command_range["body_pitch"][0],
            self.config.command_range["body_yaw"][0],
        ], dtype=np.float32)
        clip_max = np.array([
            self.config.command_range["lin_vel_x"][1],
            self.config.command_range["lin_vel_y"][1],
            self.config.command_range["ang_vel_z"][1],
            self.config.command_range["height"][1],
            self.config.command_range["body_roll"][1],
            self.config.command_range["body_pitch"][1],
            self.config.command_range["body_yaw"][1],
        ], dtype=np.float32)
        command = np.clip(command, clip_min, clip_max)
        
        # Assemble observation for policy
        # [ang_vel(3), gravity(3), command(7), lower_joint_pos(15), lower_joint_vel(15), action(15)]
        num_actions = self.config.num_actions
        self.current_obs[:3] = ang_vel
        self.current_obs[3:6] = gravity_orientation
        self.current_obs[6:13] = command
        self.current_obs[13:13 + num_actions] = lower_joint_pos
        self.current_obs[13 + num_actions:13 + num_actions * 2] = lower_joint_vel
        self.current_obs[13 + num_actions * 2:13 + num_actions * 3] = self.action
        
        # Update observation history
        self.current_obs_history[:-1] = self.current_obs_history[1:].copy()
        self.current_obs_history[-1] = self.current_obs
        
        # Run policy inference
        obs = self.current_obs_history.reshape(1, -1).astype(np.float32)
        self.action = self.policy(torch.from_numpy(obs).to(self.device).clip(-100, 100)).clip(-100, 100).detach().cpu().numpy().squeeze()
        # Compute target lower body positions
        target_dof_pos = self.config.default_lower_joint_pos + self.action * self.config.action_scale
        waist_pitch_idx = 14  
        target_dof_pos[waist_pitch_idx] = np.clip(
            target_dof_pos[waist_pitch_idx],
            self.config.waist_pitch_limits[0],  # lower limit
            self.config.waist_pitch_limits[1]   # upper limit
        )
        # Update lower body target
        self.q_lower_target = target_dof_pos
    
    def get_mode_machine(self):
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_motor_dq(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointIndex])

    def get_current_dual_arm_q(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])
    
    def get_current_lower_body_q(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[idx].q for idx in self.config.lower_joint2motor_idx])
    
    def get_current_lower_body_dq(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[idx].dq for idx in self.config.lower_joint2motor_idx])

    def get_current_velocity_commands(self):
        return np.array([self.lin_vel_x_cmd, self.lin_vel_y_cmd, self.ang_vel_z_cmd])

    def get_current_estimated_state(self):
        state_dict = self.state_estimator.estimate_state(self.get_current_motor_q())
        torso_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_world_torso"])
        left_end_effector_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_world_left_ee"])
        right_end_effector_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_world_right_ee"])
        pelvis_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_world_pelvis"])
        left_foot_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_world_left_foot"])
        right_foot_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_world_right_foot"])
        left_end_effector_to_torso_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_torso_left_ee"])
        right_end_effector_to_torso_pose = self._convert_T_pose_to_xyz_rpy(state_dict["T_torso_right_ee"])
        return {
            "torso_pose": torso_pose,
            "left_end_effector_pose": left_end_effector_pose,
            "right_end_effector_pose": right_end_effector_pose,
            "pelvis_pose": pelvis_pose,
            "left_foot_pose": left_foot_pose,
            "right_foot_pose": right_foot_pose,
            "left_end_effector_to_torso_pose": left_end_effector_to_torso_pose,
            "right_end_effector_to_torso_pose": right_end_effector_to_torso_pose
        }

    def get_current_imu_state(self):
        imu_state = self.lowstate_buffer.GetData().imu_state
        return {
            "quaternion": np.array(imu_state.quaternion),
            "accelerometer": np.array(imu_state.accelerometer),
            "gyroscope": np.array(imu_state.gyroscope),
            "rpy": np.array(imu_state.rpy)
        }

    def _convert_T_pose_to_xyz_rpy(self, pose):
        xyz = pose[:3, 3]
        rot_mat = pose[:3, :3]
        rpy = self.rot_mat_to_rpy(rot_mat)
        return np.concatenate([xyz, rpy])

    def rot_mat_to_rpy(self, R):
        """
        Convert a rotation matrix to Euler angles (roll, pitch, yaw) with ZYX order.
        This method is more numerically stable, especially near singularities.
        """
        sy = -R[2, 0]
        singular_threshold = 1e-6  # Threshold to check for singularity
        cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        # Handle singularity
        if cy < singular_threshold:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(sy, cy)
            z = 0  # Yaw is indeterminate in this case
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(sy, cy)
            z = np.arctan2(R[1, 0], R[0, 0])

        return np.array([x, y, z])

    def speed_gradual_max(self, t=5.0):
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True
    
    def speed_instant_max(self):
        self.arm_velocity_limit = 80.0
    
    def wait_for_control(self):
        """Wait for user to press 's' to start control"""
        logger_mp.info("Robot in default position.")
        logger_mp.info("Press 's' to start control...")
        while True:
            user_input = input().strip().lower()
            if user_input == 's':
                logger_mp.info("Starting control!")
                break
            else:
                logger_mp.info("Invalid input. Press 's' to start control.")
    
    def move_to_default_pos(self):
        """Move robot to default position gradually and set motor gains"""
        logger_mp.info("Moving to default position...")
        total_time = 3
        num_step = int(total_time / self.config.policy_control_dt)
        
        lower_dof_idx = self.config.lower_joint2motor_idx
        lower_dof_size = len(lower_dof_idx)
        
        arm_dof_idx = self.config.arm_joint2motor_idx
        arm_dof_size = len(arm_dof_idx)
        
        # Get current positions
        current_state = self.lowstate_buffer.GetData()
        lower_init_dof_pos = np.zeros(lower_dof_size, dtype=np.float32)
        arm_init_dof_pos = np.zeros(arm_dof_size, dtype=np.float32)
        for i in range(lower_dof_size):
            lower_init_dof_pos[i] = current_state.motor_state[lower_dof_idx[i]].q
        for i in range(arm_dof_size):
            arm_init_dof_pos[i] = current_state.motor_state[arm_dof_idx[i]].q
        
        # Gradually move to default positions
        for i in range(num_step):
            alpha = i / num_step
            # Update lower body targets
            for j in range(lower_dof_size):
                motor_idx = lower_dof_idx[j]
                target_pos = self.config.default_lower_joint_pos[j]
                interpolated_pos = lower_init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                with self.ctrl_lock:
                    self.q_lower_target[j] = interpolated_pos
                    self.msg.motor_cmd[motor_idx].kp = self.config.lower_kps[j]
                    self.msg.motor_cmd[motor_idx].kd = self.config.lower_kds[j]
            # Update arm targets
            for k in range(arm_dof_size):
                motor_idx = arm_dof_idx[k]
                target_pos = self.config.default_arm_joint_pos[k]
                interpolated_pos = arm_init_dof_pos[k] * (1 - alpha) + target_pos * alpha
                with self.ctrl_lock:
                    self.q_arm_target[k] = interpolated_pos
                    self.msg.motor_cmd[motor_idx].kp = self.config.arm_kps[k]
                    self.msg.motor_cmd[motor_idx].kd = self.config.arm_kds[k]
            time.sleep(self.config.policy_control_dt)
        
        logger_mp.info("Reached default position")
    
    def setup_damping(self):
        lower_dof_idx = self.config.lower_joint2motor_idx
        arm_dof_idx = self.config.arm_joint2motor_idx
        dof_size = len(lower_dof_idx) + len(arm_dof_idx)
        with self.ctrl_lock:
            for i in range(dof_size):
                self.msg.motor_cmd[i].kp = 0
                self.msg.motor_cmd[i].kd = 10
                self.msg.motor_cmd[i].tau = 0
            for j in range(len(lower_dof_idx)):
                self.q_lower_target[j] = 0
            for k in range(len(arm_dof_idx)):
                self.q_arm_target[k] = 0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21
    
    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5
    
    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11
    
    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14
    
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21
    
    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
