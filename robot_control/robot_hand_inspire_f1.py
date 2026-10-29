from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from inspire_f1_sdk.inspire_dds import inspire_hand_f1_ctrl, inspire_hand_f1_state

from hand_retargeting import HandRetargeting, HandType
import numpy as np
import threading
import time
from multiprocessing import Process, Array, Value

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

inspire_tip_indices = [4, 9, 14, 19, 24]
Inspire_Num_Motors = 6

kTopicInspireCtrlLeft = "rt/inspire_hand_f1/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand_f1/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand_f1/state/l"
kTopicInspireStateRight = "rt/inspire_hand_f1/state/r"

# F1 angle register value ranges (units: 0.1 degrees)
# Index 0-3 (little, ring, middle, index fingers)
F1_FINGER_MIN = 900    # closed
F1_FINGER_MAX = 1740   # open
# Index 4 (thumb bend)
F1_THUMB_BEND_MIN = 1100
F1_THUMB_BEND_MAX = 1350
# Index 5 (thumb rotate)
F1_THUMB_ROTATE_MIN = 600
F1_THUMB_ROTATE_MAX = 1800

# Per-DOF ranges for vectorized conversion
_F1_ANGLE_MINS = np.array([F1_FINGER_MIN]*4 + [F1_THUMB_BEND_MIN, F1_THUMB_ROTATE_MIN], dtype=float)
_F1_ANGLE_MAXS = np.array([F1_FINGER_MAX]*4 + [F1_THUMB_BEND_MAX, F1_THUMB_ROTATE_MAX], dtype=float)
_F1_ANGLE_RANGES = _F1_ANGLE_MAXS - _F1_ANGLE_MINS


def _normalized_to_f1_angle(normalized):
    """Convert normalized [0,1] (0=closed, 1=open) to F1 register angle values."""
    return np.clip(_F1_ANGLE_MINS + _F1_ANGLE_RANGES * normalized,
                   _F1_ANGLE_MINS, _F1_ANGLE_MAXS).astype(int).tolist()


def _f1_angle_to_normalized(angle_act):
    """Convert F1 register angle values to normalized [0,1] (0=closed, 1=open)."""
    return np.clip((np.asarray(angle_act, dtype=float) - _F1_ANGLE_MINS) / _F1_ANGLE_RANGES,
                   0.0, 1.0)


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


class Inspire_Controller:
    def __init__(self, left_hand_pos_target, right_hand_pos_target, control_freq=100.0):
        self.control_freq = control_freq
        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_F1)
        self._init_hand_forward_kinematics()

        self.left_hand_q_target  = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_q_target = Array("d", Inspire_Num_Motors, lock=True)
        self.hand_open = Value('b', 0, lock=True)

        self.hand_control_process = Process(
            target=self.control_process,
            args=(left_hand_pos_target, right_hand_pos_target,
                  self.left_hand_q_target, self.right_hand_q_target))
        self.hand_control_process.daemon = True
        self.hand_control_process.start()

        ChannelFactoryInitialize(0)

        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_hand_f1_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_hand_f1_ctrl)
        self.RightHandCmd_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_hand_f1_state)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_hand_f1_state)
        self.RightHandState_subscriber.Init()

        self.left_hand_proprio_buffer = DataBuffer()
        self.right_hand_proprio_buffer = DataBuffer()
        self.left_hand_action_buffer = DataBuffer()
        self.right_hand_action_buffer = DataBuffer()

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        self.publish_hand_command_thread = threading.Thread(target=self._publish_hand_command)
        self.publish_hand_command_thread.daemon = True
        self.publish_hand_command_thread.start()

        self._wait_for_initial_dds_data(timeout_s=5.0)

        logger_mp.info("Initialize Inspire_Controller (F1) OK!\n")

    def _wait_for_initial_dds_data(self, timeout_s=5.0):
        start = time.time()
        has_left_state = False
        has_right_state = False

        while time.time() - start < timeout_s:
            if self.left_hand_proprio_buffer.GetData() is not None:
                has_left_state = True
            if self.right_hand_proprio_buffer.GetData() is not None:
                has_right_state = True
            if has_left_state and has_right_state:
                logger_mp.info("[Inspire_Controller F1] Initial DDS hand state received.")
                return
            time.sleep(0.01)

        if not has_left_state:
            self.left_hand_proprio_buffer.SetData(np.zeros(2 * Inspire_Num_Motors, dtype=float))
        if not has_right_state:
            self.right_hand_proprio_buffer.SetData(np.zeros(2 * Inspire_Num_Motors, dtype=float))
        logger_mp.warning("[Inspire_Controller F1] Timeout waiting for initial DDS data; using defaults.")

    def _init_hand_forward_kinematics(self):
        self.left_hand_wrapper = self.hand_retargeting.left_retargeting.optimizer.robot
        self.right_hand_wrapper = self.hand_retargeting.right_retargeting.optimizer.robot
        self.left_adaptor = self.hand_retargeting.left_retargeting.optimizer.adaptor
        self.right_adaptor = self.hand_retargeting.right_retargeting.optimizer.adaptor
        self.left_target_joint_names = self.hand_retargeting.left_retargeting.optimizer.target_joint_names
        self.right_target_joint_names = self.hand_retargeting.right_retargeting.optimizer.target_joint_names
        # DDS/API proprio order for F1 hand is hardware register order:
        # [little, ring, middle, index, thumb_bend, thumb_rotate].
        # FK target_joint_names comes from YAML and is in retargeting order:
        # [thumb_yaw, thumb_pitch, index, middle, ring, little].
        # Build mapping so FK receives angles in target_joint_names order.
        self.left_hw_to_target_idx = np.array(
            [self.hand_retargeting.left_inspire_api_joint_names.index(name) for name in self.left_target_joint_names],
            dtype=np.int32,
        )
        self.right_hw_to_target_idx = np.array(
            [self.hand_retargeting.right_inspire_api_joint_names.index(name) for name in self.right_target_joint_names],
            dtype=np.int32,
        )
        self.left_hand_link_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
        self.right_hand_link_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
        self.left_hand_link_indices = [self.left_hand_wrapper.get_link_index(n) for n in self.left_hand_link_names]
        self.right_hand_link_indices = [self.right_hand_wrapper.get_link_index(n) for n in self.right_hand_link_names]

    def _build_full_qpos_from_target(self, target_qpos, robot_wrapper, target_joint_names, adaptor):
        full_qpos = np.array(robot_wrapper.q0, dtype=np.float32).copy()
        if target_qpos is None:
            return full_qpos

        q_target = np.array(target_qpos, dtype=np.float32).reshape(-1)
        for i, joint_name in enumerate(target_joint_names):
            if i >= len(q_target):
                break
            if joint_name in robot_wrapper.dof_joint_names:
                joint_idx = robot_wrapper.get_joint_index(joint_name)
                full_qpos[joint_idx] = q_target[i]

        if adaptor is not None:
            full_qpos = adaptor.forward_qpos(full_qpos)
        return full_qpos

    def _compute_hand_forward_kinematics(self):
        left_qpos_raw = self.left_hand_proprio_buffer.GetData()
        right_qpos_raw = self.right_hand_proprio_buffer.GetData()

        if left_qpos_raw is None:
            left_qpos = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        else:
            left_qpos_raw = np.asarray(left_qpos_raw, dtype=np.float32).reshape(-1)
            left_angles_hw = left_qpos_raw[:Inspire_Num_Motors]
            left_qpos = left_angles_hw[self.left_hw_to_target_idx]

        if right_qpos_raw is None:
            right_qpos = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        else:
            right_qpos_raw = np.asarray(right_qpos_raw, dtype=np.float32).reshape(-1)
            right_angles_hw = right_qpos_raw[:Inspire_Num_Motors]
            right_qpos = right_angles_hw[self.right_hw_to_target_idx]

        left_full_qpos = self._build_full_qpos_from_target(
            left_qpos, self.left_hand_wrapper, self.left_target_joint_names, self.left_adaptor)
        right_full_qpos = self._build_full_qpos_from_target(
            right_qpos, self.right_hand_wrapper, self.right_target_joint_names, self.right_adaptor)

        self.left_hand_wrapper.compute_forward_kinematics(left_full_qpos)
        self.right_hand_wrapper.compute_forward_kinematics(right_full_qpos)
        left_link_poses = [self.left_hand_wrapper.get_link_pose(idx) for idx in self.left_hand_link_indices]
        right_link_poses = [self.right_hand_wrapper.get_link_pose(idx) for idx in self.right_hand_link_indices]
        return left_link_poses, right_link_poses

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller F1] Subscribe thread started.")
        while True:
            left_hand_proprio_data = np.zeros(2 * Inspire_Num_Motors)
            right_hand_proprio_data = np.zeros(2 * Inspire_Num_Motors)
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                left_hand_proprio_data[:Inspire_Num_Motors] = _f1_angle_to_normalized(left_state_msg.angle_act)
                left_hand_proprio_data[Inspire_Num_Motors:] = np.array(left_state_msg.force_act) / 1000.0
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                right_hand_proprio_data[:Inspire_Num_Motors] = _f1_angle_to_normalized(right_state_msg.angle_act)
                right_hand_proprio_data[Inspire_Num_Motors:] = np.array(right_state_msg.force_act) / 1000.0
            left_hand_proprio_data[:Inspire_Num_Motors], right_hand_proprio_data[:Inspire_Num_Motors] = \
                self.hardware_to_joint_angles(left_hand_proprio_data[:Inspire_Num_Motors], right_hand_proprio_data[:Inspire_Num_Motors])
            self.left_hand_proprio_buffer.SetData(left_hand_proprio_data)
            self.right_hand_proprio_buffer.SetData(right_hand_proprio_data)
            time.sleep(0.05)

    def _publish_hand_command(self):
        while True:
            with self.left_hand_q_target.get_lock():
                left_q_target = np.array(self.left_hand_q_target[:])
            with self.right_hand_q_target.get_lock():
                right_q_target = np.array(self.right_hand_q_target[:])
            left_f1_cmd = _normalized_to_f1_angle(left_q_target)
            right_f1_cmd = _normalized_to_f1_angle(right_q_target)
            self._send_hand_command(left_f1_cmd, right_f1_cmd)
            time.sleep(0.05)

    def _send_hand_command(self, left_angle_cmd, right_angle_cmd):
        """Send F1 angle commands (in 0.1-degree register units) to both hands.

        Speed and force are initialized by the DDS driver process at startup,
        so we only send angle commands here (matching the DFTP pattern).
        """
        left_cmd_msg = inspire_hand_f1_ctrl(
            pos_set=[0] * 6,
            angle_set=left_angle_cmd,
            force_set=[0] * 6,
            speed_set=[0] * 6,
            mode=0b0001,
        )
        self.LeftHandCmd_publisher.Write(left_cmd_msg)
        right_cmd_msg = inspire_hand_f1_ctrl(
            pos_set=[0] * 6,
            angle_set=right_angle_cmd,
            force_set=[0] * 6,
            speed_set=[0] * 6,
            mode=0b0001,
        )
        self.RightHandCmd_publisher.Write(right_cmd_msg)

    def control_process(self, left_hand_pos_target, right_hand_pos_target,
                        left_hand_q_target, right_hand_q_target):
        logger_mp.info("[Inspire_Controller F1] Control process started.")
        self.running = True

        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()
                with left_hand_pos_target.get_lock():
                    left_hand_data  = np.array(left_hand_pos_target[:]).reshape(25, 3).copy()
                with right_hand_pos_target.get_lock():
                    right_hand_data = np.array(right_hand_pos_target[:]).reshape(25, 3).copy()
                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data == 0.0):
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    left_q_target, right_q_target = self.joint_angles_to_hardware(left_q_target, right_q_target)

                if self.hand_open.value:
                    left_q_target = np.full(Inspire_Num_Motors, 1.0)
                    right_q_target = np.full(Inspire_Num_Motors, 1.0)

                with left_hand_q_target.get_lock():
                    left_hand_q_target[:] = left_q_target
                with right_hand_q_target.get_lock():
                    right_hand_q_target[:] = right_q_target

                time_elapsed = time.time() - start_time
                sleep_time = max(0, (1 / self.control_freq) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller (F1) has been closed.")

    def get_current_hand_proprio(self, thumb_first=True):
        if thumb_first:
            left_hand_proprio = self.left_hand_proprio_buffer.GetData()
            right_hand_proprio = self.right_hand_proprio_buffer.GetData()
            left_hand_qpos = left_hand_proprio[:Inspire_Num_Motors][self.left_hw_to_target_idx]
            right_hand_qpos = right_hand_proprio[:Inspire_Num_Motors][self.right_hw_to_target_idx]
            left_hand_force = left_hand_proprio[Inspire_Num_Motors:][self.left_hw_to_target_idx]
            right_hand_force = right_hand_proprio[Inspire_Num_Motors:][self.right_hw_to_target_idx]
            return {
                "left_hand_proprio": np.concatenate([left_hand_qpos, left_hand_force]),
                "right_hand_proprio": np.concatenate([right_hand_qpos, right_hand_force]),
            }
        else:
            return {
                "left_hand_proprio": self.left_hand_proprio_buffer.GetData(),
                "right_hand_proprio": self.right_hand_proprio_buffer.GetData(),
            }

    def get_current_hand_tip_poses(self):
        left_link_poses, right_link_poses = self._compute_hand_forward_kinematics()
        left_tip_poses = [pose[:3, 3] for pose in left_link_poses]
        right_tip_poses = [pose[:3, 3] for pose in right_link_poses]
        return np.concatenate(right_tip_poses + left_tip_poses)

    def joint_angles_to_hardware(self, left_q, right_q):
        # Retargeting radians → normalized [0, 1] (0=closed, 1=open)
        # F1 URDF conventions: 0 rad = open/extended, max rad = closed/curled
        left_q_target = left_q.copy()
        right_q_target = right_q.copy()
        def normalize(val, min_val, max_val):
            return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

        for idx in range(Inspire_Num_Motors):
            if idx <= 3:
                left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.50)
                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.45)
            elif idx == 4:
                left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.42)
                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.37)
            elif idx == 5:
                left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 2.0944)
                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 2.0944)

        return left_q_target, right_q_target

    def hardware_to_joint_angles(self, left_q, right_q):
        # Normalized [0, 1] → retargeting radians (for FK)
        left_q_target = left_q.copy()
        right_q_target = right_q.copy()
        def denormalize(val, min_val, max_val):
            return max_val - (max_val - min_val) * val

        for idx in range(Inspire_Num_Motors):
            if idx <= 3:
                left_q_target[idx]  = denormalize(left_q_target[idx], 0.0, 1.50)
                right_q_target[idx] = denormalize(right_q_target[idx], 0.0, 1.46)
            elif idx == 4:
                left_q_target[idx]  = denormalize(left_q_target[idx], 0.0, 0.42)
                right_q_target[idx] = denormalize(right_q_target[idx], 0.0, 0.38)
            elif idx == 5:
                left_q_target[idx]  = denormalize(left_q_target[idx], 0.0, 2.0944)
                right_q_target[idx] = denormalize(right_q_target[idx], 0.0, 2.0944)

        return left_q_target, right_q_target
