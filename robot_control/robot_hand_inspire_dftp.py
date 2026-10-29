from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

from hand_retargeting import HandRetargeting, HandType
import numpy as np
import threading
import time
from multiprocessing import Process, Array, Value

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

inspire_tip_indices = [4, 9, 14, 19, 24] # Assuming this remains relevant for hand_retargeting
Inspire_Num_Motors = 6 # Number of motors per hand

# NEW DDS TOPIC NAMES (assuming these are the correct topics based on SDK examples)
kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"
# kTopicInspireTouchLeft = "rt/inspire_hand/touch/l"
# kTopicInspireTouchRight = "rt/inspire_hand/touch/r"

# # Touch field names in inspire_hand_touch message
# TOUCH_FIELDS = [
#     "fingerone_tip_touch",
#     "fingerone_top_touch",
#     "fingerone_palm_touch",
#     "fingertwo_tip_touch",
#     "fingertwo_top_touch",
#     "fingertwo_palm_touch",
#     "fingerthree_tip_touch",
#     "fingerthree_top_touch",
#     "fingerthree_palm_touch",
#     "fingerfour_tip_touch",
#     "fingerfour_top_touch",
#     "fingerfour_palm_touch",
#     "fingerfive_tip_touch",
#     "fingerfive_top_touch",
#     "fingerfive_middle_touch",
#     "fingerfive_palm_touch",
#     "palm_touch",
# ]

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
    def __init__(self, left_hand_pos_target, right_hand_pos_target, control_freq = 100.0):
        # left_hand_pos_target, right_hand_pos_target are shared arrays for child process for hand control loop
        self.control_freq = control_freq
        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_DFTP)
        self._init_hand_forward_kinematics()

        self.left_hand_q_target  = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_q_target = Array("d", Inspire_Num_Motors, lock=True)
        self.hand_open = Value('b', 0, lock=True)

        self.hand_control_process = Process(target=self.control_process, args=(left_hand_pos_target, right_hand_pos_target, self.left_hand_q_target, self.right_hand_q_target))
        self.hand_control_process.daemon = True
        self.hand_control_process.start()

        ChannelFactoryInitialize(0)

        # Initialize hand command publishers
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()
        # self.LeftHandTouch_subscriber = ChannelSubscriber(kTopicInspireTouchLeft, inspire_dds.inspire_hand_touch)
        # self.LeftHandTouch_subscriber.Init()
        # self.RightHandTouch_subscriber = ChannelSubscriber(kTopicInspireTouchRight, inspire_dds.inspire_hand_touch)
        # self.RightHandTouch_subscriber.Init()

        # initialize data buffers for hand proprio states and actions
        self.left_hand_proprio_buffer = DataBuffer()
        self.right_hand_proprio_buffer = DataBuffer()
        # self.left_hand_touch_buffer = DataBuffer()
        # self.right_hand_touch_buffer = DataBuffer()
        self.left_hand_action_buffer = DataBuffer()
        self.right_hand_action_buffer = DataBuffer()

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        # self.subscribe_touch_thread = threading.Thread(target=self._subscribe_hand_touch)
        # self.subscribe_touch_thread.daemon = True
        # self.subscribe_touch_thread.start()
        self.publish_hand_command_thread = threading.Thread(target=self._publish_hand_command)
        self.publish_hand_command_thread.daemon = True
        self.publish_hand_command_thread.start()

        # Ensure DDS state/touch buffers are initialized before external usage.
        self._wait_for_initial_dds_data(timeout_s=5.0)


        logger_mp.info("Initialize Inspire_Controller OK!\n")

    # def _empty_touch_dict(self):
    #     touch = {}
    #     for name in TOUCH_FIELDS:
    #         # expected lengths are defined in inspire_hand_defaut.get_inspire_hand_touch()
    #         if name.endswith("_tip_touch") or name.endswith("_middle_touch"):
    #             size = 9
    #         elif name.endswith("_top_touch") or name.endswith("_palm_touch"):
    #             # finger*_top/palm are 96/80; palm_touch is 112
    #             if name == "palm_touch":
    #                 size = 112
    #             elif "_top_" in name:
    #                 size = 96
    #             else:
    #                 size = 80
    #         else:
    #             size = 0
    #         touch[name] = np.zeros(size, dtype=np.int16)
    #     return touch

    def _wait_for_initial_dds_data(self, timeout_s=5.0):
        start = time.time()
        has_left_state = False
        has_right_state = False
        # has_left_touch = False
        # has_right_touch = False

        while time.time() - start < timeout_s:
            if self.left_hand_proprio_buffer.GetData() is not None:
                has_left_state = True
            if self.right_hand_proprio_buffer.GetData() is not None:
                has_right_state = True
            # if self.left_hand_touch_buffer.GetData() is not None:
            #     has_left_touch = True
            # if self.right_hand_touch_buffer.GetData() is not None:
            #     has_right_touch = True

            if has_left_state and has_right_state:
                logger_mp.info("[Inspire_Controller] Initial DDS hand state received.")
                return
            time.sleep(0.01)

        # Fallback initialization to avoid None usage downstream.
        if not has_left_state:
            self.left_hand_proprio_buffer.SetData(np.zeros(Inspire_Num_Motors, dtype=float))
        if not has_right_state:
            self.right_hand_proprio_buffer.SetData(np.zeros(Inspire_Num_Motors, dtype=float))
        # if not has_left_touch:
        #     self.left_hand_touch_buffer.SetData(self._empty_touch_dict())
        # if not has_right_touch:
        #     self.right_hand_touch_buffer.SetData(self._empty_touch_dict())
        logger_mp.warning("[Inspire_Controller] Timeout waiting for initial DDS data; using default initialized buffers.")

    def _init_hand_forward_kinematics(self):
        self.left_hand_wrapper = self.hand_retargeting.left_retargeting.optimizer.robot
        self.right_hand_wrapper = self.hand_retargeting.right_retargeting.optimizer.robot
        self.left_adaptor = self.hand_retargeting.left_retargeting.optimizer.adaptor
        self.right_adaptor = self.hand_retargeting.right_retargeting.optimizer.adaptor
        self.left_target_joint_names = self.hand_retargeting.left_retargeting.optimizer.target_joint_names
        self.right_target_joint_names = self.hand_retargeting.right_retargeting.optimizer.target_joint_names
        # DDS/API proprio order for Inspire hand is hardware order:
        # [pinky, ring, middle, index, thumb_pitch, thumb_yaw].
        # FK target_joint_names comes from YAML and is in retargeting order:
        # [thumb_yaw, thumb_pitch, index, middle, ring, pinky].
        # Build mapping so FK receives angles in target_joint_names order.
        self.left_hw_to_target_idx = np.array(
            [self.hand_retargeting.left_inspire_api_joint_names.index(name) for name in self.left_target_joint_names],
            dtype=np.int32,
        )
        self.right_hw_to_target_idx = np.array(
            [self.hand_retargeting.right_inspire_api_joint_names.index(name) for name in self.right_target_joint_names],
            dtype=np.int32,
        )
        self.left_hand_link_names = [ "L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip" ]
        self.right_hand_link_names = [ "R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip" ]
        self.left_hand_link_indices = [ self.left_hand_wrapper.get_link_index(link_name) for link_name in self.left_hand_link_names ]
        self.right_hand_link_indices = [ self.right_hand_wrapper.get_link_index(link_name) for link_name in self.right_hand_link_names ]

    def _build_full_qpos_from_target(self, target_qpos, robot_wrapper, target_joint_names, adaptor):
        # Build full pinocchio-order qpos from target joints and then apply kinematic adaptor
        # (e.g. mimic joints) so forward kinematics receives a complete valid vector.
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
            left_qpos,
            self.left_hand_wrapper,
            self.left_target_joint_names,
            self.left_adaptor,
        )
        right_full_qpos = self._build_full_qpos_from_target(
            right_qpos,
            self.right_hand_wrapper,
            self.right_target_joint_names,
            self.right_adaptor,
        )

        self.left_hand_wrapper.compute_forward_kinematics(left_full_qpos)
        self.right_hand_wrapper.compute_forward_kinematics(right_full_qpos)
        left_link_poses = [ self.left_hand_wrapper.get_link_pose(link_idx) for link_idx in self.left_hand_link_indices ]
        right_link_poses = [ self.right_hand_wrapper.get_link_pose(link_idx) for link_idx in self.right_hand_link_indices ]
        return left_link_poses, right_link_poses

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller] Subscribe thread started.")
        # left_hand_proprio_data: [angle_act, force_act]
        # right_hand_proprio_data: [angle_act, force_act]
        while True:
            left_hand_proprio_data = np.zeros(2*Inspire_Num_Motors)
            right_hand_proprio_data = np.zeros(2*Inspire_Num_Motors)
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                left_hand_proprio_data[:Inspire_Num_Motors] = np.array(left_state_msg.angle_act) / 1000.0
                # the original unit is g, convert to kg
                left_hand_proprio_data[Inspire_Num_Motors:] = np.array(left_state_msg.force_act) / 1000.0
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                right_hand_proprio_data[:Inspire_Num_Motors] = np.array(right_state_msg.angle_act) / 1000.0
                # the original unit is g, convert to kg
                right_hand_proprio_data[Inspire_Num_Motors:] = np.array(right_state_msg.force_act) / 1000.0
            left_hand_proprio_data[:Inspire_Num_Motors], right_hand_proprio_data[:Inspire_Num_Motors] = self.hardware_to_joint_angles(left_hand_proprio_data[:Inspire_Num_Motors], right_hand_proprio_data[:Inspire_Num_Motors])
            self.left_hand_proprio_buffer.SetData(left_hand_proprio_data)
            self.right_hand_proprio_buffer.SetData(right_hand_proprio_data)
            time.sleep(0.001)

    # def _subscribe_hand_touch(self):
    #     while True:
    #         left_touch_msg = self.LeftHandTouch_subscriber.Read()
    #         if left_touch_msg is not None:
    #             self.left_hand_touch_buffer.SetData(self._touch_msg_to_dict(left_touch_msg))
    #             print(f"Left touch: {self.left_hand_touch_buffer.GetData()}")
    #         right_touch_msg = self.RightHandTouch_subscriber.Read()
    #         if right_touch_msg is not None:
    #             self.right_hand_touch_buffer.SetData(self._touch_msg_to_dict(right_touch_msg))
    #         time.sleep(0.001)

    # def _touch_msg_to_dict(self, msg):
    #     touch_dict = {}
    #     for name in TOUCH_FIELDS:
    #         values = getattr(msg, name, None)
    #         if values is None:
    #             touch_dict[name] = None
    #         else:
    #             touch_dict[name] = np.asarray(values, dtype=np.int16)
    #     return touch_dict

    def _publish_hand_command(self):
        while True:
            with self.left_hand_q_target.get_lock():
                left_q_target = np.array(self.left_hand_q_target[:])
            with self.right_hand_q_target.get_lock():
                right_q_target = np.array(self.right_hand_q_target[:])
            scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target]
            scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target]
            self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
            time.sleep(0.001)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        # left hand command
        left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001 # mode 1: angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)
        # right hand command
        right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001 # mode 1: angle control
        self.RightHandCmd_publisher.Write(right_cmd_msg)

    def control_process(self, left_hand_pos_target, right_hand_pos_target, left_hand_q_target, right_hand_q_target):
        logger_mp.info("[Inspire_Controller] Control process started.")
        self.running = True

        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_pos_target.get_lock():
                    left_hand_data  = np.array(left_hand_pos_target[:]).reshape(25, 3).copy()
                with right_hand_pos_target.get_lock():
                    right_hand_data = np.array(right_hand_pos_target[:]).reshape(25, 3).copy()
                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data == 0.0): # if hand data has been initialized.
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

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.control_freq) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")

    # def get_current_touch_state(self):
    #     return {
    #         "left_hand_touch": self.left_hand_touch_buffer.GetData(),
    #         "right_hand_touch": self.right_hand_touch_buffer.GetData(),
    #     }

    def get_current_hand_proprio(self, thumb_first = True):
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
        # extract 3d positions of hand tips from current hand link poses
        left_link_poses, right_link_poses = self._compute_hand_forward_kinematics()
        left_tip_poses = [pose[:3, 3] for pose in left_link_poses]
        right_tip_poses = [pose[:3, 3] for pose in right_link_poses]
        return np.concatenate(right_tip_poses + left_tip_poses)

    def joint_angles_to_hardware(self, left_q, right_q):
        # left_hand_q_target/right_hand_q_target ranges: [0, 1] ==> 0.0: fully closed  1.0: fully open
        # raw left_q_target/right_q_target from hand_retargeting is in radians, ranges:
        #     - idx 0~3 (pinky, ring, middle, index): 0~1.7 (1.7 = closed)
        #     - idx 4 (thumb bend):   0~0.5
        #     - idx 5 (thumb rotation):  -0.1~1.3
        left_q_target = left_q
        right_q_target = right_q
        def normalize(val, min_val, max_val):
            return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

        for idx in range(Inspire_Num_Motors):
            if idx <= 3:
                # tweaks for our DFTP hand
                left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.435)
                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.435)
            elif idx == 4:
                # tweaks for our DFTP hand
                left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.350)
                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.350)
            elif idx == 5:
                left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.20)

        return left_q_target, right_q_target

    def hardware_to_joint_angles(self, left_q, right_q):
        # left_hand_q_target/right_hand_q_target ranges: [0, 1] ==> 0.0: fully closed  1.0: fully open
        # raw left_q_target/right_q_target from hand_retargeting is in radians, ranges:
        #     - idx 0~3 (pinky, ring, middle, index): 0~1.7 (1.7 = closed)
        #     - idx 4 (thumb bend):   0~0.5
        #     - idx 5 (thumb rotation):  -0.1~1.3
        left_q_target = left_q
        right_q_target = right_q
        def denormalize(val, min_val, max_val):
            return max_val - (max_val - min_val) * val

        for idx in range(Inspire_Num_Motors):
            if idx <= 3:
                # tweaks for our DFTP hand
                left_q_target[idx]  = denormalize(left_q_target[idx], 0.0, 1.435)
                right_q_target[idx] = denormalize(right_q_target[idx], 0.0, 1.435)
            elif idx == 4:
                # tweaks for our DFTP hand
                left_q_target[idx]  = denormalize(left_q_target[idx], 0.0, 0.350)
                right_q_target[idx] = denormalize(right_q_target[idx], 0.0, 0.350)
            elif idx == 5:
                left_q_target[idx]  = denormalize(left_q_target[idx], -0.1, 1.3)
                right_q_target[idx] = denormalize(right_q_target[idx], -0.1, 1.20)

        return left_q_target, right_q_target
