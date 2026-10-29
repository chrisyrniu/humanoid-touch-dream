#!/usr/bin/env python3
"""
DDS-based Whole Body Controller for G1 with Inspire Hands
This controller receives processed Vision Pro data (including torso commands) via DDS, 
performs IK solving for arms, policy inference for lower body, and hand retargeting,
then controls the robot directly.
"""

import numpy as np
import time
import argparse
import threading
from multiprocessing import Array
import logging_mp
import os
import sys
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
Inspire_Num_Motors = 6

from g1_wholebody import G1_29_WholeBodyController, DEFAULT_WBC_CONFIG_PATH

class G1InspireWBController:
    """DDS-based whole body controller for G1 robot with Inspire hands"""
    
    def __init__(self, args):
        self.args = args
        self.running = True
        self.control_freq = args.control_freq
        self.publish_freq = args.publish_freq
        
        # Data buffers for received teleoperation data
        self.teleop_data_lock = threading.Lock()
        self.left_arm_pose = np.eye(4)  # 4x4 transformation matrix
        self.right_arm_pose = np.eye(4)
        self.left_hand_pos = np.zeros(75)  # 75DOF hand pose
        self.right_hand_pos = np.zeros(75)
        self.torso_rpy = np.zeros(3)  # Torso roll, pitch, yaw
        self.torso_height = 0.75  # Default torso height
        self.velocity_commands = np.zeros(3)  # lin_vel_x, lin_vel_y, ang_vel_z
        self.has_velocity_commands = False  # True when policy sends explicit velocity
        self.teleop_data_received = False
        self.last_timestamp = 0.0  # Track last received timestamp
        
        # Reset state
        self.resetting = False
        self.reset_start_time = 0.0
        self.reset_start_height = 0.75
        self.reset_start_rpy = np.zeros(3)
        self.reset_start_arm_q = np.zeros(14)
        
        # Initialize whole body controller (arms + lower body)
        config_path = DEFAULT_WBC_CONFIG_PATH
        g1_type = getattr(self.args, 'g1_type', 0)
        urdf_map = {
            0: 'g1_body29_hand14_wrist4010.urdf',
            1: 'g1_body29_hand14_wrist5010.urdf',
        }
        self.g1_urdf_path = os.path.join(current_dir, 'assets', 'g1', urdf_map[g1_type])
        self.g1_urdf_dir = os.path.join(current_dir, 'assets', 'g1') + '/'
        self._init_whole_body_controller(config_path)
        # Initialize hand controller
        self._init_hand_controller()
        # Initialize DDS communication
        self._init_dds_input()
        # Initialize DDS output
        self._init_dds_output()
        # Start DDS subscription threads
        self.teleop_cmd_sub_thread = threading.Thread(target=self._dds_cmd_subscription_loop, daemon=True)
        self.teleop_cmd_sub_thread.start()
        logger_mp.info("DDS teleop cmd subscription thread started")
        self.reset_cmd_sub_thread = threading.Thread(target=self._reset_cmd_subscription_loop, daemon=True)
        self.reset_cmd_sub_thread.start()
        logger_mp.info("DDS reset cmd subscription thread started")
        logger_mp.info(f"G1 Inspire WB Controller initialized with control frequency: {self.control_freq} Hz")
        logger_mp.info(f"Hand type: {getattr(args, 'hand_type', 'dftp')}")
        # robot state publisher
        self.robot_state_pub_thread = threading.Thread(target=self._robot_state_pub_loop, daemon=True)
        self.robot_state_pub_thread.start()
        logger_mp.info("DDS robot state publisher thread started")
        # hand state publisher
        self.hand_state_pub_thread = threading.Thread(target=self._hand_state_pub_loop, daemon=True)
        self.hand_state_pub_thread.start()
        logger_mp.info("DDS hand state publisher thread started")
    
    def _init_dds_input(self):
        """Initialize DDS input communication using cyclonedds"""
        logger_mp.info("Initializing DDS input communication...")
        
        # DDS imports
        from teleop_dds_msgs import TeleopDataWB, ResetCommand
        
        # Store for use in subscription loop
        self.DataReader = DataReader
        self.TeleopDataWB = TeleopDataWB
        
        # Create DDS domain participant (domain 0) for cyclonedds
        self.participant = DomainParticipant(0)
        
        # Create synchronized topic for whole-body teleoperation
        kTopicTeleopData = "rt/teleop/synchronized_data"
        self.topic_teleop = Topic(self.participant, kTopicTeleopData, TeleopDataWB)
        self.reader_teleop = DataReader(self.participant, self.topic_teleop)
        
        # Create reset command topic
        kTopicResetCommand = "rt/robot/reset_command"
        self.topic_reset = Topic(self.participant, kTopicResetCommand, ResetCommand)
        self.reader_reset = DataReader(self.participant, self.topic_reset)
        
        logger_mp.info("DDS input initialized")
        logger_mp.info(f"Listening on DDS topic: {kTopicTeleopData}")
        logger_mp.info(f"Listening on DDS topic: {kTopicResetCommand}")

    def _init_dds_output(self):
        """Initialize DDS output communication for publishing robot state"""
        logger_mp.info("Initializing DDS output communication...")
        # DDS imports
        from teleop_dds_msgs import G1RobotState
        # Create robot state topic
        kTopicRobotState = "rt/robot/state"
        self.topic_robot_state = Topic(self.participant, kTopicRobotState, G1RobotState)
        self.writer_robot_state = DataWriter(self.participant, self.topic_robot_state)

        from teleop_dds_msgs import EEFState
        kTopicEEFState = "rt/robot/eef_state"
        self.topic_eef_state = Topic(self.participant, kTopicEEFState, EEFState)
        self.writer_eef_state = DataWriter(self.participant, self.topic_eef_state)

        from teleop_dds_msgs import ResetCommand
        kTopicControllerResetNotify = "rt/controller/reset_notify"
        self.topic_controller_reset = Topic(self.participant, kTopicControllerResetNotify, ResetCommand)
        self.writer_controller_reset = DataWriter(self.participant, self.topic_controller_reset)
        logger_mp.info(f"Controller reset notify publisher on topic: {kTopicControllerResetNotify}")

    def _init_whole_body_controller(self, config_path):
        """Initialize whole body controller (arms + lower body)"""
        logger_mp.info("Initializing G1 Whole Body Controller...")
        
        # Verify config path
        if not os.path.exists(config_path):
            logger_mp.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger_mp.info(f"Using config: {config_path}")
        
        self.wb_ctrl = G1_29_WholeBodyController(
            config_path=config_path,
            urdf_path=self.g1_urdf_path,
            urdf_dir=self.g1_urdf_dir
        )
        logger_mp.info("Whole body controller initialized successfully")
    
    def _init_hand_controller(self):
        """Initialize hand controller"""
        hand_type = getattr(self.args, 'hand_type', 'dftp')
        if hand_type == 'f1':
            from robot_hand_inspire_f1 import Inspire_Controller
        else:
            from robot_hand_inspire_dftp import Inspire_Controller
        
        logger_mp.info("Initializing Inspire Hand Controller...")
        self.left_hand_pos_target = Array('d', 75, lock=True)   # [input]
        self.right_hand_pos_target = Array('d', 75, lock=True)  # [input]
        
        self.hand_ctrl = Inspire_Controller(
            left_hand_pos_target=self.left_hand_pos_target,
            right_hand_pos_target=self.right_hand_pos_target,
            control_freq=self.control_freq
        )
        logger_mp.info("Hand controller initialized successfully")
    
    def _dds_cmd_subscription_loop(self):
        """DDS subscription loop running in separate thread"""
        logger_mp.info("DDS subscription loop started")

        while self.running:
            try:
                # Read synchronized data from single topic
                samples = self.reader_teleop.take(N=1)
                for sample in samples:
                    if sample:
                        # Verify data lengths (TeleopDataWB includes torso data)
                        if (len(sample.left_arm_pose) == 16 and
                            len(sample.right_arm_pose) == 16 and
                            len(sample.left_hand_pose) == 75 and
                            len(sample.right_hand_pose) == 75 and
                            len(sample.torso_rpy) == 3):

                            with self.teleop_data_lock:
                                # All data from the same timestamp - guaranteed synchronized
                                self.left_arm_pose = np.array(sample.left_arm_pose).reshape(4, 4)
                                self.right_arm_pose = np.array(sample.right_arm_pose).reshape(4, 4)
                                self.left_hand_pos = np.array(sample.left_hand_pose)
                                self.right_hand_pos = np.array(sample.right_hand_pose)

                                # Extract torso data (from TeleopDataWB)
                                self.torso_rpy = np.array(sample.torso_rpy)
                                self.torso_height = float(sample.torso_height)

                                # Extract velocity commands: len==3 means policy is controlling velocity
                                if hasattr(sample, 'velocity_commands') and len(sample.velocity_commands) == 3:
                                    self.velocity_commands = np.array(sample.velocity_commands)
                                    self.has_velocity_commands = True
                                else:
                                    self.has_velocity_commands = False

                                self.last_timestamp = sample.timestamp
                                self.teleop_data_received = True

                            # Clear hand open override now that fresh data is arriving,
                            # but NOT while a reset is in progress (avoid race with _open_hands)
                            if self.hand_ctrl.hand_open.value and not self.resetting:
                                self.hand_ctrl.hand_open.value = 0
                            # Update hand controller arrays
                            with self.left_hand_pos_target.get_lock():
                                self.left_hand_pos_target[:] = self.left_hand_pos
                            with self.right_hand_pos_target.get_lock():
                                self.right_hand_pos_target[:] = self.right_hand_pos

                time.sleep(0.001)  # Small sleep to avoid busy loop

            except Exception as e:
                logger_mp.error(f"Error in DDS subscription loop: {e}")
                time.sleep(0.1)

        logger_mp.info("DDS subscription loop ended")

    def _reset_cmd_subscription_loop(self):
        """Listen for reset commands and trigger smooth reset."""
        logger_mp.info("Reset command subscription loop started")
        while self.running:
            try:
                samples = self.reader_reset.take(N=1)
                for sample in samples:
                    if sample and sample.reset_requested and not self.resetting:
                        logger_mp.info("Reset command received, starting reset...")
                        self._start_reset()
                time.sleep(0.005)
            except Exception as e:
                logger_mp.error(f"Error in reset subscription loop: {e}")
                time.sleep(0.1)

    def _start_reset(self):
        """Begin a smooth 3-second reset to default pose."""
        with self.teleop_data_lock:
            self.reset_start_height = self.torso_height
            self.reset_start_rpy = self.torso_rpy.copy()
        self.reset_start_arm_q = self.wb_ctrl.get_current_dual_arm_q()
        self.reset_start_time = time.time()
        self.resetting = True
        self._open_hands()
        # Flush the teleop queue so stale commands aren't replayed after reset
        self.torso_rpy = np.zeros(3)  # Torso roll, pitch, yaw
        self.torso_height = 0.75  # Default torso height
        self.velocity_commands = np.zeros(3)
        self.has_velocity_commands = False
        try:
            while True:
                drained = self.reader_teleop.take(N=8)
                if len(drained) == 0:
                    break
        except Exception:
            pass

    def _open_hands(self):
        """Send open command to both hands via the hand controller's shared flag."""
        self.hand_ctrl.hand_open.value = 1

    def _robot_state_pub_loop(self):
        """Robot state publisher loop"""
        logger_mp.info("Robot state publisher loop started")
        while self.running:
            loop_start = time.time()
            robot_state = self.wb_ctrl.get_current_estimated_state()
            imu_state = self.wb_ctrl.get_current_imu_state()
            joint_positions = self.wb_ctrl.get_current_motor_q()
            joint_velocities = self.wb_ctrl.get_current_motor_dq()
            velocity_commands = self.wb_ctrl.get_current_velocity_commands()
            from teleop_dds_msgs import G1RobotState
            msg = G1RobotState(
                timestamp=time.time(),
                torso_pose=robot_state["torso_pose"].tolist(),
                left_end_effector_pose=robot_state["left_end_effector_pose"].tolist(),
                right_end_effector_pose=robot_state["right_end_effector_pose"].tolist(),
                pelvis_pose=robot_state["pelvis_pose"].tolist(),
                left_foot_pose=robot_state["left_foot_pose"].tolist(),
                right_foot_pose=robot_state["right_foot_pose"].tolist(),
                left_end_effector_to_torso_pose=robot_state["left_end_effector_to_torso_pose"].tolist(),
                right_end_effector_to_torso_pose=robot_state["right_end_effector_to_torso_pose"].tolist(),
                joint_positions=joint_positions.tolist(),
                joint_velocities=joint_velocities.tolist(),
                velocity_commands=velocity_commands.tolist(),
                imu_quaternion=imu_state["quaternion"].tolist(),
                imu_accelerometer=imu_state["accelerometer"].tolist(),
                imu_gyroscope=imu_state["gyroscope"].tolist(),
                imu_rpy=imu_state["rpy"].tolist(),
            )
            self.writer_robot_state.write(msg)
            elapsed = time.time() - loop_start
            sleep_time = 1.0 / self.publish_freq - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _hand_state_pub_loop(self):
        """Hand state publisher loop"""
        logger_mp.info("Hand state publisher loop started")
        while self.running:
            loop_start = time.time()
            hand_tip_states = self.hand_ctrl.get_current_hand_tip_poses()
            hand_proprio_states = self.hand_ctrl.get_current_hand_proprio()
            left_hand_proprio = hand_proprio_states["left_hand_proprio"]
            right_hand_proprio = hand_proprio_states["right_hand_proprio"]
            hand_joint_positions = np.concatenate([right_hand_proprio[:Inspire_Num_Motors], left_hand_proprio[:Inspire_Num_Motors]])
            hand_finger_forces = np.concatenate([right_hand_proprio[Inspire_Num_Motors:], left_hand_proprio[Inspire_Num_Motors:]])
            from teleop_dds_msgs import EEFState
            msg = EEFState(
                timestamp=time.time(),
                joint_positions=hand_joint_positions.tolist(),
                finger_forces=hand_finger_forces.tolist(),
                hand_link_poses=hand_tip_states.tolist()
            )
            self.writer_eef_state.write(msg)
            elapsed = time.time() - loop_start
            sleep_time = 1.0 / self.publish_freq - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def control_loop(self):
        """Main control loop - called at control_freq Hz"""
        if self.resetting:
            elapsed = time.time() - self.reset_start_time
            alpha = min(1.0, elapsed / 3.0)
            interp_height = self.reset_start_height * (1.0 - alpha) + 0.75 * alpha
            interp_rpy = self.reset_start_rpy * (1.0 - alpha)
            self.wb_ctrl._update_lower_body_state(interp_height, interp_rpy)
            default_arm_q = self.wb_ctrl.config.default_arm_joint_pos
            interp_arm_q = self.reset_start_arm_q * (1.0 - alpha) + default_arm_q * alpha
            with self.wb_ctrl.ctrl_lock:
                self.wb_ctrl.q_arm_target[:] = interp_arm_q
                self.wb_ctrl.tauff_arm_target[:] = 0.0
            if alpha >= 1.0:
                self.resetting = False
                self.teleop_data_received = False
                logger_mp.info("Reset complete, waiting for new teleop data...")
            return
        if not self.teleop_data_received:
            self.wb_ctrl.ctrl_whole_body_at_reset()
            return
        try:
            with self.teleop_data_lock:
                left_arm_pose = self.left_arm_pose.copy()
                right_arm_pose = self.right_arm_pose.copy()
                torso_rpy = self.torso_rpy.copy()
                torso_height = self.torso_height
                vel_cmds = self.velocity_commands.copy() if self.has_velocity_commands else None
            self.wb_ctrl.ctrl_whole_body_from_cmds(left_arm_pose, right_arm_pose, torso_height, torso_rpy, vel_cmds)
        except Exception as e:
            logger_mp.error(f"Error in control loop: {e}")
            import traceback
            traceback.print_exc()
    
    def _publish_controller_reset_notify(self):
        """Notify data collection that the controller triggered a reset."""
        from teleop_dds_msgs import ResetCommand
        msg = ResetCommand(timestamp=time.time(), reset_requested=True)
        self.writer_controller_reset.write(msg)
        logger_mp.info("Controller reset notification published")

    def run(self):
        """Main run loop"""
        logger_mp.info("Starting control loop...")
        logger_mp.info("Robot moving to default position...")
        
        # Wait for user to press 's' to start control
        self.wb_ctrl.wait_for_control()
        speed_ramp_enabled = False  # Track if velocity ramp has been enabled
        last_button2_reset_time = 0.0
        try:
            loop_rate = 1.0 / self.control_freq
            while self.running:
                start_time = time.time()
                # Enable velocity ramp ONLY when teleoperation data is received
                if not speed_ramp_enabled and self.teleop_data_received:
                    logger_mp.info("Teleoperation data received! Enabling velocity ramp...")
                    self.wb_ctrl.speed_gradual_max()
                    speed_ramp_enabled = True

                # Button [2]: reset robot pose and notify data collection
                if (self.wb_ctrl.remote_controller.button[2] == 1
                        and not self.resetting
                        and time.time() - last_button2_reset_time > 3.0):
                    logger_mp.info("Button [2] pressed — resetting robot and data collection")
                    last_button2_reset_time = time.time()
                    self._start_reset()
                    self._publish_controller_reset_notify()

                # Run control loop
                self.control_loop()
                # Maintain loop rate
                elapsed = time.time() - start_time
                sleep_time = loop_rate - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger_mp.warning(f"Control loop exceeded target rate: {elapsed:.4f}s > {loop_rate:.4f}s")

                if self.wb_ctrl.remote_controller.button[3] == 1:
                    break
        except KeyboardInterrupt:
            logger_mp.info("Keyboard interrupt received, shutting down...")
        finally:
            self.shutdown()

    def shutdown(self):
        logger_mp.info("Shutting down G1 Inspire WB Controller...")
        self.running = False
        self.wb_ctrl.setup_damping()
        time.sleep(0.3)
        # clean up dds resources
        try:
            if hasattr(self, 'participant'):
                self.participant.delete_contained_entities()
        except Exception as e:
            logger_mp.error(f"Error cleaning up DDS: {e}")
        logger_mp.info("Shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 Inspire Whole Body Controller with DDS")
    
    # Controller parameters
    parser.add_argument('--control_freq', type=int, default=50,
                       help='Control loop frequency in Hz (default: 50)')
    parser.add_argument('--hand_type', type=str, default='dftp',
                       choices=['dftp', 'f1'],
                       help='Hand type: dftp or f1 (default: dftp)')
    parser.add_argument('--g1_type', type=int, default=0, choices=[0, 1],
                       help='G1 URDF type: 0=g1_body29_hand14_wrist4010, 1=g1_body29_hand14_wrist5010 (default: 0)')
    parser.add_argument('--publish_freq', type=int, default=50,
                       help='Robot state publishing frequency in Hz (default: 50)')
    parsed_args = parser.parse_args()
    
    # Create controller
    try:
        controller = G1InspireWBController(parsed_args)
        
        logger_mp.info("===========================================")
        logger_mp.info("G1 Inspire WB Controller (DDS) is running...")
        logger_mp.info(f"Control Frequency: {parsed_args.control_freq} Hz")
        logger_mp.info(f"Hand Type: {parsed_args.hand_type}, G1 Type: {parsed_args.g1_type}")
        logger_mp.info("Waiting for teleoperation data via DDS...")
        logger_mp.info("===========================================")
        
        # Run controller
        controller.run()
        
    except KeyboardInterrupt:
        logger_mp.info("Interrupted by user")
    except Exception as e:
        logger_mp.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.shutdown()
        logger_mp.info("Program exited")

