import argparse
import copy
import json
import os
import time
import numpy as np
import cv2
import threading
import sys
import signal
from datetime import datetime
import tempfile
import pickle
import queue
import yaml
import torch
policy_source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'htd_policy')
sys.path.insert(0, policy_source_dir)
from policy import make_policy
from pynput import keyboard
# DDS imports for synchronized teleoperation publishing
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter

# Add robot_control directory to path to import teleop_dds_msgs
robot_control_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'robot_control')
sys.path.insert(0, robot_control_dir)
from teleop_dds_msgs import TeleopDataWB
data_collect_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_collect')
sys.path.insert(0, data_collect_dir)
from camera_utils import list_video_devices, find_device_path_by_name

# Tactile key -> (hand, start, end) for slicing full 1062-dim tactile
_TACTILE_KEY_SLICE = {
    'left_pinky_tactile': ('left', 0, 185), 'right_pinky_tactile': ('right', 0, 185),
    'left_ring_tactile': ('left', 185, 370), 'right_ring_tactile': ('right', 185, 370),
    'left_middle_tactile': ('left', 370, 555), 'right_middle_tactile': ('right', 370, 555),
    'left_index_tactile': ('left', 555, 740), 'right_index_tactile': ('right', 555, 740),
    'left_thumb_tactile': ('left', 740, 950), 'right_thumb_tactile': ('right', 740, 950),
    'left_palm_tactile': ('left', 950, 1062), 'right_palm_tactile': ('right', 950, 1062),
}

# F1 capacitive tactile: key -> (hand, start, end) for slicing 37-dim normalized data
_CAP_TACTILE_KEY_SLICE = {
    'left_pinky_tactile_cap': ('left', 0, 5), 'right_pinky_tactile_cap': ('right', 0, 5),
    'left_ring_tactile_cap': ('left', 5, 10), 'right_ring_tactile_cap': ('right', 5, 10),
    'left_middle_tactile_cap': ('left', 10, 15), 'right_middle_tactile_cap': ('right', 10, 15),
    'left_index_tactile_cap': ('left', 15, 20), 'right_index_tactile_cap': ('right', 15, 20),
    'left_thumb_tactile_cap': ('left', 20, 25), 'right_thumb_tactile_cap': ('right', 20, 25),
    'left_palm_tactile_cap': ('left', 25, 37), 'right_palm_tactile_cap': ('right', 25, 37),
    'left_tactile_cap': ('left', 0, 37), 'right_tactile_cap': ('right', 0, 37),
}

_INSPIRE_F1_TOUCH_RAW_FIELDS = [
    'little_normal_force', 'little_tangential_force', 'little_direction',
    'little_proximity_lo', 'little_proximity_hi',
    'ring_normal_force', 'ring_tangential_force', 'ring_direction',
    'ring_proximity_lo', 'ring_proximity_hi',
    'middle_normal_force', 'middle_tangential_force', 'middle_direction',
    'middle_proximity_lo', 'middle_proximity_hi',
    'index_normal_force', 'index_tangential_force', 'index_direction',
    'index_proximity_lo', 'index_proximity_hi',
    'thumb_normal_force', 'thumb_tangential_force', 'thumb_direction',
    'thumb_proximity_lo', 'thumb_proximity_hi',
    'palm_left_normal_force', 'palm_left_tangential_force', 'palm_left_direction',
    'palm_center_normal_force', 'palm_center_tangential_force', 'palm_center_direction',
    'palm_right_normal_force', 'palm_right_tangential_force', 'palm_right_direction',
]

def _mxt_inference_config(config):
    """Ignore image activation checkpointing, which is only used in training."""
    config = copy.deepcopy(config)
    if isinstance(config, dict):
        for embodiment in config.values():
            if not isinstance(embodiment, dict):
                continue
            tokenizers = embodiment.get('tokenizer', {})
            if isinstance(tokenizers, dict):
                for name, tokenizer in tokenizers.items():
                    if 'image' in name and isinstance(tokenizer, dict):
                        tokenizer.pop('gradient_checkpoint', None)
    return config


def _mxt_camera_size(config, camera):
    """Resolve saved (height, width) exactly as the MXT training loader does."""
    size = []
    for axis, fallback in (('height', 480), ('width', 640)):
        value = config.get(f'{camera}_cam_{axis}')
        if value is None:
            value = config.get(axis, fallback)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'invalid saved {camera} camera {axis}: {value!r}')
        size.append(value)
    return tuple(size)


def _hit_camera_sizes(config):
    """Resolve saved (height, width) sizes, including legacy HIT metadata."""
    sizes = {}
    for camera in ('main', 'wrist'):
        size = []
        for axis, fallback in (('height', 480), ('width', 640)):
            value = config.get(f'{camera}_cam_{axis}')
            if value is None:
                value = config.get(axis, fallback)
                if camera == 'wrist' and axis == 'width':
                    if isinstance(value, int) and not isinstance(value, bool):
                        value //= 2
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'invalid saved HIT {camera} camera {axis}: {value!r}')
            size.append(value)
        sizes[camera] = tuple(size)
    if sizes['main'][1] % 2:
        raise ValueError('saved HIT main camera width must be even for stereo splitting')
    return sizes['main'], sizes['wrist']


def _process_f1_touch_raw(raw_uint16):
    """Convert 34 raw uint16 F1 touch values to 29 processed float values."""
    out = np.zeros(29, dtype=np.float32)
    o = 0
    for i in range(5):
        base = i * 5
        out[o]     = float(raw_uint16[base])     * 0.01
        out[o + 1] = float(raw_uint16[base + 1]) * 0.01
        d = int(raw_uint16[base + 2])
        out[o + 2] = -1.0 if d == 65535 else d * (np.pi / 180.0)
        prox = int(raw_uint16[base + 3]) | (int(raw_uint16[base + 4]) << 16)
        out[o + 3] = min(prox / 16_000_000.0, 1.0)
        o += 4
    for j in range(3):
        base = 25 + j * 3
        out[o]     = float(raw_uint16[base])     * 0.01
        out[o + 1] = float(raw_uint16[base + 1]) * 0.01
        d = int(raw_uint16[base + 2])
        out[o + 2] = -1.0 if d == 65535 else d * (np.pi / 180.0)
        o += 3
    return out

def _normalize_f1_touch(raw_29):
    """Normalize F1 tactile: 29-dim raw -> 37-dim (forces/20, sin/cos dir, proximity)."""
    out = np.zeros(37, dtype=np.float32)
    ri = 0; wi = 0
    for _ in range(5):
        n, t, d, p = raw_29[ri], raw_29[ri+1], raw_29[ri+2], raw_29[ri+3]
        contact = d != -1.0
        out[wi]   = n / 20.0
        out[wi+1] = t / 20.0
        out[wi+2] = np.sin(d) if contact else 0.0
        out[wi+3] = np.cos(d) if contact else 0.0
        out[wi+4] = p
        ri += 4; wi += 5
    for _ in range(3):
        n, t, d = raw_29[ri], raw_29[ri+1], raw_29[ri+2]
        contact = d != -1.0
        out[wi]   = n / 20.0
        out[wi+1] = t / 20.0
        out[wi+2] = np.sin(d) if contact else 0.0
        out[wi+3] = np.cos(d) if contact else 0.0
        ri += 3; wi += 4
    return out

class HTDPolicyRollout:
    def __init__(self, args):
        self.args = args
        self.running = True  # Flag to control main loop
        self.embodiment = self.args.embodiment
        self.device = self.args.device
        
        self.init_cameras()

        self.unified_to_unitree_left_eef_rot_remap = np.array([
            [1, 0, 0], [0, 0, -1], [0, 1, 0]
        ], dtype=np.float64)
        self.unified_to_unitree_right_eef_rot_remap = np.array([
            [1, 0, 0], [0, 0, 1], [0, -1, 0]
        ], dtype=np.float64)
        
        self.on_reset = False
        self.on_rollout = False
  
        inspire_hand_type = getattr(self.args, 'inspire_hand_type', 'dftp')
        if inspire_hand_type == 'f1':
            self.inspire_to_unified_eef = np.array([
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)
        else:
            self.inspire_to_unified_eef = np.array([
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)

        self.pause_commands = False
        
        self.init_embodiment_states()
        self.init_policy()
        self.rollout_counter = 0
        self.infer_interval = self.args.inference_interval
        self.command_traj_dict = {}
        
        # Set up network interface if specified
        if self.args.dds_interface is not None:
            self._setup_dds_interface(self.args.dds_interface)
        
        # Initialize DDS teleoperation publisher
        self.init_dds_publisher()
        
        self.inspire_hand_type = getattr(self.args, 'inspire_hand_type', 'dftp')
        self.robot_state = None
        self.robot_state_lock = threading.Lock()
        self.init_dds_robot_state_subscriber()
        self.robot_eef_state = None
        self.robot_eef_state_lock = threading.Lock()
        self.init_dds_eef_state_subscriber()
        self.init_dds_reset_command_publisher()
        self.touch_state_lock = threading.Lock()
        if self.inspire_hand_type == 'f1':
            self.left_touch_raw = np.zeros(34, dtype=np.uint16)
            self.right_touch_raw = np.zeros(34, dtype=np.uint16)
        else:
            self.left_touch_raw = np.zeros(1062, dtype=np.int16)
            self.right_touch_raw = np.zeros(1062, dtype=np.int16)
        self.init_dds_touch_state_subscriber()
    


    def init_cameras(self):
        # initialize all cameras
        self.desired_stream_fps = self.args.desired_stream_fps
        
        # Resolution mapping for different camera types
        resolution_map = {
            "480p": (480, 640),
            "720p": (720, 1280), 
            "1080p": (1080, 1920)
        }
        
        self.head_camera_resolution = resolution_map[self.args.head_camera_res]
        self.head_view_resolution = resolution_map[self.args.head_camera_res]
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.head_frame_res = (self.head_view_resolution[0] - self.crop_size_h, self.head_view_resolution[1] - 2 * self.crop_size_w)
        self.head_color_frame = np.zeros((self.head_frame_res[0], 2 * self.head_frame_res[1], 3), dtype=np.uint8)
        device_map = list_video_devices()
        head_camera_name = "3D USB Camera"
        device_path = find_device_path_by_name(device_map, head_camera_name)
        print(f"Head camera device path: {device_path}")
        self.head_cap = cv2.VideoCapture(device_path)
        self.head_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.head_camera_resolution[1])
        self.head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.head_camera_resolution[0])
        self.head_cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
        
        if self.args.use_wrist_camera:
            self.wrist_camera_resolution = resolution_map[self.args.wrist_camera_res]
            self.wrist_view_resolution = resolution_map[self.args.wrist_camera_res]
            device_map = list_video_devices()
            wrist_camera_name = "Global Shutter Camera"
            self.wrist_left_color_frame = np.zeros((self.wrist_view_resolution[0], self.wrist_view_resolution[1], 3), dtype=np.uint8)
            self.wrist_right_color_frame = np.zeros_like(self.wrist_left_color_frame)
            wrist_paths = self._get_device_paths_by_name(device_map, wrist_camera_name)
            if len(wrist_paths) < 2:
                raise RuntimeError(f"Expected 2 wrist cameras for g1_inspire_wb, found {len(wrist_paths)}: {device_map}")
            left_path, right_path = self._select_wrist_cameras(wrist_paths)
            self.wrist_cap_left = self._create_wrist_capture(left_path)
            self.wrist_cap_right = self._create_wrist_capture(right_path)
            print(f"[wrist] Assigned left: {left_path}")
            print(f"[wrist] Assigned right: {right_path}")


    def _create_wrist_capture(self, device_path):
        """Create and configure a wrist camera VideoCapture."""
        cap = cv2.VideoCapture(device_path)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.wrist_camera_resolution[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.wrist_camera_resolution[0])
        cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
        return cap

    def _get_device_paths_by_name(self, device_map, name):
        device_paths = []
        for device_name, paths in device_map.items():
            if name in device_name:
                device_paths.append(paths[0])
        return device_paths

    def _select_wrist_cameras(self, device_paths):
        window_name = "Wrist Cam Preview"
        current_idx = 0
        selected_left = None
        selected_right = None
        cap = None
        print("[wrist] Type 'l' to set LEFT or 'r' to set RIGHT for current camera.")
        print("[wrist] Type 'q' to quit.")
        cmd_queue = queue.Queue()

        def _input_loop():
            while True:
                try:
                    cmd = input("> ").strip().lower()
                except EOFError:
                    cmd = "q"
                if cmd in ("l", "r", "q"):
                    cmd_queue.put(cmd)
        try:
            input_thread = threading.Thread(target=_input_loop, daemon=True)
            input_thread.start()
            while selected_left is None and selected_right is None:
                if cap is None:
                    cap = self._create_wrist_capture(device_paths[current_idx])

                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, frame)

                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    cmd = None

                if cmd == "l":
                    selected_left = device_paths[current_idx]
                elif cmd == "r":
                    selected_right = device_paths[current_idx]
                elif cmd == "q":
                    print("[wrist] Selection cancelled, defaulting to indices 0 and 1")
                    selected_left = device_paths[0]
                    selected_right = device_paths[1]
                cv2.waitKey(1)

            remaining_paths = [p for p in device_paths if p != selected_left and p != selected_right]
            if selected_left is None:
                selected_left = remaining_paths[0]
            elif selected_right is None:
                selected_right = remaining_paths[0]
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyWindow(window_name)
        return selected_left, selected_right

    def head_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while self.running:
                start_time = time.time()
                ret, frame = self.head_cap.read()
                if 'g1' in self.args.embodiment:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                image_left = frame[:, :self.head_frame_res[1], :]
                image_right = frame[:, self.head_frame_res[1]:, :]
                if self.crop_size_w != 0:
                    bgr = np.hstack((image_left[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                    image_right[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                else:
                    bgr = np.hstack((image_left[self.crop_size_h:, :],
                                    image_right[self.crop_size_h:, :]))

                self.head_color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.head_cap.release()

    def wrist_left_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while self.running:
                start_time = time.time()
                ret, frame = self.wrist_cap_left.read()
                if ret and frame is not None:
                    left_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.wrist_left_color_frame = cv2.resize(left_rgb, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.wrist_cap_left.release()

    def wrist_right_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while self.running:
                start_time = time.time()
                ret, frame = self.wrist_cap_right.read()
                if ret and frame is not None:
                    right_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.wrist_right_color_frame = cv2.resize(right_rgb, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.wrist_cap_right.release()

    
    def init_embodiment_states(self):
        self.body_pose_proprio = np.zeros(6)
        self.eef_pose_proprio = np.zeros(12)
        self.eef_to_body_pose_proprio = np.zeros(12)
        self.eef_link_proprio = np.zeros(30)  # 5*3*2, fingertip positions for both hands
        self.robot_joint_pos_proprio = np.zeros(29)
        self.left_eef_joint_pos_proprio = np.zeros(6)
        self.right_eef_joint_pos_proprio = np.zeros(6)
        self.left_eef_finger_force_proprio = np.zeros(6)
        self.right_eef_finger_force_proprio = np.zeros(6)
        # whole-hand tactile buffers (1062-dim, divided by 4095)
        self.left_tactile = np.zeros(1062, dtype=np.float32)
        self.right_tactile = np.zeros(1062, dtype=np.float32)
        # per-finger tactile buffers (flat, matching processed HDF5 structure)
        self.left_pinky_tactile = np.zeros(185, dtype=np.float32)
        self.right_pinky_tactile = np.zeros(185, dtype=np.float32)
        self.left_ring_tactile = np.zeros(185, dtype=np.float32)
        self.right_ring_tactile = np.zeros(185, dtype=np.float32)
        self.left_middle_tactile = np.zeros(185, dtype=np.float32)
        self.right_middle_tactile = np.zeros(185, dtype=np.float32)
        self.left_index_tactile = np.zeros(185, dtype=np.float32)
        self.right_index_tactile = np.zeros(185, dtype=np.float32)
        self.left_thumb_tactile = np.zeros(210, dtype=np.float32)
        self.right_thumb_tactile = np.zeros(210, dtype=np.float32)
        self.left_palm_tactile = np.zeros(112, dtype=np.float32)
        self.right_palm_tactile = np.zeros(112, dtype=np.float32)
        # F1 capacitive tactile buffers (37-dim whole hand, 5-dim per finger, 12-dim palm)
        self.left_tactile_cap = np.zeros(37, dtype=np.float32)
        self.right_tactile_cap = np.zeros(37, dtype=np.float32)
        self.left_pinky_tactile_cap = np.zeros(5, dtype=np.float32)
        self.right_pinky_tactile_cap = np.zeros(5, dtype=np.float32)
        self.left_ring_tactile_cap = np.zeros(5, dtype=np.float32)
        self.right_ring_tactile_cap = np.zeros(5, dtype=np.float32)
        self.left_middle_tactile_cap = np.zeros(5, dtype=np.float32)
        self.right_middle_tactile_cap = np.zeros(5, dtype=np.float32)
        self.left_index_tactile_cap = np.zeros(5, dtype=np.float32)
        self.right_index_tactile_cap = np.zeros(5, dtype=np.float32)
        self.left_thumb_tactile_cap = np.zeros(5, dtype=np.float32)
        self.right_thumb_tactile_cap = np.zeros(5, dtype=np.float32)
        self.left_palm_tactile_cap = np.zeros(12, dtype=np.float32)
        self.right_palm_tactile_cap = np.zeros(12, dtype=np.float32)
        # # wrist camera frame buffers (initialized for obs builder even before cameras start)
        # self.wrist_left_color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # self.wrist_right_color_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _setup_dds_interface(self, interface_name):
        """Set up CycloneDDS to use a specific network interface"""
        if not interface_name or 'CYCLONEDDS_URI' in os.environ:
            return
        
        config_content = f'''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain>
        <General>
            <Interfaces>
                <NetworkInterface name="{interface_name}"/>
            </Interfaces>
        </General>
    </Domain>
</CycloneDDS>'''
        
        fd, config_file = tempfile.mkstemp(suffix='.xml', prefix='cyclonedds_', text=True)
        os.close(fd)
        with open(config_file, 'w') as f:
            f.write(config_content)
        os.environ['CYCLONEDDS_URI'] = f'file://{config_file}'
        self._dds_config_file = config_file
        print(f"✓ Configured DDS network interface: {interface_name}")
    
    def init_dds_publisher(self):
        """Initialize DDS publisher for synchronized teleoperation data"""
        print("Initializing DDS publisher for synchronized teleoperation...")
        
        # Create DDS domain participant (domain 0)
        self.dds_participant = DomainParticipant(0)
        
        # Create synchronized teleoperation topic
        kTopicTeleopData = "rt/teleop/synchronized_data"
        
        self.teleop_data_type = TeleopDataWB
        print("Using TeleopDataWB for whole-body teleoperation")
        
        self.dds_topic_teleop = Topic(self.dds_participant, kTopicTeleopData, self.teleop_data_type)
        
        # Create data writer (publisher)
        self.dds_writer_teleop = DataWriter(self.dds_participant, self.dds_topic_teleop)
        
        print(f"✅ DDS publisher initialized on topic: {kTopicTeleopData}")
    
    def init_dds_robot_state_subscriber(self):
        """Initialize DDS subscriber for robot state"""
        print("Initializing DDS subscriber for robot state...")
        
        from cyclonedds.sub import DataReader
        from teleop_dds_msgs import G1RobotState
        
        kTopicRobotState = "rt/robot/state"
        self.dds_topic_robot_state = Topic(
            self.dds_participant,
            kTopicRobotState,
            G1RobotState,
        )
        self.dds_reader_robot_state = DataReader(self.dds_participant, self.dds_topic_robot_state)
        
        self.robot_state_thread = threading.Thread(target=self._robot_state_subscription_loop, daemon=True)
        self.robot_state_thread.start()
        
        print(f"✅ DDS robot state subscriber initialized on topic: {kTopicRobotState}")
    
    def init_dds_eef_state_subscriber(self):
        """Initialize DDS subscriber for EEF state (hand state)"""
        print("Initializing DDS subscriber for EEF state...")
        
        from cyclonedds.sub import DataReader
        from teleop_dds_msgs import EEFState
        
        # Create EEF state topic
        kTopicEEFState = "rt/robot/eef_state"
        self.dds_topic_eef_state = Topic(self.dds_participant, kTopicEEFState, EEFState)
        self.dds_reader_eef_state = DataReader(self.dds_participant, self.dds_topic_eef_state)
        
        # Start subscription thread
        self.eef_state_thread = threading.Thread(target=self._eef_state_subscription_loop, daemon=True)
        self.eef_state_thread.start()
        
        print(f"✅ DDS EEF state subscriber initialized on topic: {kTopicEEFState}")
    
    def _eef_state_subscription_loop(self):
        """DDS subscription loop for EEF state running in separate thread"""
        print("EEF state subscription loop started")
        
        while self.running:
            try:
                samples = self.dds_reader_eef_state.take(N=1)
                for sample in samples:
                    if sample is None:
                        continue
                    eef_state = {
                        'timestamp': sample.timestamp,
                        'joint_positions': np.array(sample.joint_positions),
                        'finger_forces': np.array(sample.finger_forces),
                        'hand_link_poses': np.array(sample.hand_link_poses),
                    }
                    with self.robot_eef_state_lock:
                        self.robot_eef_state = eef_state
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in EEF state subscription loop: {e}")
                time.sleep(0.1)
        
        print("EEF state subscription loop ended")

    def init_dds_touch_state_subscriber(self):
        """Initialize DDS subscribers for Inspire hand touch topics."""
        print(f"Initializing DDS subscriber for Inspire hand touch state (hand_type={self.inspire_hand_type})...")
        try:
            from cyclonedds.sub import DataReader

            if self.inspire_hand_type == 'f1':
                from inspire_f1_sdk.inspire_dds import inspire_hand_f1_touch
                k_topic_left = "rt/inspire_hand_f1/touch/l"
                k_topic_right = "rt/inspire_hand_f1/touch/r"
                touch_msg_type = inspire_hand_f1_touch
            else:
                from inspire_sdkpy import inspire_dds
                k_topic_left = "rt/inspire_hand/touch/l"
                k_topic_right = "rt/inspire_hand/touch/r"
                touch_msg_type = inspire_dds.inspire_hand_touch

            self.dds_topic_touch_left = Topic(self.dds_participant, k_topic_left, touch_msg_type)
            self.dds_topic_touch_right = Topic(self.dds_participant, k_topic_right, touch_msg_type)
            self.left_touch_subscriber = DataReader(self.dds_participant, self.dds_topic_touch_left)
            self.right_touch_subscriber = DataReader(self.dds_participant, self.dds_topic_touch_right)

            self.touch_state_thread = threading.Thread(target=self._touch_state_subscription_loop, daemon=True)
            self.touch_state_thread.start()
            print(f"DDS Inspire touch subscribers initialized on topics: {k_topic_left}, {k_topic_right}")
        except Exception as e:
            self.left_touch_subscriber = None
            self.right_touch_subscriber = None
            print(f"Failed to initialize Inspire touch subscribers: {e}")

    def _flatten_touch_msg(self, msg):
        if self.inspire_hand_type == 'f1':
            return np.array(
                [getattr(msg, field, 0) for field in _INSPIRE_F1_TOUCH_RAW_FIELDS],
                dtype=np.uint16,
            )
        FIELD_SIZES = {
            "fingerone_tip_touch": 9, "fingerone_top_touch": 96, "fingerone_palm_touch": 80,
            "fingertwo_tip_touch": 9, "fingertwo_top_touch": 96, "fingertwo_palm_touch": 80,
            "fingerthree_tip_touch": 9, "fingerthree_top_touch": 96, "fingerthree_palm_touch": 80,
            "fingerfour_tip_touch": 9, "fingerfour_top_touch": 96, "fingerfour_palm_touch": 80,
            "fingerfive_tip_touch": 9, "fingerfive_top_touch": 96, "fingerfive_middle_touch": 9, "fingerfive_palm_touch": 96,
            "palm_touch": 112,
        }
        flat_values = []
        for field_name, expected_size in FIELD_SIZES.items():
            field_value = getattr(msg, field_name, None)
            if field_value is None:
                values = np.zeros(expected_size, dtype=np.int16)
            else:
                values = np.asarray(field_value, dtype=np.int16).reshape(-1)
            if values.size < expected_size:
                padded = np.zeros(expected_size, dtype=np.int16)
                padded[:values.size] = values
                values = padded
            elif values.size > expected_size:
                values = values[:expected_size]
            flat_values.append(values)
        return np.concatenate(flat_values)

    def _touch_state_subscription_loop(self):
        """Subscription loop for Inspire hand touch topics."""
        print("Inspire touch subscription loop started")
        while self.running:
            try:
                left_samples = self.left_touch_subscriber.take(N=1)
                for sample in left_samples:
                    if sample is not None:
                        with self.touch_state_lock:
                            self.left_touch_raw = self._flatten_touch_msg(sample)

                right_samples = self.right_touch_subscriber.take(N=1)
                for sample in right_samples:
                    if sample is not None:
                        with self.touch_state_lock:
                            self.right_touch_raw = self._flatten_touch_msg(sample)
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in Inspire touch subscription loop: {e}")
                time.sleep(0.1)
        print("Inspire touch subscription loop ended")

    def init_dds_reset_command_publisher(self):
        """Initialize DDS publisher for reset commands"""
        print("Initializing DDS publisher for reset commands...")
        
        from teleop_dds_msgs import ResetCommand
        
        # Create reset command topic
        kTopicResetCommand = "rt/robot/reset_command"
        self.dds_topic_reset_command = Topic(self.dds_participant, kTopicResetCommand, ResetCommand)
        self.dds_writer_reset_command = DataWriter(self.dds_participant, self.dds_topic_reset_command)
        
        print(f"✅ DDS reset command publisher initialized on topic: {kTopicResetCommand}")
    
    def _robot_state_subscription_loop(self):
        """DDS subscription loop for robot state running in separate thread"""
        print("Robot state subscription loop started")
        
        while self.running:
            try:
                samples = self.dds_reader_robot_state.take(N=1)
                for sample in samples:
                    if sample is None:
                        continue
                    robot_state = {
                        'timestamp': sample.timestamp,
                        'torso_pose': np.array(sample.torso_pose),
                        'left_eef_pose': np.array(sample.left_end_effector_pose),
                        'right_eef_pose': np.array(sample.right_end_effector_pose),
                        'left_eef_to_torso_pose': np.array(sample.left_end_effector_to_torso_pose),
                        'right_eef_to_torso_pose': np.array(sample.right_end_effector_to_torso_pose),
                        'joint_positions': np.array(sample.joint_positions),
                        'joint_velocities': np.array(sample.joint_velocities),
                        'velocity_commands': np.array(sample.velocity_commands),
                        'imu_accelerometer': np.array(sample.imu_accelerometer),
                        'imu_gyroscope': np.array(sample.imu_gyroscope),
                        'imu_rpy': np.array(sample.imu_rpy),
                    }
                    with self.robot_state_lock:
                        self.robot_state = robot_state
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in robot state subscription loop: {e}")
                time.sleep(0.1)
        
        print("Robot state subscription loop ended")
    

    def update_embodiment_proprio_states(self):
        with self.robot_state_lock:
            if self.robot_state is None:
                return
            self.body_pose_proprio = self.robot_state['torso_pose'].copy()
            right_eef_pose = self.robot_state['right_eef_pose'].copy()
            left_eef_pose = self.robot_state['left_eef_pose'].copy()
            right_eef_to_body = self.robot_state['right_eef_to_torso_pose'].copy()
            left_eef_to_body = self.robot_state['left_eef_to_torso_pose'].copy()
            self.robot_joint_pos_proprio = self.robot_state['joint_positions'].copy()
        # apply rotation remaps (same as data_collect)
        right_eef_rot = self.rpy_to_rot_mat(right_eef_pose[3:])
        right_eef_rot = right_eef_rot @ self.unified_to_unitree_right_eef_rot_remap.T
        right_eef_pose[3:] = self.rot_mat_to_rpy(right_eef_rot)
        left_eef_rot = self.rpy_to_rot_mat(left_eef_pose[3:])
        left_eef_rot = left_eef_rot @ self.unified_to_unitree_left_eef_rot_remap.T
        left_eef_pose[3:] = self.rot_mat_to_rpy(left_eef_rot)
        self.eef_pose_proprio = np.concatenate([right_eef_pose, left_eef_pose])
        right_rot = self.rpy_to_rot_mat(right_eef_to_body[3:])
        right_rot = right_rot @ self.unified_to_unitree_right_eef_rot_remap.T
        right_eef_to_body[3:] = self.rot_mat_to_rpy(right_rot)
        left_rot = self.rpy_to_rot_mat(left_eef_to_body[3:])
        left_rot = left_rot @ self.unified_to_unitree_left_eef_rot_remap.T
        left_eef_to_body[3:] = self.rot_mat_to_rpy(left_rot)
        self.eef_to_body_pose_proprio = np.concatenate([right_eef_to_body, left_eef_to_body])
        with self.robot_eef_state_lock:
            if self.robot_eef_state is None:
                return
            eef_link_raw = self.robot_eef_state['hand_link_poses'].copy()
            finger_forces = self.robot_eef_state.get('finger_forces', np.zeros(12))
            eef_joint_positions = self.robot_eef_state.get('joint_positions', np.zeros(12))
        right_eef_link = eef_link_raw[:15].reshape(-1, 3)
        left_eef_link = eef_link_raw[15:30].reshape(-1, 3)
        right_eef_link = (self.inspire_to_unified_eef[:3, :3] @ right_eef_link.T).T.flatten()
        left_eef_link = (self.inspire_to_unified_eef[:3, :3] @ left_eef_link.T).T.flatten()
        self.eef_link_proprio = np.concatenate([right_eef_link, left_eef_link])
        self.right_eef_finger_force_proprio = np.array(finger_forces[:6], dtype=np.float32)
        self.left_eef_finger_force_proprio = np.array(finger_forces[6:12], dtype=np.float32)
        self.right_eef_joint_pos_proprio = np.array(eef_joint_positions[:6], dtype=np.float32)
        self.left_eef_joint_pos_proprio = np.array(eef_joint_positions[6:12], dtype=np.float32)
        with self.touch_state_lock:
            left_raw = self.left_touch_raw.copy()
            right_raw = self.right_touch_raw.copy()
        if self.inspire_hand_type == 'f1':
            left_29 = _process_f1_touch_raw(left_raw)
            right_29 = _process_f1_touch_raw(right_raw)
            lt = _normalize_f1_touch(left_29)
            rt = _normalize_f1_touch(right_29)
            self.left_tactile_cap = lt
            self.right_tactile_cap = rt
            self.left_pinky_tactile_cap = lt[0:5]
            self.right_pinky_tactile_cap = rt[0:5]
            self.left_ring_tactile_cap = lt[5:10]
            self.right_ring_tactile_cap = rt[5:10]
            self.left_middle_tactile_cap = lt[10:15]
            self.right_middle_tactile_cap = rt[10:15]
            self.left_index_tactile_cap = lt[15:20]
            self.right_index_tactile_cap = rt[15:20]
            self.left_thumb_tactile_cap = lt[20:25]
            self.right_thumb_tactile_cap = rt[20:25]
            self.left_palm_tactile_cap = lt[25:37]
            self.right_palm_tactile_cap = rt[25:37]
        else:
            lt = left_raw.astype(np.float32) / 4095.0
            rt = right_raw.astype(np.float32) / 4095.0
            self.left_tactile = lt
            self.right_tactile = rt
            self.left_pinky_tactile = lt[:185]
            self.right_pinky_tactile = rt[:185]
            self.left_ring_tactile = lt[185:370]
            self.right_ring_tactile = rt[185:370]
            self.left_middle_tactile = lt[370:555]
            self.right_middle_tactile = rt[370:555]
            self.left_index_tactile = lt[555:740]
            self.right_index_tactile = rt[555:740]
            self.left_thumb_tactile = lt[740:950]
            self.right_thumb_tactile = rt[740:950]
            self.left_palm_tactile = lt[950:1062]
            self.right_palm_tactile = rt[950:1062]

    def send_reset_command(self):
        """Send reset command to robot controller via DDS"""
        if not hasattr(self, 'dds_writer_reset_command'):
            return
        from teleop_dds_msgs import ResetCommand
        msg = ResetCommand(
            timestamp=time.time(),
            reset_requested=True
        )
        self.dds_writer_reset_command.write(msg)
        print(f"Reset command sent to {self.embodiment} controller")
    
    def publish_wb_teleop_data(self, left_eef_pos, left_eef_rot, right_eef_pos, right_eef_rot, left_hand_joints, right_hand_joints, teleop_torso_rot_uni, teleop_torso_height, velocity_commands=None):
        """
        Publish synchronized whole-body teleoperation data via DDS (includes torso data)
        Args:
            left_eef_pos: 3D position of left end-effector
            left_eef_rot: RPY rotation of left end-effector
            right_eef_pos: 3D position of right end-effector
            right_eef_rot: RPY rotation of right end-effector
            left_hand_joints: 75D hand joints
            right_hand_joints: 75D hand joints
            teleop_torso_rot_uni: RPY rotation of torso (3D)
            teleop_torso_height: Height of torso (scalar)
            velocity_commands: 3D velocity [lin_vel_x, lin_vel_y, ang_vel_z] (optional)
        """
        # Convert RPY to rotation matrix
        left_rot_mat = self.rpy_to_rot_mat(left_eef_rot)
        right_rot_mat = self.rpy_to_rot_mat(right_eef_rot)
        
        # Construct 4x4 transformation matrices for arms
        left_arm_pose = np.eye(4)
        left_arm_pose[:3, :3] = left_rot_mat
        left_arm_pose[:3, 3] = left_eef_pos
        
        right_arm_pose = np.eye(4)
        right_arm_pose[:3, :3] = right_rot_mat
        right_arm_pose[:3, 3] = right_eef_pos

        if velocity_commands is None:
            vel_list = []
        else:
            vel_list = velocity_commands.tolist() if hasattr(velocity_commands, 'tolist') else list(velocity_commands)

        msg = TeleopDataWB(
            timestamp=time.time(),
            left_arm_pose=left_arm_pose.flatten().tolist(),
            right_arm_pose=right_arm_pose.flatten().tolist(),
            left_hand_pose=left_hand_joints.tolist(),
            right_hand_pose=right_hand_joints.tolist(),
            torso_rpy=teleop_torso_rot_uni.tolist() if hasattr(teleop_torso_rot_uni, 'tolist') else list(teleop_torso_rot_uni),
            torso_height=float(teleop_torso_height),
            velocity_commands=vel_list
        )
        self.dds_writer_teleop.write(msg)

    def policy_rollout(self):
        rollout_idx = self.rollout_counter % self.infer_interval
        if self.policy_class == 'HIT':
            if rollout_idx == 0:
                self.update_embodiment_proprio_states()
                qpos, curr_image = self._build_g1_hit_obs()
                with torch.no_grad():
                    a_hat = self.bc_model(qpos, curr_image)
                actions_np = a_hat.squeeze(0).cpu().detach().numpy()
                actions_np = self.hit_post_process(actions_np)
                self._process_g1_hit_action_trajectory(actions_np)
            self._execute_g1_action(rollout_idx)
        else:
            use_dream = self.args.dream_touch and self.has_dream_head
            if rollout_idx == 0:
                self.update_embodiment_proprio_states()
                obs_dict, obs_mask_dict = self._build_g1_obs()
                with torch.no_grad():
                    result = self.bc_model(obs_dict, self.embodiment, obs_mask_dict, infer_dream=use_dream)
                if use_dream:
                    self.act_dict, dream_dict = result
                    self._flush_dream_accum()
                    self._init_dream_accum(dream_dict)
                else:
                    self.act_dict = result
                self._process_g1_action_trajectory()
            self._execute_g1_action(rollout_idx)
            if use_dream and self._dream_accum is not None:
                self._record_dream_gt(rollout_idx)
        self.rollout_counter += 1

    def _build_g1_obs(self):
        """Build observation and mask dicts for g1_inspire_wb from current proprio state."""
        obs_dict = {}
        obs_mask_dict = {}
        # body_pose_proprio (6D)
        obs_dict['body_pose_proprio'] = torch.tensor(self.body_pose_proprio).float().to(self.device).unsqueeze(0)
        obs_mask_dict['body_pose_proprio'] = torch.ones(6).bool().to(self.device).unsqueeze(0)
        # eef_pose_proprio (12D absolute EEF pose)
        if 'eef_pose_proprio' in self.obs_keys:
            obs_dict['eef_pose_proprio'] = torch.tensor(self.eef_pose_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['eef_pose_proprio'] = torch.ones(12).bool().to(self.device).unsqueeze(0)
        # eef_to_body_pose: split or combined depending on model config
        if 'left_eef_to_body_pose_proprio' in self.obs_keys:
            obs_dict['right_eef_to_body_pose_proprio'] = torch.tensor(self.eef_to_body_pose_proprio[:6]).float().to(self.device).unsqueeze(0)
            obs_dict['left_eef_to_body_pose_proprio'] = torch.tensor(self.eef_to_body_pose_proprio[6:12]).float().to(self.device).unsqueeze(0)
            obs_mask_dict['right_eef_to_body_pose_proprio'] = torch.ones(6).bool().to(self.device).unsqueeze(0)
            obs_mask_dict['left_eef_to_body_pose_proprio'] = torch.ones(6).bool().to(self.device).unsqueeze(0)
        elif 'eef_to_body_pose_proprio' in self.obs_keys:
            obs_dict['eef_to_body_pose_proprio'] = torch.tensor(self.eef_to_body_pose_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['eef_to_body_pose_proprio'] = torch.ones(12).bool().to(self.device).unsqueeze(0)
        # eef_link_proprio: split or combined
        if 'left_eef_link_proprio' in self.obs_keys:
            obs_dict['right_eef_link_proprio'] = torch.tensor(self.eef_link_proprio[:15]).float().to(self.device).unsqueeze(0)
            obs_dict['left_eef_link_proprio'] = torch.tensor(self.eef_link_proprio[15:30]).float().to(self.device).unsqueeze(0)
            obs_mask_dict['right_eef_link_proprio'] = torch.ones(15).bool().to(self.device).unsqueeze(0)
            obs_mask_dict['left_eef_link_proprio'] = torch.ones(15).bool().to(self.device).unsqueeze(0)
        elif 'eef_link_proprio' in self.obs_keys:
            obs_dict['eef_link_proprio'] = torch.tensor(self.eef_link_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['eef_link_proprio'] = torch.ones(30).bool().to(self.device).unsqueeze(0)
        # main image
        main_frame = self.head_color_frame.copy()
        target_h, target_w = self.main_cam_size
        if main_frame.shape[:2] != (target_h, target_w):
            main_frame = cv2.resize(main_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        main_image = torch.from_numpy(main_frame).float()
        main_image.div_(255.0)
        main_image = main_image.permute(2, 0, 1).to(self.device)
        obs_dict['main_image'] = main_image.unsqueeze(0)
        obs_mask_dict['main_image'] = torch.tensor([True]).to(self.device).unsqueeze(0)
        # wrist images (dual)
        if self.args.use_wrist_camera:
            target_h, target_w = self.wrist_cam_size
            if 'left_wrist_image' in self.obs_keys:
                left_frame = self.wrist_left_color_frame.copy()
                if left_frame.shape[1] != target_w or left_frame.shape[0] != target_h:
                    left_frame = cv2.resize(left_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                left_wrist = torch.from_numpy(left_frame).float().div_(255.0)
                left_wrist = left_wrist.permute(2, 0, 1).to(self.device)
                obs_dict['left_wrist_image'] = left_wrist.unsqueeze(0)
                obs_mask_dict['left_wrist_image'] = torch.tensor([True]).to(self.device).unsqueeze(0)
            if 'right_wrist_image' in self.obs_keys:
                right_frame = self.wrist_right_color_frame.copy()
                if right_frame.shape[1] != target_w or right_frame.shape[0] != target_h:
                    right_frame = cv2.resize(right_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                right_wrist = torch.from_numpy(right_frame).float().div_(255.0)
                right_wrist = right_wrist.permute(2, 0, 1).to(self.device)
                obs_dict['right_wrist_image'] = right_wrist.unsqueeze(0)
                obs_mask_dict['right_wrist_image'] = torch.tensor([True]).to(self.device).unsqueeze(0)
        # robot joint pos proprio (29-dim)
        if 'robot_joint_pos_proprio' in self.obs_keys:
            obs_dict['robot_joint_pos_proprio'] = torch.tensor(self.robot_joint_pos_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['robot_joint_pos_proprio'] = torch.ones(29).bool().to(self.device).unsqueeze(0)
        # eef joint pos proprio (6-dim each)
        if 'left_eef_joint_pos_proprio' in self.obs_keys:
            obs_dict['left_eef_joint_pos_proprio'] = torch.tensor(self.left_eef_joint_pos_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['left_eef_joint_pos_proprio'] = torch.ones(6).bool().to(self.device).unsqueeze(0)
        if 'right_eef_joint_pos_proprio' in self.obs_keys:
            obs_dict['right_eef_joint_pos_proprio'] = torch.tensor(self.right_eef_joint_pos_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['right_eef_joint_pos_proprio'] = torch.ones(6).bool().to(self.device).unsqueeze(0)
        # finger force proprio
        if 'left_eef_finger_force_proprio' in self.obs_keys:
            obs_dict['left_eef_finger_force_proprio'] = torch.tensor(self.left_eef_finger_force_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['left_eef_finger_force_proprio'] = torch.ones(6).bool().to(self.device).unsqueeze(0)
        if 'right_eef_finger_force_proprio' in self.obs_keys:
            obs_dict['right_eef_finger_force_proprio'] = torch.tensor(self.right_eef_finger_force_proprio).float().to(self.device).unsqueeze(0)
            obs_mask_dict['right_eef_finger_force_proprio'] = torch.ones(6).bool().to(self.device).unsqueeze(0)
        # tactile obs (per-finger)
        tactile_keys = [k for k in self.obs_keys if 'tactile' in k]
        for key in tactile_keys:
            buf = getattr(self, key, None)
            if buf is not None:
                obs_dict[key] = torch.tensor(buf).float().to(self.device).unsqueeze(0)
                obs_mask_dict[key] = torch.ones(buf.shape[0]).bool().to(self.device).unsqueeze(0)
        return obs_dict, obs_mask_dict

    def _process_g1_action_trajectory(self):
        """Process model output into teleop-ready trajectory for g1_inspire_wb."""
        body_pose_traj = self.act_dict['body_pose'].squeeze(0).cpu().detach().numpy()
        # eef_to_body: split or combined
        if 'left_eef_to_body_pose' in self.act_dict:
            right_eef_traj = self.act_dict['right_eef_to_body_pose'].squeeze(0).cpu().detach().numpy()
            left_eef_traj = self.act_dict['left_eef_to_body_pose'].squeeze(0).cpu().detach().numpy()
        else:
            combined = self.act_dict['eef_to_body_pose'].squeeze(0).cpu().detach().numpy()
            right_eef_traj = combined[:, :6]
            left_eef_traj = combined[:, 6:]
        # eef_link: split or combined
        if 'left_eef_link_pose' in self.act_dict:
            right_link_traj = self.act_dict['right_eef_link_pose'].squeeze(0).cpu().detach().numpy()
            left_link_traj = self.act_dict['left_eef_link_pose'].squeeze(0).cpu().detach().numpy()
        else:
            combined = self.act_dict['eef_link_pose'].squeeze(0).cpu().detach().numpy()
            right_link_traj = combined[:, :15]
            left_link_traj = combined[:, 15:]
        vel_traj = self.act_dict['velocity_commands'].squeeze(0).cpu().detach().numpy()
        self.command_traj_dict = {
            'body_pose': body_pose_traj,
            'right_eef_to_body': right_eef_traj,
            'left_eef_to_body': left_eef_traj,
            'right_eef_link': right_link_traj,
            'left_eef_link': left_link_traj,
            'velocity_commands': vel_traj,
        }

    def _init_dream_accum(self, dream_dict):
        """Start a new accumulation window: store dream predictions and allocate GT buffers."""
        horizon = self.infer_interval
        accum = {}
        for key, val in dream_dict.items():
            accum[f'dream_{key}'] = val.squeeze(0).cpu().numpy()
        if self.inspire_hand_type == 'f1':
            accum['gt_left_tactile'] = np.zeros((horizon, 37), dtype=np.float32)
            accum['gt_right_tactile'] = np.zeros((horizon, 37), dtype=np.float32)
        else:
            accum['gt_left_tactile'] = np.zeros((horizon, 1062), dtype=np.float32)
            accum['gt_right_tactile'] = np.zeros((horizon, 1062), dtype=np.float32)
        accum['gt_left_force'] = np.zeros((horizon, 6), dtype=np.float32)
        accum['gt_right_force'] = np.zeros((horizon, 6), dtype=np.float32)
        accum['timestamps'] = np.zeros(horizon, dtype=np.float64)
        h, w = self.head_color_frame.shape[:2]
        accum['head_cam'] = np.zeros((horizon, h, w, 3), dtype=np.uint8)
        ema_teachers = self.bc_model.model.ema_teachers[self.embodiment] if self.embodiment in self.bc_model.model.ema_teachers else {}
        for key in self.dream_keys:
            if not self.dream_is_latent.get(key, False):
                continue
            if key not in ema_teachers or key not in _TACTILE_KEY_SLICE:
                continue
            latent_dim = dream_dict[key].shape[-1]
            accum[f'gt_latent_{key}'] = np.zeros((horizon, latent_dim), dtype=np.float32)
        accum['rollout_counter'] = self.rollout_counter
        self._dream_accum = accum

    def _record_dream_gt(self, step_idx):
        """Record current GT tactile, force, and head cam into the accumulation buffer."""
        if self.inspire_hand_type == 'f1':
            self._dream_accum['gt_left_tactile'][step_idx] = self.left_tactile_cap.copy()
            self._dream_accum['gt_right_tactile'][step_idx] = self.right_tactile_cap.copy()
        else:
            with self.touch_state_lock:
                lt = self.left_touch_raw.astype(np.float32) / 4095.0
                rt = self.right_touch_raw.astype(np.float32) / 4095.0
            self._dream_accum['gt_left_tactile'][step_idx] = lt
            self._dream_accum['gt_right_tactile'][step_idx] = rt
        with self.robot_eef_state_lock:
            eef = self.robot_eef_state
        if eef is not None:
            finger_forces = eef.get('finger_forces', np.zeros(12))
            self._dream_accum['gt_left_force'][step_idx] = np.array(finger_forces[6:12], dtype=np.float32)
            self._dream_accum['gt_right_force'][step_idx] = np.array(finger_forces[:6], dtype=np.float32)
        else:
            self._dream_accum['gt_left_force'][step_idx] = self.left_eef_finger_force_proprio
            self._dream_accum['gt_right_force'][step_idx] = self.right_eef_finger_force_proprio
        self._dream_accum['head_cam'][step_idx] = self.head_color_frame.copy()
        self._dream_accum['timestamps'][step_idx] = time.time()
        ema_teachers = self.bc_model.model.ema_teachers[self.embodiment] if self.embodiment in self.bc_model.model.ema_teachers else {}
        for key in self.dream_keys:
            if not self.dream_is_latent.get(key, False) or key not in ema_teachers or key not in _TACTILE_KEY_SLICE:
                continue
            if self.inspire_hand_type == 'f1':
                continue
            with self.touch_state_lock:
                lt_raw = self.left_touch_raw.astype(np.float32) / 4095.0
                rt_raw = self.right_touch_raw.astype(np.float32) / 4095.0
            hand, start, end = _TACTILE_KEY_SLICE[key]
            tactile = lt_raw[start:end] if hand == 'left' else rt_raw[start:end]
            with torch.no_grad():
                gt_tensor = torch.from_numpy(tactile).float().to(self.device).unsqueeze(0)
                gt_latent = ema_teachers[key](gt_tensor)
            self._dream_accum[f'gt_latent_{key}'][step_idx] = gt_latent.cpu().numpy().squeeze(0)

    def _flush_dream_accum(self):
        """Save the accumulated dream window to disk (synchronous). No-op if empty."""
        if self._dream_accum is None:
            return
        path = os.path.join(self.dream_touch_save_dir, f'step_{self.dream_save_counter:06d}.npz')
        np.savez(path, **self._dream_accum)
        self.dream_save_counter += 1
        self._dream_accum = None

    # ---- HIT observation / action helpers ----

    def _get_hit_proprio_value(self, obs_name):
        """Map an observation_name from HIT YAML to the corresponding runtime array."""
        mapping = {
            'proprioceptions/body_proprio': lambda: self.body_pose_proprio,
            'proprioceptions/eef_proprio': lambda: self.eef_pose_proprio,
            'proprioceptions/left_eef_to_body_proprio': lambda: self.eef_to_body_pose_proprio[6:12],
            'proprioceptions/right_eef_to_body_proprio': lambda: self.eef_to_body_pose_proprio[:6],
            'proprioceptions/left_eef_link_proprio': lambda: self.eef_link_proprio[15:30],
            'proprioceptions/right_eef_link_proprio': lambda: self.eef_link_proprio[:15],
            'proprioceptions/robot_joint_pos_proprio': lambda: self.robot_joint_pos_proprio,
            'proprioceptions/left_eef_joint_pos_proprio': lambda: self.left_eef_joint_pos_proprio,
            'proprioceptions/right_eef_joint_pos_proprio': lambda: self.right_eef_joint_pos_proprio,
            'proprioceptions/left_eef_finger_force_proprio': lambda: self.left_eef_finger_force_proprio,
            'proprioceptions/right_eef_finger_force_proprio': lambda: self.right_eef_finger_force_proprio,
            'proprioceptions/other/left_touch': lambda: self.left_tactile,
            'proprioceptions/other/right_touch': lambda: self.right_tactile,
        }
        fn = mapping.get(obs_name)
        if fn is None:
            raise ValueError(f"Unknown observation name for HIT rollout: {obs_name}")
        return fn()

    def _build_g1_hit_obs(self):
        """Build qpos tensor and image tensor for HIT inference on g1_inspire_wb."""
        # --- qpos ---
        obs_parts = [self._get_hit_proprio_value(name) for name in self.hit_observation_names]
        qpos_np = np.concatenate(obs_parts, axis=-1).astype(np.float32)
        qpos_np = self.hit_pre_process(qpos_np)
        qpos = torch.from_numpy(qpos_np).float().to(self.device).unsqueeze(0)

        # --- images ---
        main_frame = self.head_color_frame.copy()

        images = {}
        for cam_name in self.hit_camera_names:
            if cam_name == 'main_left':
                w_half = main_frame.shape[1] // 2
                img = main_frame[:, :w_half, :]
            elif cam_name == 'main_right':
                w_half = main_frame.shape[1] // 2
                img = main_frame[:, w_half:, :]
            elif cam_name == 'wrist_left':
                img = self.wrist_left_color_frame.copy()
            elif cam_name == 'wrist_right':
                img = self.wrist_right_color_frame.copy()
            else:
                raise ValueError(f"Unknown camera name: {cam_name}")
            if cam_name in ('main_left', 'main_right'):
                target_h, main_w = self.hit_main_cam_size
                target_w = main_w // 2
            else:
                target_h, target_w = self.hit_wrist_cam_size
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img_tensor = torch.from_numpy(img).float().div_(255.0)
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            images[cam_name] = img_tensor

        if len({tuple(img.shape) for img in images.values()}) == 1:
            curr_image = torch.stack(list(images.values()), dim=0).to(self.device).unsqueeze(0)
        else:
            curr_image = {
                name: img.to(self.device).unsqueeze(0) for name, img in images.items()
            }
        return qpos, curr_image

    def _process_g1_hit_action_trajectory(self, actions_np):
        """Split flat denormalized action array (chunk_size, action_dim) into trajectory dict.

        Action order matches action_name in HIT YAML:
          actions/body(6), actions/left_eef_to_body(6), actions/right_eef_to_body(6),
          actions/left_eef_link(15), actions/right_eef_link(15), actions/velocity_commands(3)
        """
        action_dim_map = {
            'actions/body': 6,
            'actions/left_eef_to_body': 6,
            'actions/right_eef_to_body': 6,
            'actions/left_eef_link': 15,
            'actions/right_eef_link': 15,
            'actions/velocity_commands': 3,
        }
        offset = 0
        parsed = {}
        for name in self.hit_action_names:
            dim = action_dim_map[name]
            parsed[name] = actions_np[:, offset:offset + dim]
            offset += dim

        self.command_traj_dict = {
            'body_pose': parsed['actions/body'],
            'right_eef_to_body': parsed.get('actions/right_eef_to_body', np.zeros((actions_np.shape[0], 6))),
            'left_eef_to_body': parsed.get('actions/left_eef_to_body', np.zeros((actions_np.shape[0], 6))),
            'right_eef_link': parsed.get('actions/right_eef_link', np.zeros((actions_np.shape[0], 15))),
            'left_eef_link': parsed.get('actions/left_eef_link', np.zeros((actions_np.shape[0], 15))),
            'velocity_commands': parsed.get('actions/velocity_commands', np.zeros((actions_np.shape[0], 3))),
        }

    def _eef_link_to_hand_joints(self, eef_link_15d):
        """Convert 15D fingertip positions (5 fingers x 3D) to 75D hand joint array."""
        hand_25x3 = np.zeros((25, 3))
        hand_25x3[4] = eef_link_15d[0:3]    # thumb
        hand_25x3[9] = eef_link_15d[3:6]    # index
        hand_25x3[14] = eef_link_15d[6:9]   # middle
        hand_25x3[19] = eef_link_15d[9:12]  # ring
        hand_25x3[24] = eef_link_15d[12:15] # pinky
        unified_to_inspire = self.inspire_to_unified_eef[:3, :3].T
        return (unified_to_inspire @ hand_25x3.T).T.flatten()

    def _execute_g1_action(self, rollout_idx):
        """Send g1_inspire_wb action at given trajectory index as DDS teleop command."""
        body_pose = self.command_traj_dict['body_pose'][rollout_idx]
        left_eef = self.command_traj_dict['left_eef_to_body'][rollout_idx]
        right_eef = self.command_traj_dict['right_eef_to_body'][rollout_idx]
        left_link = self.command_traj_dict['left_eef_link'][rollout_idx]
        right_link = self.command_traj_dict['right_eef_link'][rollout_idx]
        # torso
        torso_rot = body_pose[3:6].copy()
        torso_height = body_pose[2] + 0.75
        # right EEF: offsets + rotation remap (same as teleop in data_collect)
        right_pos = right_eef[:3].copy()
        right_pos[0] += 0.15
        right_pos[2] += 0.45
        right_rot = self.rpy_to_rot_mat(right_eef[3:6]) @ self.unified_to_unitree_right_eef_rot_remap
        right_rot_rpy = self.rot_mat_to_rpy(right_rot)
        # left EEF
        left_pos = left_eef[:3].copy()
        left_pos[0] += 0.15
        left_pos[2] += 0.45
        left_rot = self.rpy_to_rot_mat(left_eef[3:6]) @ self.unified_to_unitree_left_eef_rot_remap
        left_rot_rpy = self.rot_mat_to_rpy(left_rot)
        # hand joints from fingertip positions
        right_hand_joints = self._eef_link_to_hand_joints(right_link)
        left_hand_joints = self._eef_link_to_hand_joints(left_link)
        # velocity commands from inferred model
        vel_cmds = None
        if self.args.send_velocity_commands and 'velocity_commands' in self.command_traj_dict:
            vel_cmds = self.command_traj_dict['velocity_commands'][rollout_idx]
        self.publish_wb_teleop_data(
            left_pos, left_rot_rpy, right_pos, right_rot_rpy,
            left_hand_joints, right_hand_joints,
            torso_rot, torso_height, vel_cmds
        )

    def init_policy(self):
        self.policy_class = self.args.policy_class
        model_path = self.args.model_path
        self.has_dream_head = False

        if self.policy_class == 'MXT':
            self._init_mxt_policy(model_path)
        elif self.policy_class == 'HIT':
            self._init_hit_policy(model_path)
        else:
            raise NotImplementedError(f"Unsupported policy_class: {self.policy_class}")

    def _init_mxt_policy(self, model_path):
        with open(os.path.join(model_path, 'all_configs.json'), 'r') as f:
            saved_config = json.load(f)
        self.main_cam_size = _mxt_camera_size(saved_config, 'main')
        self.wrist_cam_size = _mxt_camera_size(saved_config, 'wrist')
        embodiment_config_path = self.args.embodiment_config_path
        transformer_trunk_config_path = self.args.trunk_config_path
        with open(embodiment_config_path, 'r') as f:
            emb_dict = _mxt_inference_config(yaml.load(f, Loader=yaml.FullLoader))
        with open(transformer_trunk_config_path, 'r') as f:
            transformer_dict = yaml.load(f, Loader=yaml.FullLoader)

        stats_path = os.path.join(model_path, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats_dict = pickle.load(f)
        share_cross_attn = False
        policy_config = {
            'norm_stats': stats_dict, 
            "embodiment_args_dict": emb_dict,
            "transformer_args": transformer_dict,
            "share_cross_attn": share_cross_attn,
            }
        policy = make_policy('MXT', policy_config)
        policy.deserialize(
            torch.load(
                f'{model_path}/policy_last.ckpt',
                map_location=self.device,
                weights_only=True,
            ),
            mode='strict',
        )
        policy.to(self.device)
        policy.eval()
        actual_device = next(policy.parameters()).device
        print(f'loaded MXT model from {model_path} on {actual_device}')
        self.bc_model = policy
        self.obs_keys = [k for k in emb_dict[self.embodiment]['tokenizer'].keys()
                         if 'proprio' in k or 'image' in k or 'tactile' in k]
        self.action_keys = list(emb_dict[self.embodiment]['action_head'].keys())
        print(f'Obs keys: {self.obs_keys}')
        print(f'Action keys: {self.action_keys}')

        dream_head_cfg = emb_dict[self.embodiment].get('dream_head', {})
        self.dream_keys = list(dream_head_cfg.keys())
        self.has_dream_head = len(self.dream_keys) > 0
        if self.args.dream_touch and self.has_dream_head:
            self.dream_is_latent = {
                k: ('tactile' in k and 'tactile_cap' not in k and not dream_head_cfg[k].get('dream_raw', False))
                for k in self.dream_keys
            }
            save_dir = self.args.dream_touch_save_dir or os.path.join(model_path, 'dream_touch_results')
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.dream_touch_save_dir = os.path.join(save_dir, ts)
            os.makedirs(self.dream_touch_save_dir, exist_ok=True)
            self.dream_save_counter = 0
            self._dream_accum = None
            print(f'Dream touch saving enabled -> {self.dream_touch_save_dir}')
            print(f'Dream keys: {self.dream_keys}')
            latent_keys = [k for k, v in self.dream_is_latent.items() if v]
            if latent_keys:
                print(f'Latent (EMA) dream keys (will encode GT): {latent_keys}')

    def _init_hit_policy(self, model_path):
        config_path = os.path.join(model_path, 'all_configs.json')
        with open(config_path, 'r') as f:
            all_config = json.load(f)
        policy_class = all_config['policy_class']
        policy_config = all_config['policy_config']

        stats_path = os.path.join(model_path, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats_dict = pickle.load(f)
        self.hit_pre_process = lambda s: (s - stats_dict['qpos_mean']) / stats_dict['qpos_std']
        self.hit_post_process = lambda a: a * stats_dict['action_std'] + stats_dict['action_mean']

        policy = make_policy(policy_class, policy_config)
        policy.load_state_dict(torch.load(f'{model_path}/policy_last.ckpt', map_location=self.device))
        policy.to(self.device)
        policy.eval()
        actual_device = next(policy.parameters()).device
        print(f'loaded HIT model from {model_path} on {actual_device}')
        self.bc_model = policy

        self.hit_data_config = getattr(self.args, 'hit_data_config', None)
        if self.hit_data_config is None:
            hit_config_path = self.args.hit_config_path
            if not hit_config_path and 'resolved_data_config' in all_config:
                self.hit_data_config = all_config['resolved_data_config']
            else:
                hit_config_path = hit_config_path or all_config.get('config', '')
                with open(hit_config_path, 'r') as f:
                    self.hit_data_config = yaml.safe_load(f)
        self.hit_camera_names = self.hit_data_config['camera_names']
        self.hit_observation_names = self.hit_data_config['observation_name']
        self.hit_action_names = self.hit_data_config['action_name']
        self.hit_main_cam_size, self.hit_wrist_cam_size = _hit_camera_sizes(all_config)
        print(f'HIT cameras: {self.hit_camera_names}')
        print(f'HIT observations: {self.hit_observation_names}')
        print(f'HIT actions: {self.hit_action_names}')
        print(f'HIT main image size (H, W): {self.hit_main_cam_size}; wrist: {self.hit_wrist_cam_size}')

    def on_press_key(self, key):
        try:
            if not self.on_reset:
                if key.char == '1':
                    self.send_reset_command()
                    self.on_reset = True
                    self.on_rollout = False
            if self.on_reset:
                if key.char == '2':
                    self.on_rollout = True
                    self.on_reset = False
            if self.on_rollout:
                if key.char == '3':
                    self.pause_commands = True
                    print('Paused commands. Press 2 to resume.')
            if self.on_rollout and self.pause_commands:
                if key.char == '2':
                    self.pause_commands = False
        except AttributeError:
            # Handle special keys (like function keys, arrow keys, etc.)
            pass  #

    def keyboard_listener_thread(self):
        with keyboard.Listener(on_press=self.on_press_key) as listener:
            listener.join()

    def run(self):
        """Main loop for policy rollout and teleoperation publishing"""
        print("HTD Policy Rollout with DDS Teleoperation")
        print(f"Control frequency: {self.args.control_freq} Hz")
        print(f"Embodiment: {self.args.embodiment}")
        
        head_camera_streaming_thread = threading.Thread(target=self.head_camera_stream_thread, daemon=True)
        head_camera_streaming_thread.start()
        if self.args.use_wrist_camera:
            threading.Thread(target=self.wrist_left_camera_stream_thread, daemon=True).start()
            threading.Thread(target=self.wrist_right_camera_stream_thread, daemon=True).start()

        keyboard_listener_thread = threading.Thread(target=self.keyboard_listener_thread, daemon=True)
        keyboard_listener_thread.start()

        # Main control loop
        loop_period = 1.0 / self.args.control_freq
        
        try:
            while self.running:
                loop_start = time.time()
                if self.on_reset:
                    self.rollout_counter = 0
                elif self.on_rollout:
                    if not self.pause_commands:
                        self.policy_rollout()
                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > loop_period * 1.5:
                    print(f"Warning: Control loop exceeded target rate: {elapsed:.4f}s > {loop_period:.4f}s")
                    
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, shutting down...")
            self.running = False

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

    def rpy_to_rot_mat(self, rpy):
        """
        Convert Euler angles (roll, pitch, yaw) with ZYX order to a rotation matrix.
        """
        x, y, z = rpy
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)

        R = np.array([
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy]
        ])
        return R

    def close(self):
        """Clean shutdown of data collector"""
        self.running = False

        if hasattr(self, '_dream_accum') and self._dream_accum is not None:
            self._flush_dream_accum()

        # Clean up DDS resources
        if hasattr(self, 'dds_participant'):
            try:
                # For cyclonedds, delete writer and topic first, then participant
                if hasattr(self, 'dds_writer_teleop'):
                    del self.dds_writer_teleop
                if hasattr(self, 'dds_topic_teleop'):
                    del self.dds_topic_teleop
                del self.dds_participant
                print("DDS resources cleaned up")
            except Exception as e:
                print(f"Error cleaning up DDS: {e}")
        
        # Clean up temporary config file
        if hasattr(self, '_dds_config_file') and os.path.exists(self._dds_config_file):
            try:
                if tempfile.gettempdir() in self._dds_config_file:
                    os.unlink(self._dds_config_file)
            except Exception:
                pass
        

def signal_handler(sig, frame):
    print('\npressed Ctrl+C! exiting...')
    # Force immediate exit - don't wait for cleanup
    import os
    os._exit(0)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _validate_policy_rollout_args(parser, args):
    """Validate policy artifacts and configs before initializing cameras or DDS."""
    if args.replay:
        parser.error(
            "--replay is not supported by the HTD g1_inspire_wb rollout; "
            "use live MXT or HIT policy inference"
        )

    if args.policy_class == 'HIT':
        policy_source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'htd_policy')
        config_aliases = {
            'dream_touch_scripts/hit/g1_inspire_wb_visual_proprio.yaml': 'visual.yaml',
            'dream_touch_scripts/hit/g1_inspire_wb_visual_proprio_touch.yaml': 'touch.yaml',
            'dream_touch_scripts/hit/g1_inspire_wb_visual_proprio_walk.yaml': 'visual_walk.yaml',
            'dream_touch_scripts/hit/g1_inspire_wb_visual_proprio_touch_walk.yaml': 'touch_walk.yaml',
        }
        if not isinstance(args.model_path, str) or not args.model_path:
            parser.error("--model_path is required for HIT policy rollout")

        artifact_config_path = os.path.join(args.model_path, 'all_configs.json')
        required_artifact_paths = [
            artifact_config_path,
            os.path.join(args.model_path, 'dataset_stats.pkl'),
            os.path.join(args.model_path, 'policy_last.ckpt'),
        ]
        missing_artifact_paths = [
            path for path in required_artifact_paths if not os.path.isfile(path)
        ]
        if missing_artifact_paths:
            parser.error(
                "HIT --model_path is missing required artifact files: "
                f"{missing_artifact_paths}"
            )
        try:
            with open(artifact_config_path, 'r') as f:
                artifact_config = json.load(f)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parser.error(f"invalid HIT artifact metadata {artifact_config_path}: {exc}")
        if not isinstance(artifact_config, dict):
            parser.error(
                f"HIT artifact metadata must be a JSON object: {artifact_config_path}"
            )
        if artifact_config.get('policy_class') != 'HIT':
            parser.error(
                "HIT artifact metadata must declare policy_class='HIT': "
                f"{artifact_config_path}"
            )
        policy_config = artifact_config.get('policy_config')
        if not isinstance(policy_config, dict):
            parser.error(
                f"HIT artifact policy_config must be a JSON object: "
                f"{artifact_config_path}"
            )
        try:
            _, wrist_size = _hit_camera_sizes(artifact_config)
        except ValueError as exc:
            parser.error(str(exc))
        for axis, recorded_size in zip(('height', 'width'), wrist_size):
            requested_size = getattr(args, f'wrist_cam_{axis}', None)
            if requested_size is not None and requested_size != recorded_size:
                parser.error(f'--wrist_cam_{axis} must match the recorded training size {recorded_size}')

        embedded_config = artifact_config.get('resolved_data_config')
        if 'resolved_data_config' in artifact_config and not isinstance(embedded_config, dict):
            parser.error('HIT resolved_data_config must be a JSON object')
        if not args.hit_config_path and embedded_config is not None:
            hit_config_path = '<embedded resolved_data_config>'
            hit_config = embedded_config
        else:
            hit_config_path = args.hit_config_path or artifact_config.get('config')
            if not isinstance(hit_config_path, str) or not hit_config_path:
                parser.error(
                    '--hit_config_path is required when the HIT artifact has neither '
                    'embedded data configuration nor a usable config path'
                )
            if not os.path.isfile(hit_config_path) and hit_config_path in config_aliases:
                hit_config_path = os.path.join(policy_source_dir, 'configs', 'act', config_aliases[hit_config_path])
            if not os.path.isfile(hit_config_path):
                parser.error(f'HIT config file not found: {hit_config_path}')
            try:
                with open(hit_config_path, 'r') as f:
                    hit_config = yaml.safe_load(f)
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                parser.error(f'invalid HIT config {hit_config_path}: {exc}')
            if not isinstance(hit_config, dict):
                parser.error(f'HIT config must be a YAML mapping: {hit_config_path}')

        camera_names = hit_config.get('camera_names')
        observation_names = hit_config.get('observation_name')
        action_names = hit_config.get('action_name')
        for field_name, values in (
            ('observation_name', observation_names),
            ('action_name', action_names),
        ):
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
            ):
                parser.error(
                    f"HIT {field_name} must be a nonempty list of unique strings: "
                    f"{hit_config_path}"
                )

        observation_dims = {
            'proprioceptions/body_proprio': 6,
            'proprioceptions/eef_proprio': 12,
            'proprioceptions/left_eef_to_body_proprio': 6,
            'proprioceptions/right_eef_to_body_proprio': 6,
            'proprioceptions/left_eef_link_proprio': 15,
            'proprioceptions/right_eef_link_proprio': 15,
            'proprioceptions/robot_joint_pos_proprio': 29,
            'proprioceptions/left_eef_joint_pos_proprio': 6,
            'proprioceptions/right_eef_joint_pos_proprio': 6,
            'proprioceptions/left_eef_finger_force_proprio': 6,
            'proprioceptions/right_eef_finger_force_proprio': 6,
            'proprioceptions/other/left_touch': 1062,
            'proprioceptions/other/right_touch': 1062,
        }
        action_dims = {
            'actions/body': 6,
            'actions/left_eef_to_body': 6,
            'actions/right_eef_to_body': 6,
            'actions/left_eef_link': 15,
            'actions/right_eef_link': 15,
            'actions/velocity_commands': 3,
        }
        unknown_observations = [
            name for name in observation_names if name not in observation_dims
        ]
        if (getattr(args, 'inspire_hand_type', 'dftp') == 'f1'
                and {'proprioceptions/other/left_touch', 'proprioceptions/other/right_touch'}.intersection(observation_names)):
            parser.error('HIT DFTP tactile observations require --inspire_hand_type dftp')
        unknown_actions = [name for name in action_names if name not in action_dims]
        if unknown_observations:
            parser.error(
                f"unsupported G1 HIT observation_name entries in {hit_config_path}: "
                f"{unknown_observations}"
            )
        if unknown_actions:
            parser.error(
                f"unsupported G1 HIT action_name entries in {hit_config_path}: "
                f"{unknown_actions}"
            )
        if 'actions/body' not in action_names:
            parser.error(
                "G1 HIT action_name must include actions/body because rollout "
                f"always publishes a body pose: {hit_config_path}"
            )

        allowed_camera_names = {
            'main_left', 'main_right', 'wrist_left', 'wrist_right'
        }
        if (
            not isinstance(camera_names, list)
            or not camera_names
            or any(not isinstance(name, str) for name in camera_names)
            or len(camera_names) != len(set(camera_names))
        ):
            parser.error(
                f"HIT camera_names must be a nonempty list of unique strings: "
                f"{hit_config_path}"
            )
        unsupported = [
            name for name in camera_names if name not in allowed_camera_names
        ]
        if unsupported:
            parser.error(
                f"unsupported G1 HIT camera_names in {hit_config_path}: {unsupported}"
            )
        if camera_names[:2] != ['main_left', 'main_right']:
            parser.error(
                "G1 HIT camera_names must start with main_left, main_right in "
                f"that order: {hit_config_path}"
            )

        artifact_camera_names = policy_config.get('camera_names')
        if camera_names != artifact_camera_names:
            parser.error(
                "HIT YAML camera_names do not match the checkpoint artifact "
                f"camera order: yaml={camera_names}, artifact={artifact_camera_names}"
            )

        backbone_map = hit_config.get('backbone_map')
        if (
            not isinstance(backbone_map, list)
            or len(backbone_map) != len(camera_names)
            or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in backbone_map
            )
        ):
            parser.error(
                f"HIT backbone_map must contain one integer per camera: "
                f"{hit_config_path}"
            )
        if backbone_map != policy_config.get('backbone_map'):
            parser.error(
                "HIT YAML backbone_map does not match the checkpoint artifact: "
                f"yaml={backbone_map}, artifact={policy_config.get('backbone_map')}"
            )

        expected_state_dim = sum(observation_dims[name] for name in observation_names)
        expected_action_dim = sum(action_dims[name] for name in action_names)
        if expected_state_dim != policy_config.get('state_dim'):
            parser.error(
                "HIT YAML observation contract does not match checkpoint state_dim: "
                f"yaml={expected_state_dim}, artifact={policy_config.get('state_dim')}"
            )
        if expected_action_dim != policy_config.get('action_dim'):
            parser.error(
                "HIT YAML action contract does not match checkpoint action_dim: "
                f"yaml={expected_action_dim}, artifact={policy_config.get('action_dim')}"
            )
        if hit_config.get('tactile_config') != policy_config.get('tactile_config'):
            parser.error(
                "HIT YAML tactile_config does not match the checkpoint artifact"
            )

        if embedded_config is not None:
            recorded_config = embedded_config
        else:
            recorded_config_ref = artifact_config.get('config')
            if not isinstance(recorded_config_ref, str) or not recorded_config_ref:
                parser.error(
                    f"HIT artifact does not record its training config path: "
                    f"{artifact_config_path}"
                )
            recorded_config_candidates = [recorded_config_ref]
            if not os.path.isabs(recorded_config_ref):
                recorded_config_candidates.extend([
                    os.path.join(
                        policy_source_dir, recorded_config_ref
                    ),
                    os.path.join(args.model_path, recorded_config_ref),
                ])
            if recorded_config_ref in config_aliases:
                recorded_config_candidates.append(os.path.join(policy_source_dir, 'configs', 'act', config_aliases[recorded_config_ref]))
            recorded_config_path = next(
                (path for path in recorded_config_candidates if os.path.isfile(path)),
                None,
            )
            if recorded_config_path is None:
                parser.error(
                    "HIT artifact's recorded training config is unavailable; bundle or "
                    f"restore it before rollout: {recorded_config_ref}"
                )
            try:
                with open(recorded_config_path, 'r') as f:
                    recorded_config = yaml.safe_load(f)
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                parser.error(f"invalid recorded HIT config {recorded_config_path}: {exc}")
            if not isinstance(recorded_config, dict):
                parser.error(
                    f"recorded HIT config must be a YAML mapping: {recorded_config_path}"
                )
        contract_fields = (
            'camera_names',
            'backbone_map',
            'observation_name',
            'action_name',
            'tactile_config',
        )
        mismatched_fields = [
            field
            for field in contract_fields
            if hit_config.get(field) != recorded_config.get(field)
        ]
        if mismatched_fields:
            parser.error(
                "selected HIT YAML does not match the artifact's recorded training "
                f"config for fields: {mismatched_fields}"
            )

        wrist_camera_names = [
            name for name in camera_names if name.startswith('wrist_')
        ]
        if wrist_camera_names and wrist_camera_names != ['wrist_left', 'wrist_right']:
            parser.error(
                "G1 HIT wrist cameras must be wrist_left, wrist_right in that "
                f"order: {hit_config_path}"
            )
        config_uses_wrist = bool(wrist_camera_names)
        if args.use_wrist_camera != config_uses_wrist:
            parser.error(
                "--use_wrist_camera must match the HIT config/checkpoint camera "
                f"contract; expected {config_uses_wrist} for {camera_names}"
            )

        args.hit_config_path = '' if embedded_config is not None and not args.hit_config_path else hit_config_path
        args.hit_data_config = hit_config
        return

    if args.policy_class != 'MXT':
        return

    if not isinstance(args.model_path, str) or not args.model_path:
        parser.error("--model_path is required for MXT policy rollout")

    artifact_config_path = os.path.join(args.model_path, 'all_configs.json')
    required_artifact_paths = [
        artifact_config_path,
        os.path.join(args.model_path, 'dataset_stats.pkl'),
        os.path.join(args.model_path, 'policy_last.ckpt'),
    ]
    missing_artifact_paths = [
        path for path in required_artifact_paths if not os.path.isfile(path)
    ]
    if missing_artifact_paths:
        parser.error(
            "MXT --model_path is missing required artifact files: "
            f"{missing_artifact_paths}"
        )
    try:
        with open(artifact_config_path, 'r') as f:
            artifact_config = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"invalid MXT artifact metadata {artifact_config_path}: {exc}")
    if not isinstance(artifact_config, dict):
        parser.error(
            f"MXT artifact metadata must be a JSON object: {artifact_config_path}"
        )
    if artifact_config.get('policy_class') != 'MXT':
        parser.error(
            "MXT artifact metadata must declare policy_class='MXT': "
            f"{artifact_config_path}"
        )
    policy_config = artifact_config.get('policy_config')
    if not isinstance(policy_config, dict):
        parser.error(
            f"MXT artifact policy_config must be a JSON object: "
            f"{artifact_config_path}"
        )

    try:
        _mxt_camera_size(artifact_config, 'main')
        wrist_size = _mxt_camera_size(artifact_config, 'wrist')
    except ValueError as exc:
        parser.error(str(exc))
    for axis, recorded_size in zip(('height', 'width'), wrist_size):
        override = getattr(args, f'wrist_cam_{axis}', None)
        if override is not None and override != recorded_size:
            parser.error(f'--wrist_cam_{axis} must match the recorded training size {recorded_size}')

    config_paths = (
        ('--embodiment_config_path', args.embodiment_config_path),
        ('--trunk_config_path', args.trunk_config_path),
    )
    missing = [flag for flag, value in config_paths if not value]
    if missing:
        parser.error(
            f"{' and '.join(missing)} required for MXT policy rollout; "
            "select the matching HTD/G1 config pair explicitly"
        )

    nonexistent = [
        f"{flag}={value}"
        for flag, value in config_paths
        if not os.path.isfile(value)
    ]
    if nonexistent:
        parser.error(f"MXT config file not found: {', '.join(nonexistent)}")

    parsed_configs = []
    for label, path in (
        ('embodiment', args.embodiment_config_path),
        ('trunk', args.trunk_config_path),
    ):
        try:
            with open(path, 'r') as f:
                parsed_config = yaml.safe_load(f)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            parser.error(f"invalid MXT {label} config {path}: {exc}")
        if not isinstance(parsed_config, dict):
            parser.error(f"MXT {label} config must be a YAML mapping: {path}")
        parsed_configs.append(parsed_config)

    embodiment_config, trunk_config = parsed_configs
    saved_embodiment_config = policy_config.get('embodiment_args_dict')
    saved_trunk_config = policy_config.get('transformer_args')
    if _mxt_inference_config(embodiment_config) != _mxt_inference_config(saved_embodiment_config):
        parser.error(
            "selected MXT embodiment YAML does not match the checkpoint artifact"
        )
    if trunk_config != saved_trunk_config:
        parser.error(
            "selected MXT trunk YAML does not match the checkpoint artifact"
        )

    g1_config = embodiment_config.get('g1_inspire_wb')
    if not isinstance(g1_config, dict):
        parser.error(
            "MXT embodiment config must define the g1_inspire_wb rollout contract"
        )
    chunk_size = artifact_config.get('chunk_size')
    for group in ('action_head', 'dream_head'):
        heads = g1_config.get(group, {})
        if not isinstance(heads, dict):
            parser.error(f'MXT g1_inspire_wb {group} must be a mapping')
        for name, head in heads.items():
            horizon = head.get('action_horizon') if isinstance(head, dict) else None
            if (not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
                    or (chunk_size is not None and horizon != chunk_size)):
                parser.error(f'MXT {group}.{name} action_horizon must be positive and match checkpoint chunk_size')

    tokenizer_config = g1_config.get('tokenizer')
    if not isinstance(tokenizer_config, dict):
        parser.error(
            "MXT g1_inspire_wb config must define a tokenizer mapping"
        )

    wrist_keys = {'left_wrist_image', 'right_wrist_image'}
    configured_wrist_keys = wrist_keys.intersection(tokenizer_config)
    if configured_wrist_keys and configured_wrist_keys != wrist_keys:
        parser.error(
            "MXT wrist-camera tokenizer contract must include both "
            "left_wrist_image and right_wrist_image"
        )
    config_uses_wrist = configured_wrist_keys == wrist_keys
    if args.use_wrist_camera != config_uses_wrist:
        parser.error(
            "--use_wrist_camera must match the MXT config/checkpoint tokenizer "
            f"contract; expected {config_uses_wrist}"
        )

    tactile_keys = {
        key for key in tokenizer_config if isinstance(key, str) and 'tactile' in key
    }
    uses_f1_touch = any('tactile_cap' in key for key in tactile_keys)
    uses_dftp_touch = any('tactile_cap' not in key for key in tactile_keys)
    if uses_f1_touch and uses_dftp_touch:
        parser.error(
            "MXT tokenizer contract cannot mix F1 tactile_cap and DFTP tactile inputs"
        )
    expected_hand_type = (
        'f1' if uses_f1_touch else 'dftp' if uses_dftp_touch else None
    )
    if expected_hand_type is not None and args.inspire_hand_type != expected_hand_type:
        parser.error(
            "--inspire_hand_type must match the MXT config/checkpoint tokenizer "
            f"contract; expected {expected_hand_type!r}"
        )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(prog="HTD G1 whole-body policy rollout")
    parser.add_argument('--embodiment', type=str, default="g1_inspire_wb", choices=["g1_inspire_wb"], help='HTD policy embodiment')
    parser.add_argument("--head_camera_res", type=str, default="480p", choices=["480p", "720p", "1080p"], help="head camera resolution: 480p, 720p, or 1080p")
    parser.add_argument("--use_wrist_camera", type=str2bool, default=False, help="whether to use wrist camera")
    parser.add_argument("--wrist_camera_res", type=str, default="480p", choices=["480p", "720p", "1080p"], help="wrist camera resolution: 480p, 720p, or 1080p")
    parser.add_argument("--desired_stream_fps", type=int, default=60, help="desired camera streaming fps")
    parser.add_argument("--control_freq", type=int, default=30, help="policy rollout control frequency")
    # exp_name
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument('--dds_interface', type=str, default=None,
                       help='Network interface name to use for DDS (e.g., enp4s0, wlxc83a35a4f24d)')
    parser.add_argument("--device", type=str, default='cuda:0', help="device to use for the model")
    parser.add_argument("--embodiment_config_path", type=str, default=None, help="path to the embodiment config YAML (required for MXT policy rollout)")
    parser.add_argument("--trunk_config_path", type=str, default=None, help="path to the transformer trunk YAML paired with the embodiment config (required for MXT policy rollout)")
    parser.add_argument("--model_path", type=str, default='', help="path to the model")
    parser.add_argument("--replay", type=str2bool, default=False, help="if replay a trajectory")
    parser.add_argument("--inference_interval", type=int, default=60, help="model inference interval within the control loop (coupled with the control frequency)") #
    parser.add_argument("--wrist_cam_width", type=int, default=None, help="optional wrist width check (input size is read from the training metadata)")
    parser.add_argument("--wrist_cam_height", type=int, default=None, help="optional wrist height check (input size is read from the training metadata)")
    parser.add_argument("--policy_class", type=str, default='MXT', choices=['MXT', 'HIT'], help="policy class: MXT or HIT")
    parser.add_argument("--hit_config_path", type=str, default='', help="path to HIT data config YAML (observation/action/camera names)")
    parser.add_argument("--dream_touch", type=str2bool, default=False, help="save dream touch predictions, GT tactile/force, and head cam during MXT rollout")
    parser.add_argument("--dream_touch_save_dir", type=str, default='', help="directory for dream touch results (default: dream_touch_results/ under model_path)")
    parser.add_argument("--send_velocity_commands", type=str2bool, default=False, help="send inferred velocity commands to the robot for locomotion control")
    parser.add_argument("--inspire_hand_type", type=str, default='dftp', choices=['dftp', 'f1'], help="Inspire hand type: dftp (default) or f1 (capacitive)")

    args = parser.parse_args()
    _validate_policy_rollout_args(parser, args)
    
    # Create data collector
    htd_policy_rollout = HTDPolicyRollout(args)
    try:
        htd_policy_rollout.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        htd_policy_rollout.close()
        print("Program exited")
        
