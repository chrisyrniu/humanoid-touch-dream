import argparse
import os
import time
import numpy as np
import cv2
import threading
import queue
from camera_utils import list_video_devices, find_device_path_by_name
from multiprocessing import shared_memory
import sys
import signal
import h5py
from datetime import datetime
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import gc
import tempfile

# DDS imports for synchronized teleoperation publishing
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter

# Add robot_control directory to path to import teleop_dds_msgs
robot_control_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'robot_control')
sys.path.insert(0, robot_control_dir)
from teleop_dds_msgs import TeleopData, TeleopDataWB


INSPIRE_TOUCH_FIELD_SIZES = {
    "fingerone_tip_touch": 9, # (3, 3)
    "fingerone_top_touch": 96, # (12, 8)
    "fingerone_palm_touch": 80, # (10, 8)
    "fingertwo_tip_touch": 9, # (3, 3)
    "fingertwo_top_touch": 96, # (12, 8)
    "fingertwo_palm_touch": 80, # (10, 8)
    "fingerthree_tip_touch": 9, # (3, 3)
    "fingerthree_top_touch": 96, # (12, 8)
    "fingerthree_palm_touch": 80, # (10, 8)
    "fingerfour_tip_touch": 9, # (3, 3)
    "fingerfour_top_touch": 96, # (12, 8)
    "fingerfour_palm_touch": 80, # (10, 8)
    "fingerfive_tip_touch": 9, # (3, 3)
    "fingerfive_top_touch": 96, # (12, 8)
    "fingerfive_middle_touch": 9, # (3, 3)
    "fingerfive_palm_touch": 96, # (12, 8)
    "palm_touch": 112, # (14, 8)
}
INSPIRE_TOUCH_DIM_PER_HAND = sum(INSPIRE_TOUCH_FIELD_SIZES.values())

INSPIRE_F1_TOUCH_RAW_FIELDS = [
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
INSPIRE_F1_TOUCH_RAW_DIM = len(INSPIRE_F1_TOUCH_RAW_FIELDS)  # 34
# After processing: 5 fingers × (normal_N, tangential_N, direction_rad, proximity_norm) = 20
#                 + 3 palms   × (normal_N, tangential_N, direction_rad)                  =  9
INSPIRE_F1_TOUCH_DIM_PER_HAND = 29

_F1_TOUCH_NO_CONTACT_DIR = -1.0
_F1_TOUCH_PROXIMITY_MAX = 16_000_000.0

def _process_f1_touch_raw(raw):
    """Convert 34 raw uint16 touch values to 29 processed float values.

    Per-finger (×5):  normal (N), tangential (N), direction (rad), proximity [0,1]
    Per-palm   (×3):  normal (N), tangential (N), direction (rad)
    """
    r = raw.view(np.uint16)
    out = np.zeros(INSPIRE_F1_TOUCH_DIM_PER_HAND, dtype=np.float32)
    o = 0
    for i in range(5):
        base = i * 5
        out[o]     = float(r[base])     * 0.01
        out[o + 1] = float(r[base + 1]) * 0.01
        d = int(r[base + 2])
        out[o + 2] = _F1_TOUCH_NO_CONTACT_DIR if d == 65535 else d * (np.pi / 180.0)
        prox = int(r[base + 3]) | (int(r[base + 4]) << 16)
        out[o + 3] = min(prox / _F1_TOUCH_PROXIMITY_MAX, 1.0)
        o += 4
    for j in range(3):
        base = 25 + j * 3
        out[o]     = float(r[base])     * 0.01
        out[o + 1] = float(r[base + 1]) * 0.01
        d = int(r[base + 2])
        out[o + 2] = _F1_TOUCH_NO_CONTACT_DIR if d == 65535 else d * (np.pi / 180.0)
        o += 3
    return out


class HTDDataCollector:
    def __init__(self, args):
        self.args = args
        self.running = True  # Flag to control main loop
        self.checkpt = 0
        self.manipulate_eef_idx = [0]
        self.update_manipulate_eef_idx()
        self.embodiment = self.args.embodiment
        self.use_dual_wrist_camera = False
        self.inspire_hand_type = getattr(self.args, 'inspire_hand_type', 'dftp')
        self.enable_inspire_touch_collection = (
            self.args.collect_data
            and self.args.collect_inspire_touch
            and self.embodiment == 'g1_inspire_wb'
        )
        
        self.init_cameras()
        self.init_vr_streamer()
        
        # command buffer (kept for compatibility, but not published via ROS2)
        self.command = np.zeros(20)  # body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        # eef 6d pose in two types of defined unified frame
        self.eef_uni_command = np.zeros((2, 6))

        # initial values
        self.initial_receive = True
        self.init_body_pos = np.zeros(3)
        self.init_body_rot = np.eye(3)
        self.init_right_eef_pos = np.zeros(3)
        self.init_right_eef_rot = np.eye(3)
        self.init_left_eef_pos = np.zeros(3)
        self.init_left_eef_rot = np.eye(3)
        self.init_gripper_angles = np.zeros(2)

        self.is_changing_reveive_status = False
        
        self.on_reset = False
        self.on_collect = False
        self.on_save_data = False
        
        # self.pinch_gripper_angle_scale = 10.0
        self.pinch_dist_gripper_full_close = 0.02
        self.pinch_dist_gripper_full_open = 0.15
        # self.gripper_full_close_angle = 0.33
        self.gripper_full_close_angle = 0.01
        self.gripper_full_open_angle = 1.8
        self.eef_xyz_scale = 1.0
        
        # Rotation from the Apple Vision Pro world frame to the unified physical world frame.
        # equivalent to: first rotate along x axis with 90 degrees, then rotate along z axis with -90 degrees
        
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])

        # cancel the transformation applied by avp_stream to make data consistent with the original avp data
        self.z_up_inv = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
  
        if self.inspire_hand_type == 'f1':
            self.inspire_to_unified_eef_local = np.array([
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)
        else:
            self.inspire_to_unified_eef_local = np.array([
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)

        self.unified_to_unitree_left_eef_rot_remap = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float64)
        self.unified_to_unitree_right_eef_rot_remap = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)

        self.last_reset_time = 0.0
        self.last_calib_time = 0.0
        
        self.pause_commands = False

        self.record_episode_counter = 0
        
        # Background data saving (no queue needed - direct monitoring)
        self.trajectory_lock = threading.Lock()
        if self.args.collect_data:
            self.save_thread = threading.Thread(target=self.background_save_worker, daemon=True)
            self.save_thread.start()
            self.saving_in_progress = False  # Track if background thread is actively saving
        
        self.reset_trajectories()
        self.init_embodiment_states()
        
        # Initialize previous transformation matrices for safe updates
        self.prev_left_hand_data = np.eye(4)
        self.prev_right_hand_data = np.eye(4)
        
        # Set up network interface if specified
        if self.args.dds_interface is not None:
            self._setup_dds_interface(self.args.dds_interface)
        
        # Initialize DDS teleoperation publisher
        self.init_dds_publisher()
        self.prev_head_data = np.eye(4)
        
        self.init_dds_reset_command_publisher()
        self.init_dds_controller_reset_subscriber()
        if self.embodiment == 'g1_inspire_wb':
            # State containers and their locks must exist before the
            # subscription threads start touching them.
            self.robot_state = None
            self.robot_state_lock = threading.Lock()  # Thread-safe access
            self.init_dds_robot_state_subscriber()
            self.robot_eef_state = None
            self.robot_eef_state_lock = threading.Lock()  # Thread-safe access
            self.init_dds_eef_state_subscriber()
        if self.enable_inspire_touch_collection:
            # Subscribe touch data directly from Inspire hand touch DDS topics.
            if self.inspire_hand_type == 'f1':
                raw_dim = INSPIRE_F1_TOUCH_RAW_DIM
                touch_dtype = np.uint16
            else:
                raw_dim = INSPIRE_TOUCH_DIM_PER_HAND
                touch_dtype = np.int16
            self.left_touch_raw = np.zeros(raw_dim, dtype=touch_dtype)
            self.right_touch_raw = np.zeros(raw_dim, dtype=touch_dtype)
            self.touch_state_lock = threading.Lock()
            self.left_touch_received = False
            self.right_touch_received = False
            self.init_dds_touch_state_subscriber()

        # data collection
        if self.args.collect_data:
            self.data_folder = 'demonstrations'
            self.exp_name = self.args.exp_name
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_data_folder = self.data_folder + '/' + self.exp_name + f'/{self.args.embodiment}/' + current_time
            if not os.path.exists(self.exp_data_folder):
                os.makedirs(self.exp_data_folder)

    def init_vr_streamer(self):
        if self.args.see_through_mode:
            self.init_avp_stream()
        else:
            self.init_tele_vision()

    def cam_frame_callback(self, frame):
        """
        Frame callback for camera images - provides frames from head_color_frame for streaming.
        This is called by VisionProStreamer when device=None to get frames.
        head_color_frame is in RGB format, stereo side-by-side (height, 2*width, 3).
        """
        # Check if we have a valid camera frame
        if not hasattr(self, 'head_color_frame') or self.head_color_frame is None:
            # Return blank frame if no camera frame available yet
            # Size should match the requested streaming size: 1280x480 for stereo
            return np.zeros((self.head_camera_resolution[0], 2*self.head_camera_resolution[1], 3), dtype=np.uint8)
        
        # Check if frame is all zeros (not initialized yet or no data received)
        if np.all(self.head_color_frame == 0):
            return np.zeros((self.head_camera_resolution[0], 2*self.head_camera_resolution[1], 3), dtype=np.uint8)
        
        # head_color_frame is in RGB format, already stereo side-by-side
        # Convert RGB to BGR for avp_stream (expects BGR format)
        bgr_frame = cv2.cvtColor(self.head_color_frame, cv2.COLOR_RGB2BGR)
        
        # Resize to match requested streaming size (1280x480) if needed
        # The frame might be 720p (720 height) but streaming requests 480p (480 height)
        target_height, target_width = self.head_camera_resolution[0], 2*self.head_camera_resolution[1]
        if bgr_frame.shape[:2] != (target_height, target_width):
            bgr_frame = cv2.resize(bgr_frame, (target_width, target_height))
        
        return bgr_frame
    
    def init_avp_stream(self):
        from avp_stream import VisionProStreamer
        self.vr_streamer = VisionProStreamer(ip=self.args.see_through_mode_ip, record=False)
        if self.args.see_through_video:
            # Search for a valid video device
            video_device = self.head_camera_device_path

            # Try to start video streaming, continue gracefully if it fails
            try:
                self.vr_streamer.register_frame_callback(self.cam_frame_callback)
                self.vr_streamer.configure_video(
                    device=None,
                    format=None,
                    size=f"{2*self.head_camera_resolution[1]}x{self.head_camera_resolution[0]}",
                    fps=30,
                    stereo=True
                )
                self.vr_streamer.start_webrtc(port=9999)
                print("✓ Video streaming started successfully")
            except Exception as e:
                print(f"Failed to start video streaming: {e}")
                print("  Continuing without video streaming...")
                import traceback
                traceback.print_exc()

    def init_tele_vision(self):
        from tele_vision import OpenTeleVision
        if self.args.head_camera_type == 0:
            img_shape = (self.head_frame_res[0], self.head_frame_res[1], 3)
            self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize, name="pAlad1n3traiT")
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
            self.vr_streamer = OpenTeleVision(self.head_frame_res, self.shm.name, False)
        elif self.args.head_camera_type == 1:
            img_shape = (self.head_frame_res[0], 2 * self.head_frame_res[1], 3)
            self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
            self.vr_streamer = OpenTeleVision(self.head_frame_res, self.shm.name, True)
        else:
            raise NotImplementedError("Not supported camera.")

    def reset_pov_transform(self):
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])

    def get_gripper_angle_from_pinch_dist(self, pinch_dist):
        scale = (self.gripper_full_open_angle - self.gripper_full_close_angle) / (self.pinch_dist_gripper_full_open - self.pinch_dist_gripper_full_close)
        angle = (pinch_dist - self.pinch_dist_gripper_full_close) * scale + self.gripper_full_close_angle
        return np.clip(angle, self.gripper_full_close_angle, self.gripper_full_open_angle)
    
    def update_manipulate_eef_idx(self):
        if self.args.manipulate_mode == 1:
            # right manipulation
            self.manipulate_eef_idx = [0]
        elif self.args.manipulate_mode == 2:
            # left manipulation
            self.manipulate_eef_idx = [1]
        elif self.args.manipulate_mode == 3:
            # left manipulation
            self.manipulate_eef_idx = [0, 1]



    def init_cameras(self):
        # initialize all cameras
        self.desired_stream_fps = self.args.desired_stream_fps
        
        # Resolution mapping for different camera types
        resolution_map = {
            "480p": (480, 640),
            "720p": (720, 1280), 
            "1080p": (1080, 1920)
        }
        
        # initialize head camera
        # realsense as head camera
        self.head_cam_time = time.time()
        if self.args.head_camera_type == 0:
            import pyrealsense2 as rs
            self.head_camera_resolution = resolution_map[self.args.head_camera_res]
            self.head_frame_res = resolution_map[self.args.head_camera_res]
            self.head_color_frame = np.zeros((self.head_frame_res[0], self.head_frame_res[1], 3), dtype=np.uint8)
            self.head_cam_pipeline = rs.pipeline()
            self.head_cam_config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.head_cam_pipeline)
            pipeline_profile = self.head_cam_config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("a head camera is required")
                exit(0)
            self.head_cam_config.enable_stream(rs.stream.color, self.head_camera_resolution[1], self.head_camera_resolution[0], rs.format.bgr8, 30)
            # start streaming head cam
            self.head_cam_pipeline.start(self.head_cam_config)
        # stereo rgb camera (dual lens) as the head camera 
        elif self.args.head_camera_type == 1:
            # self.head_camera_resolution: the resoltion read from the camera, please note 480p will crop the original image and change the fov of the camera 
            # our dual-lens camera supports 1080p@60fps
            # 720p and 1080p have the same fov
            # use the following command to check the supported resolutions and fps: v4l2-ctl -d /dev/video0 --list-formats-ext
            self.head_camera_resolution = resolution_map[self.args.head_camera_res]
            # self.head_view_resolution: the resolution of the images that are seen and recorded, no cropping here
            self.head_view_resolution = resolution_map[self.args.head_camera_res]
            self.crop_size_w = 0
            self.crop_size_h = 0
            self.head_frame_res = (self.head_view_resolution[0] - self.crop_size_h, self.head_view_resolution[1] - 2 * self.crop_size_w)
            self.head_color_frame = np.zeros((self.head_frame_res[0], 2 * self.head_frame_res[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            head_camera_name = "3D USB Camera"
            device_path = find_device_path_by_name(device_map, head_camera_name)
            # Store device path for reuse in see-through mode
            self.head_camera_device_path = device_path
            self.head_cap = cv2.VideoCapture(device_path)
            self.head_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.head_camera_resolution[1])
            self.head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.head_camera_resolution[0])
            self.head_cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
        else:
            raise NotImplementedError("Not supported camera.")
        
        # initialize wrist camera/cameras
        if self.args.use_wrist_camera:
            self.wrist_camera_resolution = resolution_map[self.args.wrist_camera_res]
            self.wrist_view_resolution = resolution_map[self.args.wrist_camera_res]
            self.use_dual_wrist_camera = self.args.manipulate_mode == 3
            self.wrist_color_frame = np.zeros((self.wrist_view_resolution[0], self.wrist_view_resolution[1], 3), dtype=np.uint8)
            self.wrist_left_color_frame = np.zeros_like(self.wrist_color_frame)
            self.wrist_right_color_frame = np.zeros_like(self.wrist_color_frame)

            device_map = list_video_devices()
            wrist_camera_name = "Global Shutter Camera"
            wrist_device_paths = self._get_device_paths_by_name(device_map, wrist_camera_name)
            if len(wrist_device_paths) == 0:
                raise RuntimeError("No wrist camera devices found")

            if self.use_dual_wrist_camera:
                if len(wrist_device_paths) < 2:
                    raise RuntimeError(f"Expected 2 wrist cameras for bimanual mode, found {len(wrist_device_paths)}")
                left_path, right_path = self._select_wrist_cameras(wrist_device_paths)
                self.wrist_cap_left = self._create_wrist_capture(left_path)
                self.wrist_cap_right = self._create_wrist_capture(right_path)
                print(f"[wrist] Assigned left: {left_path}")
                print(f"[wrist] Assigned right: {right_path}")
            else:
                if len(wrist_device_paths) > 1:
                    print(f"[wrist] Multiple wrist cameras found, using first: {wrist_device_paths[0]}")
                self.wrist_cap1 = self._create_wrist_capture(wrist_device_paths[0])

    def _get_device_paths_by_name(self, device_map, name):
        device_paths = []
        for device_name, paths in device_map.items():
            if name in device_name:
                device_paths.append(paths[0])
        return device_paths

    def _create_wrist_capture(self, device_path):
        cap = cv2.VideoCapture(device_path)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.wrist_camera_resolution[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.wrist_camera_resolution[0])
        cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
        return cap

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

    def prompt_rendering(self):
        viewer_img = self.head_color_frame
        height = viewer_img.shape[0]
        width = viewer_img.shape[1]
        if self.on_save_data:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=height/8)
            drawer.text((width * 0.3 - 250, (height - 80) / 2), "SAVING DATA", font=font, fill=(10, 255, 10))
            viewer_img = np.array(im)
        elif self.on_reset:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=53)
            drawer.text((width * 0.3 - 250, (height - 80) / 2), "PINCH to START", font=font, fill=(255, 63, 63))
            # drawer.text((767, 200), "PINCH to START", font=font, fill=(255, 63, 63))
            viewer_img = np.array(im)
        elif self.on_collect:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=height/30)
            if len(self.manipulate_eef_idx) == 1:
                drawer.text((width / 10, (height - 80) / 4), '{:.2f}, {:.2f}, {:.2f} / {:.2f}, {:.2f}, {:.2f}'.format(
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+1],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+2], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+3],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+4], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+5]
                        ), font=font, fill=(255, 63, 63))
            elif len(self.manipulate_eef_idx) == 2:
                drawer.text((width / 10, (height - 80) / 4), '{:.2f}, {:.2f}, {:.2f} / {:.2f}, {:.2f}, {:.2f}'.format(
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]+1],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]+2], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+1], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+2]
                        ), font=font, fill=(255, 63, 63))
            viewer_img = np.array(im)
        return viewer_img

    def head_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        if self.args.head_camera_type == 0:
            try:
                while self.running:
                    start_time = time.time()
                    # handle head camera (realsense) streaming 
                    frames = self.head_cam_pipeline.wait_for_frames()
                    # depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        return
                    head_color_frame = np.asanyarray(color_frame.get_data())
                    head_color_frame = cv2.resize(head_color_frame, (self.head_frame_res[1], self.head_frame_res[0]))
                    self.head_color_frame = cv2.cvtColor(head_color_frame, cv2.COLOR_BGR2RGB)
                    if not self.args.see_through_mode:
                        np.copyto(self.img_array, self.head_color_frame)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
            finally:
                self.head_cam_pipeline.stop()
        elif self.args.head_camera_type == 1:
            try:
                while self.running:
                    start_time = time.time()
                    ret, frame = self.head_cap.read()
                    if self.args.embodiment == 'g1_inspire_wb':
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    self.head_cam_time = time.time()
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
                    viewer_img = self.prompt_rendering()
                    if not self.args.see_through_mode:
                        np.copyto(self.img_array, viewer_img)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                self.head_cap.release()
        else:
            raise NotImplementedError('Not supported camera.')
        
    def wrist_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while self.running:
                start_time = time.time()
                ret, frame = self.wrist_cap1.read()
                if ret and frame is not None:
                    wrist_color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.wrist_color_frame = cv2.resize(wrist_color_frame, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # print(1/(time.time() - start_time))
        finally:
            self.wrist_cap1.release()

    def wrist_camera_left_stream_thread(self):
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

    def wrist_camera_right_stream_thread(self):
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

        
    def collect_data(self, event=None):
        # publish command (if applicable) and collect data of the embodiment
        # publish teleop commands at a fixed rate
        # need to check the duration to finish one execution and if a separate thread is needed: takes 0.0002-0.0003s
        self.update_manipulate_eef_idx()
        # Tracking Streamer receives its first packet asynchronously.
        if self.args.see_through_mode and getattr(self.vr_streamer, 'latest', None) is None:
            return
        self.process_vr_data()

        # reset (finger pinch)
        if self.left_pinch_dist1_vr < 0.008 and self.left_pinch_dist0_vr > 0.05 and self.left_pinch_dist2_vr > 0.05 and self.left_pinch_dist3_vr > 0.05 and time.time() - self.last_reset_time > 3.0:
            self.on_reset = True
            self.on_collect = False
            self.last_reset_time = time.time()
            self.initial_receive = True
            self.reset_pov_transform()
            # Background thread will automatically handle episode completion and reset trajectories
            # Don't reset trajectories here - let background thread handle it after saving
            print("reset")

            if self.embodiment == 'g1_inspire_wb':
                self.send_reset_command()

        # reset (remote controller button [2] — robot already resetting itself)
        with self.controller_reset_lock:
            controller_wants_reset = self.controller_reset_flag
            self.controller_reset_flag = False
        if controller_wants_reset and not self.on_reset and time.time() - self.last_reset_time > 3.0:
            self.on_reset = True
            self.on_collect = False
            self.last_reset_time = time.time()
            self.initial_receive = True
            self.reset_pov_transform()
            print("reset (remote controller button [2])")
        # initialize and calibrate
        if self.on_reset:
            # the initialization only begins at a certain signal (e.g., a gesture)
            # the gesture is temporarily designed for single hand manipulation
            if self.left_pinch_dist0_vr < 0.008 and self.left_pinch_dist1_vr > 0.05 and self.left_pinch_dist2_vr > 0.05 and self.left_pinch_dist3_vr > 0.05:
                if self.args.collect_data and self.saving_in_progress:
                    print("Waiting for previous episode to finish saving...")
                    return
                self.reset_embodiment_states()
                if self.embodiment == 'human':
                    # Human proprio is derived from the calibrated target
                    # state, so calibration must happen first.
                    self.update_embodiment_target_states(reset_uni_frame=True)
                    if not self.update_embodiment_proprio_states():
                        return
                else:
                    # Robot proprio comes from DDS.  Wait for every required
                    # sample before calibrating or recording the first frame.
                    if not self.update_embodiment_proprio_states():
                        return
                    self.update_embodiment_target_states(reset_uni_frame=True)
                self.on_reset = False
                self.on_collect = True

                if self.args.collect_data:
                    self.update_trajectories()
            else:
                return
        # start collect
        elif self.on_collect:
            if self.left_pinch_dist3_vr < 0.015 and self.left_pinch_dist0_vr > 0.05 and self.left_pinch_dist1_vr > 0.05 and self.left_pinch_dist2_vr > 0.05:
                self.pause_commands = True
                print("Pause sending commands")
                return
            if self.pause_commands and self.left_pinch_dist0_vr < 0.008 and self.left_pinch_dist1_vr > 0.05 and self.left_pinch_dist2_vr > 0.05 and self.left_pinch_dist3_vr > 0.05:
                print("Restart sending commands")
                self.pause_commands = False
            if self.pause_commands:
                return
            self.reset_embodiment_states()
            self.update_embodiment_target_states(reset_uni_frame=False)
            if not self.update_embodiment_proprio_states():
                return
            if self.args.collect_data:
                self.update_trajectories()
    
    def init_embodiment_states(self):
        # target states
        self.body_pose_uni = np.zeros(6)
        self.delta_body_pose_uni = np.zeros(6)
        self.eef_pose_uni = np.zeros(12)
        self.delta_eef_pose_uni = np.zeros(12)
        self.eef_to_body_pose = np.zeros(12)
        self.gripper_angle = np.zeros(2)
        self.delta_gripper_angle = np.zeros(2)
        self.right_hand_joints = np.zeros(75) # (25, 3)
        self.left_hand_joints = np.zeros(75) # (25, 3)
        # camera pose relative to its initial pose in the unified frame
        self.head_camera_to_init_pose_uni = np.zeros(6)
        self.eef_joints_target_pose = np.zeros(150) # 75*2 to account for maximum number of hand joints from both human hands

        # proprio states
        self.body_pose_proprio = np.zeros(6)
        self.eef_pose_proprio = np.zeros(12)
        self.eef_to_body_pose_proprio = np.zeros(12)
        self.gripper_angle_proprio = np.zeros(2)
        if self.embodiment == 'human':
            self.eef_link_proprio = np.zeros(150)
        elif self.embodiment == 'g1_inspire_wb':
            self.robot_joint_proprio = np.zeros(58) # robot joint positions + joint velocities
            self.eef_joint_proprio = np.zeros(24) # 6*4, inspire hand joint positions + finger forces
            self.eef_link_proprio = np.zeros(30) # 3d pose* 5 finger tips * 2 hands
            self.velocity_commands = np.zeros(3) # 3 floats (lin_vel_x, lin_vel_y, ang_vel_z)
            self.imu_proprio = np.zeros(9) # 3d accelerometer + 3d gyroscope + 3d rpy
            if self.inspire_hand_type == 'f1':
                self.left_touch = np.zeros(INSPIRE_F1_TOUCH_DIM_PER_HAND, dtype=np.float32)
                self.right_touch = np.zeros(INSPIRE_F1_TOUCH_DIM_PER_HAND, dtype=np.float32)
            else:
                self.left_touch = np.zeros(INSPIRE_TOUCH_DIM_PER_HAND, dtype=np.float32)
                self.right_touch = np.zeros(INSPIRE_TOUCH_DIM_PER_HAND, dtype=np.float32)
        else:
            self.eef_link_proprio = np.zeros(150)

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
        
        self.dds_participant = DomainParticipant(0)
        kTopicTeleopData = "rt/teleop/synchronized_data"
        if self.embodiment == 'g1_inspire_wb':
            self.teleop_data_type = TeleopDataWB
            print("Using TeleopDataWB for whole-body teleoperation")
        else:
            self.teleop_data_type = TeleopData
            print("Using TeleopData for standard teleoperation")
        self.dds_topic_teleop = Topic(self.dds_participant, kTopicTeleopData, self.teleop_data_type)
        self.dds_writer_teleop = DataWriter(self.dds_participant, self.dds_topic_teleop)
        
        print(f"✅ DDS publisher initialized on topic: {kTopicTeleopData}")
    
    def init_dds_robot_state_subscriber(self):
        """Initialize DDS subscriber for robot state (end-effector pose)"""
        print("Initializing DDS subscriber for robot state...")
        
        from cyclonedds.sub import DataReader
        from teleop_dds_msgs import G1RobotState
        
        kTopicRobotState = "rt/robot/state"
        self.dds_topic_robot_state = Topic(self.dds_participant, kTopicRobotState, G1RobotState)
        self.dds_reader_robot_state = DataReader(self.dds_participant, self.dds_topic_robot_state)
        self.robot_state_thread = threading.Thread(target=self._robot_state_subscription_loop, daemon=True)
        self.robot_state_thread.start()
        print(f"✅ DDS robot state subscriber initialized on topic: {kTopicRobotState}")
    
    def init_dds_eef_state_subscriber(self):
        """Initialize DDS subscriber for EEF state (hand state)"""
        print("Initializing DDS subscriber for EEF state...")
        
        from cyclonedds.sub import DataReader
        from teleop_dds_msgs import EEFState
        
        kTopicEEFState = "rt/robot/eef_state"
        self.dds_topic_eef_state = Topic(self.dds_participant, kTopicEEFState, EEFState)
        self.dds_reader_eef_state = DataReader(self.dds_participant, self.dds_topic_eef_state)
        self.eef_state_thread = threading.Thread(target=self._eef_state_subscription_loop, daemon=True)
        self.eef_state_thread.start()
        print(f"✅ DDS EEF state subscriber initialized on topic: {kTopicEEFState}")

    def _flatten_touch_msg(self, msg):
        if self.inspire_hand_type == 'f1':
            return self._flatten_f1_touch_msg(msg)
        flat_values = []
        for field_name in INSPIRE_TOUCH_FIELD_SIZES.keys():
            field_value = getattr(msg, field_name, None)
            if field_value is None:
                values = np.zeros(INSPIRE_TOUCH_FIELD_SIZES[field_name], dtype=np.int16)
            else:
                values = np.asarray(field_value, dtype=np.int16).reshape(-1)
            expected_size = INSPIRE_TOUCH_FIELD_SIZES[field_name]
            if values.size < expected_size:
                padded = np.zeros(expected_size, dtype=np.int16)
                padded[:values.size] = values
                values = padded
            elif values.size > expected_size:
                values = values[:expected_size]
            flat_values.append(values)
        return np.concatenate(flat_values)

    def _flatten_f1_touch_msg(self, msg):
        return np.array(
            [getattr(msg, field, 0) for field in INSPIRE_F1_TOUCH_RAW_FIELDS],
            dtype=np.uint16,
        )

    def init_dds_touch_state_subscriber(self):
        """Initialize DDS subscribers for Inspire hand touch topics (left/right)."""
        print(f"Initializing DDS subscriber for Inspire hand touch state (hand_type={self.inspire_hand_type})...")
        try:
            from cyclonedds.sub import DataReader

            if self.inspire_hand_type == 'f1':
                from inspire_f1_sdk.inspire_dds import inspire_hand_f1_touch
                k_topic_inspire_touch_left = "rt/inspire_hand_f1/touch/l"
                k_topic_inspire_touch_right = "rt/inspire_hand_f1/touch/r"
                touch_msg_type = inspire_hand_f1_touch
            else:
                from inspire_sdkpy import inspire_dds
                k_topic_inspire_touch_left = "rt/inspire_hand/touch/l"
                k_topic_inspire_touch_right = "rt/inspire_hand/touch/r"
                touch_msg_type = inspire_dds.inspire_hand_touch

            self.dds_topic_inspire_touch_left = Topic(self.dds_participant, k_topic_inspire_touch_left, touch_msg_type)
            self.dds_topic_inspire_touch_right = Topic(self.dds_participant, k_topic_inspire_touch_right, touch_msg_type)
            self.left_touch_subscriber = DataReader(self.dds_participant, self.dds_topic_inspire_touch_left)
            self.right_touch_subscriber = DataReader(self.dds_participant, self.dds_topic_inspire_touch_right)

            self.touch_state_thread = threading.Thread(target=self._touch_state_subscription_loop, daemon=True)
            self.touch_state_thread.start()
            self._wait_for_initial_touch_data(timeout_s=5.0)
            print(f"✅ DDS Inspire touch subscribers initialized on topics: {k_topic_inspire_touch_left}, {k_topic_inspire_touch_right}")
        except Exception as e:
            self.left_touch_subscriber = None
            self.right_touch_subscriber = None
            print(f"Failed to initialize Inspire touch subscribers: {e}")

    def _wait_for_initial_touch_data(self, timeout_s=5.0):
        start = time.time()
        while time.time() - start < timeout_s:
            if self.left_touch_received and self.right_touch_received:
                print("✅ Initial Inspire touch DDS data received.")
                return
            time.sleep(0.01)
        print("Timeout waiting for initial Inspire touch data; using zero-initialized touch buffers.")

    def _touch_state_subscription_loop(self):
        """Subscription loop for Inspire hand touch topics."""
        print("Inspire touch subscription loop started")
        while self.running:
            try:
                left_samples = self.left_touch_subscriber.take(N=1)
                for sample in left_samples:
                    if sample is not None:
                        left_touch = self._flatten_touch_msg(sample)
                        with self.touch_state_lock:
                            self.left_touch_raw = left_touch
                        self.left_touch_received = True

                right_samples = self.right_touch_subscriber.take(N=1)
                for sample in right_samples:
                    if sample is not None:
                        right_touch = self._flatten_touch_msg(sample)
                        with self.touch_state_lock:
                            self.right_touch_raw = right_touch
                        self.right_touch_received = True
                time.sleep(0.005)
            except Exception as e:
                print(f"Error in Inspire touch subscription loop: {e}")
                time.sleep(0.1)
        print("Inspire touch subscription loop ended")
    
    def init_dds_reset_command_publisher(self):
        """Initialize the DDS publisher for G1 whole-body reset commands."""
        print("Initializing DDS publisher for reset commands...")
        
        from teleop_dds_msgs import ResetCommand
        
        # Create reset command topic
        kTopicResetCommand = "rt/robot/reset_command"
        self.dds_topic_reset_command = Topic(self.dds_participant, kTopicResetCommand, ResetCommand)
        self.dds_writer_reset_command = DataWriter(self.dds_participant, self.dds_topic_reset_command)
        
        print(f"✅ DDS reset command publisher initialized on topic: {kTopicResetCommand}")

    def init_dds_controller_reset_subscriber(self):
        """Subscribe to reset notifications published by the robot controller (e.g. remote button [2])."""
        print("Initializing DDS subscriber for controller reset notifications...")

        from cyclonedds.sub import DataReader
        from teleop_dds_msgs import ResetCommand

        kTopicControllerResetNotify = "rt/controller/reset_notify"
        self.dds_topic_controller_reset = Topic(self.dds_participant, kTopicControllerResetNotify, ResetCommand)
        self.dds_reader_controller_reset = DataReader(self.dds_participant, self.dds_topic_controller_reset)
        self.controller_reset_flag = False
        self.controller_reset_lock = threading.Lock()

        self.controller_reset_thread = threading.Thread(target=self._controller_reset_subscription_loop, daemon=True)
        self.controller_reset_thread.start()
        print(f"✅ DDS controller reset subscriber initialized on topic: {kTopicControllerResetNotify}")

    def _controller_reset_subscription_loop(self):
        """Background loop that listens for controller-initiated reset notifications."""
        while self.running:
            try:
                samples = self.dds_reader_controller_reset.take(N=1)
                for sample in samples:
                    if sample and sample.reset_requested:
                        with self.controller_reset_lock:
                            self.controller_reset_flag = True
                time.sleep(0.005)
            except Exception as e:
                print(f"Error in controller reset subscription loop: {e}")
                time.sleep(0.1)
    
    def _eef_state_subscription_loop(self):
        """DDS subscription loop for EEF state running in separate thread"""
        print("EEF state subscription loop started")
        
        while self.running:
            try:
                # Read EEF state from DDS
                samples = self.dds_reader_eef_state.take(N=1)
                for sample in samples:
                    eef_state = {
                        'timestamp': sample.timestamp,
                        'joint_positions': np.array(sample.joint_positions),
                        'finger_forces': np.array(sample.finger_forces),
                        'hand_link_poses': np.array(sample.hand_link_poses), # 3d finger tip positions of left and right hands
                    }
                    with self.robot_eef_state_lock:
                        self.robot_eef_state = eef_state
                time.sleep(0.005)
                
            except Exception as e:
                print(f"Error in EEF state subscription loop: {e}")
                time.sleep(0.1)
        print("EEF state subscription loop ended")

    def _robot_state_subscription_loop(self):
        """DDS subscription loop for robot state running in separate thread"""
        print("Robot state subscription loop started")
        
        while self.running:
            try:
                # Read robot state from DDS
                samples = self.dds_reader_robot_state.take(N=1)
                for sample in samples:
                    robot_state = {
                        'timestamp': sample.timestamp,
                        'torso_pose': np.array(sample.torso_pose), # xyz+rpy
                        'left_eef_pose': np.array(sample.left_end_effector_pose), # xyz+rpy
                        'right_eef_pose': np.array(sample.right_end_effector_pose), # xyz+rpy
                        'left_eef_to_torso_pose': np.array(sample.left_end_effector_to_torso_pose), # xyz+rpy
                        'right_eef_to_torso_pose': np.array(sample.right_end_effector_to_torso_pose), # xyz+rpy
                        'joint_positions': np.array(sample.joint_positions), # 29 joints
                        'joint_velocities': np.array(sample.joint_velocities), # 29 joints
                        'velocity_commands': np.array(sample.velocity_commands), # 3 floats (lin_vel_x, lin_vel_y, ang_vel_z)
                        'imu_accelerometer': np.array(sample.imu_accelerometer),
                        'imu_gyroscope': np.array(sample.imu_gyroscope),
                        'imu_rpy': np.array(sample.imu_rpy),
                    }
                    with self.robot_state_lock:
                        self.robot_state = robot_state.copy()
                time.sleep(0.005)
                
            except Exception as e:
                print(f"Error in robot state subscription loop: {e}")
                time.sleep(0.1)
        
        print("Robot state subscription loop ended")
    

    def _snapshot_dds_state(self, state_attr, lock, keys):
        """Return copies of ``keys`` from a cached DDS state dictionary."""
        with lock:
            state = getattr(self, state_attr, None)
            if state is None:
                return None
            return {key: np.array(state[key], copy=True) for key in keys}

    def _warn_proprio_not_ready(self, source):
        now = time.time()
        if now - getattr(self, '_last_proprio_wait_warn', 0.0) > 1.0:
            print(f"Waiting for {source} before collecting...")
            self._last_proprio_wait_warn = now

    def update_embodiment_proprio_states(self):
        """Refresh proprio state without mutating cached DDS samples.

        Returns False until every required state source has produced a sample.
        """
        # human
        if self.embodiment == 'human':
            self.body_pose_proprio = self.body_pose_uni.copy()
            self.eef_pose_proprio = self.eef_pose_uni.copy()
            self.eef_to_body_pose_proprio = self.eef_to_body_pose.copy()
            self.gripper_angle_proprio = self.gripper_angle.copy()
            self.eef_link_proprio = np.concatenate([self.right_hand_joints, self.left_hand_joints])
        elif self.embodiment == 'g1_inspire_wb':
            robot_state = self._snapshot_dds_state(
                'robot_state', self.robot_state_lock,
                ['torso_pose', 'right_eef_pose', 'left_eef_pose',
                 'right_eef_to_torso_pose', 'left_eef_to_torso_pose',
                 'joint_positions', 'joint_velocities', 'velocity_commands',
                 'imu_accelerometer', 'imu_gyroscope', 'imu_rpy'],
            )
            eef_state = self._snapshot_dds_state(
                'robot_eef_state', self.robot_eef_state_lock,
                ['hand_link_poses', 'joint_positions', 'finger_forces'],
            )
            if robot_state is None or eef_state is None:
                self._warn_proprio_not_ready('g1_inspire_wb robot state')
                return False
            self.body_pose_proprio = robot_state['torso_pose']
            right_eef_pose_proprio = robot_state['right_eef_pose']
            left_eef_pose_proprio = robot_state['left_eef_pose']
            right_eef_to_body_pose_proprio = robot_state['right_eef_to_torso_pose']
            left_eef_to_body_pose_proprio = robot_state['left_eef_to_torso_pose']
            self.robot_joint_proprio = np.concatenate([robot_state['joint_positions'], robot_state['joint_velocities']])
            self.velocity_commands = robot_state['velocity_commands']
            self.imu_proprio = np.concatenate([robot_state['imu_accelerometer'], robot_state['imu_gyroscope'], robot_state['imu_rpy']])

            right_eef_pose_proprio_rot = self.rpy_to_rot_mat(right_eef_pose_proprio[3:]) @ self.unified_to_unitree_right_eef_rot_remap.T
            left_eef_pose_proprio_rot = self.rpy_to_rot_mat(left_eef_pose_proprio[3:]) @ self.unified_to_unitree_left_eef_rot_remap.T
            self.eef_pose_proprio = np.concatenate([
                right_eef_pose_proprio[:3], self.rot_mat_to_rpy(right_eef_pose_proprio_rot),
                left_eef_pose_proprio[:3], self.rot_mat_to_rpy(left_eef_pose_proprio_rot),
            ])
            right_eef_to_body_pose_proprio_rot = self.rpy_to_rot_mat(right_eef_to_body_pose_proprio[3:]) @ self.unified_to_unitree_right_eef_rot_remap.T
            left_eef_to_body_pose_proprio_rot = self.rpy_to_rot_mat(left_eef_to_body_pose_proprio[3:]) @ self.unified_to_unitree_left_eef_rot_remap.T
            self.eef_to_body_pose_proprio = np.concatenate([
                right_eef_to_body_pose_proprio[:3], self.rot_mat_to_rpy(right_eef_to_body_pose_proprio_rot),
                left_eef_to_body_pose_proprio[:3], self.rot_mat_to_rpy(left_eef_to_body_pose_proprio_rot),
            ])
            eef_link_proprio = eef_state['hand_link_poses']
            self.eef_joint_proprio = np.concatenate([eef_state['joint_positions'], eef_state['finger_forces']])
            right_eef_link_proprio = eef_link_proprio[0:15].reshape(-1, 3)
            left_eef_link_proprio = eef_link_proprio[15:30].reshape(-1, 3)
            right_eef_link_proprio = (self.inspire_to_unified_eef_local[:3, :3] @ right_eef_link_proprio.T).T.flatten()
            left_eef_link_proprio = (self.inspire_to_unified_eef_local[:3, :3] @ left_eef_link_proprio.T).T.flatten()
            self.eef_link_proprio = np.concatenate([right_eef_link_proprio, left_eef_link_proprio])
            if self.enable_inspire_touch_collection:
                with self.touch_state_lock:
                    if self.inspire_hand_type == 'f1':
                        self.left_touch = _process_f1_touch_raw(self.left_touch_raw)
                        self.right_touch = _process_f1_touch_raw(self.right_touch_raw)
                    else:
                        self.left_touch = np.asarray(self.left_touch_raw, dtype=np.float32).reshape(-1) / 4095.0
                        self.right_touch = np.asarray(self.right_touch_raw, dtype=np.float32).reshape(-1) / 4095.0
        return True

    def send_reset_command(self):
        """Send a reset command to the G1 whole-body controller via DDS."""
        if hasattr(self, 'dds_writer_reset_command'):
            from teleop_dds_msgs import ResetCommand
            msg = ResetCommand(
                timestamp=time.time(),
                reset_requested=True
            )
            self.dds_writer_reset_command.write(msg)
            print("✅ Reset command sent")

    def publish_wb_teleop_data(self, left_eef_pos, left_eef_rot, right_eef_pos, right_eef_rot, left_hand_joints, right_hand_joints, teleop_torso_rot_uni, teleop_torso_height):
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

        # Publish via DDS with torso data using TeleopDataWB
        msg = TeleopDataWB(
            timestamp=time.time(),
            left_arm_pose=left_arm_pose.flatten().tolist(),
            right_arm_pose=right_arm_pose.flatten().tolist(),
            left_hand_pose=left_hand_joints.tolist(),
            right_hand_pose=right_hand_joints.tolist(),
            torso_rpy=teleop_torso_rot_uni.tolist() if hasattr(teleop_torso_rot_uni, 'tolist') else list(teleop_torso_rot_uni),
            torso_height=float(teleop_torso_height),
            velocity_commands=[]
        )
        self.dds_writer_teleop.write(msg)

    def process_vr_data(self):
        if not self.args.see_through_mode:
            # left_hand_data: [4, 4] - check for singularity
            new_left_hand_data = self.vr_streamer.left_hand
            left_hand_data, left_hand_safe = self.safe_mat_update(self.prev_left_hand_data, new_left_hand_data)
            if left_hand_safe:
                self.prev_left_hand_data = left_hand_data.copy()
            else:
                pass
                # print("Warning: Using previous left_hand_data due to singularity")

            # right_hand_data: [4, 4] - check for singularity
            new_right_hand_data = self.vr_streamer.right_hand
            right_hand_data, right_hand_safe = self.safe_mat_update(self.prev_right_hand_data, new_right_hand_data)
            if right_hand_safe:
                self.prev_right_hand_data = right_hand_data.copy()
            else:
                pass
                # print("Warning: Using previous right_hand_data due to singularity")

            # head_data: [4, 4] - check for singularity
            new_head_data = self.vr_streamer.head_matrix
            head_data, head_safe = self.safe_mat_update(self.prev_head_data, new_head_data)
            if head_safe:
                self.prev_head_data = head_data.copy()
            else:
                pass
                # print("Warning: Using previous head_data due to singularity")
            self.left_wrist_pos_vr = left_hand_data[:3, 3]
            left_wrist_rot = left_hand_data[:3, :3]
            self.right_wrist_pos_vr = right_hand_data[:3, 3]
            right_wrist_rot = right_hand_data[:3, :3]
            self.head_pos_vr = head_data[:3, 3]
            head_rot = head_data[:3, :3]        
            ### define new head rot and wrist local frame at reset to align with the unified frame which refers to the issacgym/isaaclab world frame
            ### issacgym/isaaclab world frame: x forward, y leftward, z upward
            ### unified frame: z upward (x-y plane horizontal to the ground), x aligns with the forward direction of human head at reset (rotate around z axis using the head yaw)
            head_rot_new = np.zeros_like(head_rot)
            head_rot_new[:, 0] = -head_rot[:, 2]
            head_rot_new[:, 1] = -head_rot[:, 0]
            head_rot_new[:, 2] = head_rot[:, 1]
            # keep the old one to extract head yaw at reset, then rotate the unified frame using the head yaw
            self.head_rot_old_vr = head_rot
            self.head_rot_vr = head_rot_new
            
            left_wrist_rot_new = np.zeros_like(left_wrist_rot)
            left_wrist_rot_new[:, 0] = -left_wrist_rot[:, 2]
            left_wrist_rot_new[:, 1] = -left_wrist_rot[:, 0]
            left_wrist_rot_new[:, 2] = left_wrist_rot[:, 1]
            self.left_wrist_rot_vr = left_wrist_rot_new
            
            right_wrist_rot_new = np.zeros_like(right_wrist_rot)
            right_wrist_rot_new[:, 0] = -right_wrist_rot[:, 2]
            right_wrist_rot_new[:, 1] = -right_wrist_rot[:, 0]
            right_wrist_rot_new[:, 2] = right_wrist_rot[:, 1]
            self.right_wrist_rot_vr = right_wrist_rot_new

            # left_landmarks_data: [25, 3]
            left_landmarks_data = self.vr_streamer.left_landmarks
            self.left_landmarks_data_vr = left_landmarks_data
            # right_landmarks_data: [25, 3]
            right_landmarks_data = self.vr_streamer.right_landmarks
            self.right_landmarks_data_vr = right_landmarks_data

            # thumb to index finger
            self.left_pinch_dist0_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[9]) 
            self.right_pinch_dist0_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[9]) 
            # thumb to middle finger
            self.left_pinch_dist1_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[14]) 
            self.right_pinch_dist1_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[14]) 
            # thumb to ring finger
            self.left_pinch_dist2_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[19])
            self.right_pinch_dist2_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[19])
            # thumb to little finger
            self.left_pinch_dist3_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[24])
            self.right_pinch_dist3_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[24])
        
        else:
            # left_hand_data: [4, 4] - check for singularity
            new_left_hand_data = self.z_up_inv @ self.vr_streamer.latest['left_wrist'][0]
            left_hand_data, left_hand_safe = self.safe_mat_update(self.prev_left_hand_data, new_left_hand_data)
            if left_hand_safe:
                self.prev_left_hand_data = left_hand_data.copy()
            else:
                pass
                # print("Warning: Using previous left_hand_data due to singularity")

            # right_hand_data: [4, 4] - check for singularity
            new_right_hand_data = self.z_up_inv @ self.vr_streamer.latest['right_wrist'][0]
            right_hand_data, right_hand_safe = self.safe_mat_update(self.prev_right_hand_data, new_right_hand_data)
            if right_hand_safe:
                self.prev_right_hand_data = right_hand_data.copy()
            else:
                pass
                # print("Warning: Using previous right_hand_data due to singularity")

            # head_data: [4, 4] - check for singularity
            new_head_data = self.z_up_inv @ self.vr_streamer.latest['head'][0]
            head_data, head_safe = self.safe_mat_update(self.prev_head_data, new_head_data)
            if head_safe:
                self.prev_head_data = head_data.copy()
            else:
                pass
                # print("Warning: Using previous head_data due to singularity")
            self.left_wrist_pos_vr = left_hand_data[:3, 3]
            left_wrist_rot = left_hand_data[:3, :3]
            self.right_wrist_pos_vr = right_hand_data[:3, 3]
            right_wrist_rot = right_hand_data[:3, :3]
            self.head_pos_vr = head_data[:3, 3]
            head_rot = head_data[:3, :3]        
            ### define new head rot and wrist local frame at reset to align with the unified frame which refers to the issacgym/isaaclab world frame
            ### issacgym/isaaclab world frame: x forward, y leftward, z upward
            ### unified frame: z upward (x-y plane horizontal to the ground), x aligns with the forward direction of human head at reset (rotate around z axis using the head yaw)
            
            ### head_rot_old: make avp_head local frame consistent with the tele_vision's head local frame (only to calculate head yaw at reset)
            head_rot_old = np.zeros_like(head_rot)
            head_rot_old[:, 0] = head_rot[:, 0]
            head_rot_old[:, 1] = head_rot[:, 2]
            head_rot_old[:, 2] = -head_rot[:, 1]
            self.head_rot_old_vr = head_rot_old
            
            head_rot_new = np.zeros_like(head_rot)
            head_rot_new[:, 0] = head_rot[:, 1]
            head_rot_new[:, 1] = -head_rot[:, 0]
            head_rot_new[:, 2] = head_rot[:, 2]
            # keep the old one to extract head yaw at reset, then rotate the unified frame using the head yaw
            self.head_rot_vr = head_rot_new
            
            left_wrist_rot_new = np.zeros_like(left_wrist_rot)
            left_wrist_rot_new[:, 0] = left_wrist_rot[:, 0]
            left_wrist_rot_new[:, 1] = left_wrist_rot[:, 2]
            left_wrist_rot_new[:, 2] = -left_wrist_rot[:, 1]
            self.left_wrist_rot_vr = left_wrist_rot_new
            
            right_wrist_rot_new = np.zeros_like(right_wrist_rot)
            right_wrist_rot_new[:, 0] = -right_wrist_rot[:, 0]
            right_wrist_rot_new[:, 1] = right_wrist_rot[:, 2]
            right_wrist_rot_new[:, 2] = right_wrist_rot[:, 1]
            self.right_wrist_rot_vr = right_wrist_rot_new

            # left_landmarks_data: [25, 3]
            left_landmarks_data = self.vr_streamer.latest['left_fingers'][:, 0:3, 3]
            # to align with tele_vision's data, transform it to the avp world frame
            self.left_landmarks_data_vr = (left_wrist_rot @ left_landmarks_data.T).T + left_hand_data[:3, 3]
            # right_landmarks_data: [25, 3]
            right_landmarks_data = self.vr_streamer.latest['right_fingers'][:, 0:3, 3]
            self.right_landmarks_data_vr = (right_wrist_rot @ right_landmarks_data.T).T + right_hand_data[:3, 3]

            # thumb to index finger
            self.left_pinch_dist0_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[9]) 
            self.right_pinch_dist0_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[9]) 
            # thumb to middle finger
            self.left_pinch_dist1_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[14]) 
            self.right_pinch_dist1_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[14]) 
            # thumb to ring finger
            self.left_pinch_dist2_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[19])
            self.right_pinch_dist2_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[19])
            # thumb to little finger
            self.left_pinch_dist3_vr = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[24])
            self.right_pinch_dist3_vr = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[24])


    def update_embodiment_target_states(self, reset_uni_frame=False):
        if reset_uni_frame:
            ### calculate the poses at reset for calibration
            # extract the head rotation around y axis in the world frame of apple vision pro (head yaw)
            head_rpy = self.rot_mat_to_rpy_zxy(self.head_rot_old_vr)
            # use the head yaw to build a new frame to map the 6d poses of the human hand and head
            # the new operator_pov_transform represents the rotation matrix of the apple vision pro frame relative to the unified frame (IsaacGym/world frame)
            # unified frame: z upward (x-y plane horizontal to the ground), x aligns with the forward direction of human head at reset (rotate around z axis using the head yaw)
            z_axis_rot = self.rpy_to_rot_mat(np.array([0, 0, -head_rpy[1]]))
            self.operator_pov_transform = z_axis_rot @ self.operator_pov_transform
            self.init_body_pos[:] = self.operator_pov_transform @ self.head_pos_vr
            self.init_body_rot[:] = self.operator_pov_transform @ self.head_rot_vr
            self.init_right_eef_pos[:] = self.operator_pov_transform @ self.right_wrist_pos_vr - self.init_body_pos
            self.init_right_eef_rot[:] = self.operator_pov_transform @ self.right_wrist_rot_vr
            self.init_left_eef_pos[:] = self.operator_pov_transform @ self.left_wrist_pos_vr - self.init_body_pos
            self.init_left_eef_rot[:] = self.operator_pov_transform @ self.left_wrist_rot_vr
        ### update calibrated states for different embodiments
        # use the head position at reset as the origin of the frame
        if self.embodiment == 'human' or self.embodiment == 'g1_inspire_wb':
            # body (head) pose [6]
            body_pos_uni = self.operator_pov_transform @ self.head_pos_vr - self.init_body_pos
            body_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ self.head_rot_vr)
            body_pose_uni = np.concatenate((body_pos_uni, body_rot_uni))
            if reset_uni_frame:
                self.delta_body_pose_uni = np.zeros_like(self.body_pose_uni)
            else:
                self.delta_body_pose_uni = body_pose_uni - self.body_pose_uni
            self.body_pose_uni = body_pose_uni
            # eef pose [6 * 2]
            right_eef_pos_uni = self.operator_pov_transform @ self.right_wrist_pos_vr - self.init_body_pos
            right_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ self.right_wrist_rot_vr)
            left_eef_pos_uni = self.operator_pov_transform @ self.left_wrist_pos_vr - self.init_body_pos
            left_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ self.left_wrist_rot_vr)
            eef_pose_uni = np.concatenate([right_eef_pos_uni, right_eef_rot_uni, left_eef_pos_uni, left_eef_rot_uni])
            if reset_uni_frame:
                self.delta_eef_pose_uni = np.zeros_like(self.eef_pose_uni)
            else:
                self.delta_eef_pose_uni = eef_pose_uni - self.eef_pose_uni
            self.eef_pose_uni = eef_pose_uni
            # eef relative to body [6 * 2]
            right_eef_to_body_pos = self.rpy_to_rot_mat(body_rot_uni).T @ (right_eef_pos_uni - body_pos_uni)
            right_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(right_eef_rot_uni))
            left_eef_to_body_pos = self.rpy_to_rot_mat(body_rot_uni).T @ (left_eef_pos_uni - body_pos_uni)
            left_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(left_eef_rot_uni))
            self.eef_to_body_pose = np.concatenate([right_eef_to_body_pos, right_eef_to_body_rot, left_eef_to_body_pos, left_eef_to_body_rot])
            # simulated gripper (hand as gripper) [1 * 2]
            right_gripper_angle = self.get_gripper_angle_from_pinch_dist(self.right_pinch_dist0_vr)
            left_gripper_angle = self.get_gripper_angle_from_pinch_dist(self.left_pinch_dist0_vr)
            gripper_angle = np.array([right_gripper_angle, left_gripper_angle])
            if reset_uni_frame:
                self.delta_gripper_angle = np.zeros_like(self.gripper_angle)
            else:
                self.delta_gripper_angle = gripper_angle - self.gripper_angle
            self.gripper_angle = gripper_angle

            # calculate the hand joints in the local wrist frame
            right_hand_joints = (self.right_wrist_rot_vr.T @ (self.right_landmarks_data_vr - self.right_wrist_pos_vr).T).T
            self.right_hand_joints = right_hand_joints.flatten()
            left_hand_joints = (self.left_wrist_rot_vr.T @ (self.left_landmarks_data_vr - self.left_wrist_pos_vr).T).T
            self.left_hand_joints = left_hand_joints.flatten()

            # main camera pose relative to its initial pose in the unified frame (we assume that the main camera is fixed on the head)
            head_camera_to_init_rot = self.rot_mat_to_rpy(self.init_body_rot.T @ (self.operator_pov_transform @ self.head_rot_vr))
            self.head_camera_to_init_pose_uni = np.concatenate((body_pos_uni, head_camera_to_init_rot))

            self.eef_joints_target_pose = np.concatenate([self.right_hand_joints, self.left_hand_joints])
        
        else:
            raise ValueError(f"Unsupported embodiment: {self.embodiment}")

        ### calculate teleoperation command to be broadcast to the embodiment-specific controller
        if not reset_uni_frame:
            if self.embodiment == 'human':
                return
            elif self.embodiment == 'g1_inspire_wb':
                # mapping rules: 
                # (1) human head height changes to humanoid torso height changes (in the unified frame)
                # (2) human head rotations to humanoid torso rotations (in the unified frame)
                # torso 
                # torso rpy
                # teleop_torso_rot_uni = self.init_body_rot.T @ self.rpy_to_rot_mat(body_rot_uni)
                teleop_torso_rot_uni = body_rot_uni.copy()
                # torso height (body_pos_uni[2] = 0 at reset)
                teleop_torso_height = body_pos_uni[2] + 0.75
                
                # left eef
                teleop_left_eef_pos_uni = self.eef_to_body_pose[6:9].copy()
                # head to pelvis offsets (for arm ik)
                teleop_left_eef_pos_uni[0] += 0.15
                teleop_left_eef_pos_uni[2] += 0.45
                # the sent eef rpy command is in the torso frame
                left_eef_rot_in_torso = self.rpy_to_rot_mat(self.eef_to_body_pose[9:].copy())
                # unitree g1 eef convention
                teleop_left_eef_rot_uni = left_eef_rot_in_torso @ self.unified_to_unitree_left_eef_rot_remap
                teleop_left_eef_rot_uni = self.rot_mat_to_rpy(teleop_left_eef_rot_uni)
                # right eef
                teleop_right_eef_pos_uni = self.eef_to_body_pose[:3].copy()
                # head to pelvis offsets (for arm ik)
                teleop_right_eef_pos_uni[0] += 0.15
                teleop_right_eef_pos_uni[2] += 0.45
                # the sent eefrpy command is in the torso frame
                right_eef_rot_in_torso = self.rpy_to_rot_mat(self.eef_to_body_pose[3:6].copy())
                # unitree g1 eef convention
                teleop_right_eef_rot_uni = right_eef_rot_in_torso @ self.unified_to_unitree_right_eef_rot_remap
                teleop_right_eef_rot_uni = self.rot_mat_to_rpy(teleop_right_eef_rot_uni)
                # inspire convention
                unified_eef_to_inspire = self.inspire_to_unified_eef_local[:3, :3].T
                teleop_left_hand_joints = (unified_eef_to_inspire @ left_hand_joints.T).T.flatten()
                teleop_right_hand_joints = (unified_eef_to_inspire @ right_hand_joints.T).T.flatten()
                
                self.publish_wb_teleop_data(teleop_left_eef_pos_uni, teleop_left_eef_rot_uni, teleop_right_eef_pos_uni, teleop_right_eef_rot_uni, teleop_left_hand_joints, teleop_right_hand_joints, teleop_torso_rot_uni, teleop_torso_height)

            else:
                raise ValueError(f"Unsupported embodiment: {self.embodiment}")


    def reset_embodiment_states(self):
        # need to reset: make sure add the updated state to the trajectory meanwhile keep previous ones unchanged
        # not reset to zeros: copy the value to compute delta
        self.body_pose_uni = self.body_pose_uni.copy()
        self.delta_body_pose_uni = self.delta_body_pose_uni.copy()
        self.eef_pose_uni = self.eef_pose_uni.copy()
        self.delta_eef_pose_uni = self.delta_eef_pose_uni.copy()
        self.eef_to_body_pose = self.eef_to_body_pose.copy()
        self.gripper_angle = self.gripper_angle.copy()
        self.delta_gripper_angle = self.delta_gripper_angle.copy()
        self.right_hand_joints = self.right_hand_joints.copy()
        self.left_hand_joints = self.left_hand_joints.copy()
        # camera pose relative to its initial pose in the unified frame
        self.head_camera_to_init_pose_uni = self.head_camera_to_init_pose_uni.copy()
        # proprio states does not need to be reset (no delta needed)

    def reset_trajectories(self):
        with self.trajectory_lock:
            self.main_cam_image_history = []
            self.wrist_cam_image_history = []
            self.wrist_left_cam_image_history = []
            self.wrist_right_cam_image_history = []
            self.body_pose_history = []
            self.delta_body_pose_history = []
            self.eef_pose_history = []
            self.delta_eef_pose_history = []
            self.eef_to_body_pose_history = []
            self.gripper_angle_history = []
            self.delta_gripper_angle_history = []
            self.head_camera_to_init_pose_history = []
            self.head_cam_timestamp_history = []
            self.eef_joints_target_pose_history = []

            self.body_pose_proprio_history = []
            self.eef_pose_proprio_history = []
            self.eef_to_body_pose_proprio_history = []
            self.gripper_angle_proprio_history = []
            self.eef_link_proprio_history = []
            if self.embodiment != 'human':
                self.robot_joint_proprio_history = []
                self.eef_joint_proprio_history = []
                if self.embodiment == 'g1_inspire_wb':
                    self.velocity_commands_history = []
                    self.imu_proprio_history = []
                    if self.enable_inspire_touch_collection:
                        self.left_touch_history = []
                        self.right_touch_history = []

    def update_trajectories(self):
        with self.trajectory_lock:
            self.main_cam_image_history.append(self.head_color_frame)
            if self.args.use_wrist_camera:
                if self.use_dual_wrist_camera:
                    self.wrist_left_cam_image_history.append(self.wrist_left_color_frame)
                    self.wrist_right_cam_image_history.append(self.wrist_right_color_frame)
                else:
                    self.wrist_cam_image_history.append(self.wrist_color_frame)
            self.body_pose_history.append(self.body_pose_uni)
            self.delta_body_pose_history.append(self.delta_body_pose_uni)
            self.eef_pose_history.append(self.eef_pose_uni)
            self.delta_eef_pose_history.append(self.delta_eef_pose_uni)
            self.eef_to_body_pose_history.append(self.eef_to_body_pose)
            self.gripper_angle_history.append(self.gripper_angle)
            self.delta_gripper_angle_history.append(self.delta_gripper_angle)
            self.head_camera_to_init_pose_history.append(self.head_camera_to_init_pose_uni)
            self.head_cam_timestamp_history.append(self.head_cam_time)
            self.eef_joints_target_pose_history.append(self.eef_joints_target_pose)

            self.body_pose_proprio_history.append(self.body_pose_proprio)
            self.eef_pose_proprio_history.append(self.eef_pose_proprio)
            self.eef_to_body_pose_proprio_history.append(self.eef_to_body_pose_proprio)
            self.gripper_angle_proprio_history.append(self.gripper_angle_proprio)
            self.eef_link_proprio_history.append(self.eef_link_proprio)

            if self.embodiment != 'human':
                self.robot_joint_proprio_history.append(self.robot_joint_proprio)
                self.eef_joint_proprio_history.append(self.eef_joint_proprio)
                if self.embodiment == 'g1_inspire_wb':
                    self.velocity_commands_history.append(self.velocity_commands)
                    self.imu_proprio_history.append(self.imu_proprio)
                    if self.enable_inspire_touch_collection:
                        self.left_touch_history.append(self.left_touch.copy())
                        self.right_touch_history.append(self.right_touch.copy())

    def get_embodiment_masks(self):
        self.img_main_mask = np.array([True])
        self.img_wrist_mask = np.array([False, False])
        if self.args.use_wrist_camera:
            for eef_idx in self.manipulate_eef_idx:
                self.img_wrist_mask[self.manipulate_eef_idx] = True
        
        # Keep x, y, z, roll, pitch, and yaw, matching HTD data and G1 rollout.
        self.proprio_body_mask = np.ones(6, dtype=bool)
        self.proprio_eef_mask = np.array([False] * 12)
        self.proprio_gripper_mask = np.array([False, False])
        self.proprio_other_mask = np.array([False, False])      
      
        self.act_body_mask = np.array([True, True, True, True, True, True])
        if self.embodiment == 'human':
            self.act_body_mask = np.array([True, True, True, True, True, True])
        elif self.embodiment == 'g1_inspire_wb':
            self.act_body_mask = np.array([False, False, True, True, True, True])
        self.act_eef_mask = np.array([False] * 12)
        self.act_gripper_mask = np.array([False, False])
        for eef_idx in self.manipulate_eef_idx:
            self.proprio_eef_mask[6*eef_idx:6+6*eef_idx] = True
            self.proprio_gripper_mask[eef_idx] = True
            self.proprio_other_mask[eef_idx] = True
            self.act_eef_mask[6*eef_idx:6+6*eef_idx] = True
            self.act_gripper_mask[eef_idx] = True


    def generate_videos_from_hdf5(self, save_path, episode_num, control_freq, use_wrist_camera):
        """Generate videos by reading frames from the saved HDF5 file"""
        hdf5_file_path = os.path.join(save_path, f"episode_{episode_num}.hdf5")
        
        try:
            with h5py.File(hdf5_file_path, 'r') as f:
                image_group = f['observations/images']
                
                # generate main camera video
                # check image save format
                main_dset = image_group['main']
                if len(main_dset.shape) == 1:
                    # png or jpg bytes format
                    main_images = []
                    for i in range(len(main_dset)):
                        if self.args.image_save_format == "png":
                            png_bytes = main_dset[i]  # 1D uint8 array
                            rgb_img = self._decode_png_bytes(png_bytes)
                        else:
                            jpg_bytes = main_dset[i]  # 1D uint8 array
                            rgb_img = self._decode_jpg_bytes(jpg_bytes)
                        main_images.append(rgb_img)
                    main_images = np.array(main_images)
                    print(f"Decoded {len(main_images)} images for main camera video")
                else:
                    # raw numpy arrays format
                    main_images = main_dset[:]
                    print(f"Using raw numpy arrays for main camera video")
                
                if len(main_images) > 0:
                    h, w, _ = main_images[0].shape
                    main_cam_video = cv2.VideoWriter(os.path.join(save_path, f"episode_{episode_num}_main_cam_video.mp4"), 
                                                     cv2.VideoWriter_fourcc(*'mp4v'), control_freq, (w, h))
                    
                    for image in main_images:
                        # swap back to bgr for opencv (images are RGB from PNG or raw arrays)
                        image = image[:, :, [2, 1, 0]] 
                        main_cam_video.write(image)
                    main_cam_video.release()
                    print(f"Generated main camera video with {len(main_images)} frames")
                
                # Generate wrist camera video(s) if available
                if use_wrist_camera:
                    def _load_wrist_images(dset):
                        if len(dset.shape) == 1:
                            images = []
                            for i in range(len(dset)):
                                if self.args.image_save_format == "png":
                                    png_bytes = dset[i]
                                    rgb_img = self._decode_png_bytes(png_bytes)
                                else:
                                    jpg_bytes = dset[i]
                                    rgb_img = self._decode_jpg_bytes(jpg_bytes)
                                images.append(rgb_img)
                            return np.array(images)
                        return dset[:]

                    if 'wrist_left' in image_group and 'wrist_right' in image_group:
                        wrist_left_images = _load_wrist_images(image_group['wrist_left'])
                        wrist_right_images = _load_wrist_images(image_group['wrist_right'])
                        if len(wrist_left_images) > 0:
                            h, w, _ = wrist_left_images[0].shape
                            wrist_left_video = cv2.VideoWriter(os.path.join(save_path, f"episode_{episode_num}_wrist_left_cam_video.mp4"), 
                                                              cv2.VideoWriter_fourcc(*'mp4v'), control_freq, (w, h))
                            for image in wrist_left_images:
                                image = image[:, :, [2, 1, 0]]
                                wrist_left_video.write(image)
                            wrist_left_video.release()
                            print(f"Generated left wrist camera video with {len(wrist_left_images)} frames")
                        if len(wrist_right_images) > 0:
                            h, w, _ = wrist_right_images[0].shape
                            wrist_right_video = cv2.VideoWriter(os.path.join(save_path, f"episode_{episode_num}_wrist_right_cam_video.mp4"), 
                                                               cv2.VideoWriter_fourcc(*'mp4v'), control_freq, (w, h))
                            for image in wrist_right_images:
                                image = image[:, :, [2, 1, 0]]
                                wrist_right_video.write(image)
                            wrist_right_video.release()
                            print(f"Generated right wrist camera video with {len(wrist_right_images)} frames")
                    elif 'wrist' in image_group:
                        wrist_images = _load_wrist_images(image_group['wrist'])
                        if len(wrist_images) > 0:
                            h, w, _ = wrist_images[0].shape
                            wrist_cam_video = cv2.VideoWriter(os.path.join(save_path, f"episode_{episode_num}_wrist_cam_video.mp4"), 
                                                            cv2.VideoWriter_fourcc(*'mp4v'), control_freq, (w, h))
                            for image in wrist_images:
                                image = image[:, :, [2, 1, 0]]
                                wrist_cam_video.write(image)
                            wrist_cam_video.release()
                            print(f"Generated wrist camera video with {len(wrist_images)} frames")
                        
        except Exception as e:
            print(f"Error generating videos from HDF5: {e}")
            import traceback
            traceback.print_exc()


    def background_save_worker(self):
        """
        Background thread that continuously monitors data lists and saves when enough data accumulates.
        Also automatically handles episode completion when on_collect becomes False.
        """
        while True:
            try:
                # Check if we're collecting data and have enough accumulated
                if self.args.collect_data and self.on_collect and len(self.eef_pose_history) > 0:
                    # Initialize episode tracking if needed
                    if not hasattr(self, 'current_episode_num'):
                        self.current_episode_num = self.increment_counter
                        self.episode_start_time = time.time()
                        self.frames_saved_count = 0
                        print(f"Background: Starting episode {self.current_episode_num}")

                    # Check if we have enough data (2 seconds worth)
                    frames_per_2_seconds = 2 * self.args.control_freq
                    current_frame_count = len(self.eef_pose_history)

                    if current_frame_count >= frames_per_2_seconds:
                        # Process the batch directly
                        self.saving_in_progress = True
                        self.process_batch_save(frames_per_2_seconds)
                        self.saving_in_progress = False
                        continue

                # Check for episode completion (on_collect became False)
                elif self.args.collect_data and not self.on_collect and hasattr(self, 'current_episode_num'):
                    # Episode completed - save any remaining data
                    self.saving_in_progress = True
                    self.handle_episode_completion()
                    self.saving_in_progress = False
                    continue
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)

            except Exception as e:
                print(f"Error in background save worker: {e}")
                self.saving_in_progress = False
                time.sleep(1.0)
                continue

    def process_batch_save(self, frames_to_save):
        """Process batch saving"""
        episode_num = self.current_episode_num

        print(f"Background: Processing batch save for episode {episode_num} ({frames_to_save} frames)...")

        # Extract data from lists (this is now done in background thread)
        save_data = {
            'episode_num': episode_num,
            'save_path': self.exp_data_folder,
            'main_cam_image_history': self.main_cam_image_history[:frames_to_save],
            'wrist_cam_image_history': self.wrist_cam_image_history[:frames_to_save],
            'wrist_left_cam_image_history': self.wrist_left_cam_image_history[:frames_to_save],
            'wrist_right_cam_image_history': self.wrist_right_cam_image_history[:frames_to_save],
            'body_pose_history': self.body_pose_history[:frames_to_save],
            'delta_body_pose_history': self.delta_body_pose_history[:frames_to_save],
            'eef_pose_history': self.eef_pose_history[:frames_to_save],
            'delta_eef_pose_history': self.delta_eef_pose_history[:frames_to_save],
            'eef_to_body_pose_history': self.eef_to_body_pose_history[:frames_to_save],
            'gripper_angle_history': self.gripper_angle_history[:frames_to_save],
            'delta_gripper_angle_history': self.delta_gripper_angle_history[:frames_to_save],
            'head_camera_to_init_pose_history': self.head_camera_to_init_pose_history[:frames_to_save],
            'head_cam_timestamp_history': self.head_cam_timestamp_history[:frames_to_save],
            'body_pose_proprio_history': self.body_pose_proprio_history[:frames_to_save],
            'eef_pose_proprio_history': self.eef_pose_proprio_history[:frames_to_save],
            'eef_to_body_pose_proprio_history': self.eef_to_body_pose_proprio_history[:frames_to_save],
            'gripper_angle_proprio_history': self.gripper_angle_proprio_history[:frames_to_save],
            'eef_link_proprio_history': self.eef_link_proprio_history[:frames_to_save],
            **({
                'robot_joint_proprio_history': self.robot_joint_proprio_history[:frames_to_save],
                'eef_joint_proprio_history': self.eef_joint_proprio_history[:frames_to_save],
            } if self.embodiment != 'human' else {}),
            **({
                'velocity_commands_history': self.velocity_commands_history[:frames_to_save],
                'imu_proprio_history': self.imu_proprio_history[:frames_to_save],
            } if self.embodiment == 'g1_inspire_wb' else {}),
            **({
                'left_touch_history': self.left_touch_history[:frames_to_save],
                'right_touch_history': self.right_touch_history[:frames_to_save],
            } if self.enable_inspire_touch_collection else {}),
            'eef_joints_target_pose_history': self.eef_joints_target_pose_history[:frames_to_save],
            'use_wrist_camera': self.args.use_wrist_camera,
            'use_dual_wrist_camera': self.use_dual_wrist_camera if self.args.use_wrist_camera else False,
            'save_video': self.args.save_video,
            'control_freq': self.args.control_freq,
            'is_realtime_save': True,
            'is_last_save': False
        }
        # Pop the saved data from memory (now done in background thread)
        for _ in range(frames_to_save):
            self.main_cam_image_history.pop(0)
            if self.args.use_wrist_camera:
                if self.use_dual_wrist_camera:
                    self.wrist_left_cam_image_history.pop(0)
                    self.wrist_right_cam_image_history.pop(0)
                else:
                    self.wrist_cam_image_history.pop(0)
            self.body_pose_history.pop(0)
            self.delta_body_pose_history.pop(0)
            self.eef_pose_history.pop(0)
            self.delta_eef_pose_history.pop(0)
            self.eef_to_body_pose_history.pop(0)
            self.gripper_angle_history.pop(0)
            self.delta_gripper_angle_history.pop(0)
            self.head_camera_to_init_pose_history.pop(0)
            self.head_cam_timestamp_history.pop(0)
            self.body_pose_proprio_history.pop(0)
            self.eef_pose_proprio_history.pop(0)
            self.eef_to_body_pose_proprio_history.pop(0)
            self.gripper_angle_proprio_history.pop(0)
            self.eef_link_proprio_history.pop(0)
            if self.embodiment != 'human':
                self.robot_joint_proprio_history.pop(0)
                self.eef_joint_proprio_history.pop(0)
            if self.embodiment == 'g1_inspire_wb':
                self.velocity_commands_history.pop(0)
                self.imu_proprio_history.pop(0)
            if self.enable_inspire_touch_collection:
                self.left_touch_history.pop(0)
                self.right_touch_history.pop(0)
            self.eef_joints_target_pose_history.pop(0)

        print(f"Background: Popped {frames_to_save} frames from memory")

        # Convert to numpy arrays and save to HDF5
        self.convert_and_save_to_hdf5(save_data)
        # Update counters
        self.frames_saved_count += frames_to_save

    def handle_episode_completion(self):
        """Handle episode completion automatically - no flush_trajectory needed"""
        episode_num = self.current_episode_num
        episode_duration = time.time() - self.episode_start_time
        current_frame_count = len(self.eef_pose_history)

        if current_frame_count > 0:
            print(f"Background: Episode {episode_num} completed ({episode_duration:.1f}s). Saved {self.frames_saved_count} frames, saving final {current_frame_count} frames...")

            # Snapshot all lists under lock to avoid race with update_trajectories
            with self.trajectory_lock:
                n = current_frame_count
                save_data = {
                    'episode_num': episode_num,
                    'save_path': self.exp_data_folder,
                    'main_cam_image_history': self.main_cam_image_history[:n],
                    'wrist_cam_image_history': self.wrist_cam_image_history[:n],
                    'wrist_left_cam_image_history': self.wrist_left_cam_image_history[:n],
                    'wrist_right_cam_image_history': self.wrist_right_cam_image_history[:n],
                    'body_pose_history': self.body_pose_history[:n],
                    'delta_body_pose_history': self.delta_body_pose_history[:n],
                    'eef_pose_history': self.eef_pose_history[:n],
                    'delta_eef_pose_history': self.delta_eef_pose_history[:n],
                    'eef_to_body_pose_history': self.eef_to_body_pose_history[:n],
                    'gripper_angle_history': self.gripper_angle_history[:n],
                    'delta_gripper_angle_history': self.delta_gripper_angle_history[:n],
                    'head_camera_to_init_pose_history': self.head_camera_to_init_pose_history[:n],
                    'head_cam_timestamp_history': self.head_cam_timestamp_history[:n],
                    'body_pose_proprio_history': self.body_pose_proprio_history[:n],
                    'eef_pose_proprio_history': self.eef_pose_proprio_history[:n],
                    'eef_to_body_pose_proprio_history': self.eef_to_body_pose_proprio_history[:n],
                    'gripper_angle_proprio_history': self.gripper_angle_proprio_history[:n],
                    'eef_link_proprio_history': self.eef_link_proprio_history[:n],
                    **({
                        'robot_joint_proprio_history': self.robot_joint_proprio_history[:n],
                        'eef_joint_proprio_history': self.eef_joint_proprio_history[:n],
                    } if self.embodiment != 'human' else {}),
                    **({
                        'velocity_commands_history': self.velocity_commands_history[:n],
                        'imu_proprio_history': self.imu_proprio_history[:n],
                    } if self.embodiment == 'g1_inspire_wb' else {}),
                    **({
                        'left_touch_history': self.left_touch_history[:n],
                        'right_touch_history': self.right_touch_history[:n],
                    } if self.enable_inspire_touch_collection else {}),
                    'eef_joints_target_pose_history': self.eef_joints_target_pose_history[:n],
                    'use_wrist_camera': self.args.use_wrist_camera,
                    'use_dual_wrist_camera': self.use_dual_wrist_camera if self.args.use_wrist_camera else False,
                    'save_video': self.args.save_video,
                    'control_freq': self.args.control_freq,
                    'is_realtime_save': True,
                    'is_last_save': True
                }

            self.convert_and_save_to_hdf5(save_data)
        else:
            print(f"Background: Episode {episode_num} completed ({episode_duration:.1f}s). All {self.frames_saved_count} frames already saved, generating video from complete HDF5 dataset...")
            # Generate video directly from HDF5 file
            self.generate_videos_from_hdf5(self.exp_data_folder, episode_num, self.args.control_freq, self.args.use_wrist_camera)

        # Reset episode tracking
        if hasattr(self, 'current_episode_num'):
            delattr(self, 'current_episode_num')
        if hasattr(self, 'episode_start_time'):
            delattr(self, 'episode_start_time')
        if hasattr(self, 'frames_saved_count'):
            delattr(self, 'frames_saved_count')

        # Clear memory immediately
        self.reset_trajectories()
        # force garbage collection to clean up memory after each episode
        gc.collect()

    def convert_and_save_to_hdf5(self, save_data):
        """Convert lists to numpy arrays and save to HDF5 file"""
        print(f"Background: Converting {len(save_data['main_cam_image_history'])} frames to numpy arrays...")

        # Convert raw lists to numpy arrays in background thread
        numpy_data = {
            'episode_num': save_data['episode_num'],
            'save_path': save_data['save_path'],
            'main_cam_image_history': np.array(save_data['main_cam_image_history']),
            'wrist_cam_image_history': np.array(save_data['wrist_cam_image_history']),
            'wrist_left_cam_image_history': np.array(save_data['wrist_left_cam_image_history']),
            'wrist_right_cam_image_history': np.array(save_data['wrist_right_cam_image_history']),
            'body_pose_history': np.array(save_data['body_pose_history']),
            'delta_body_pose_history': np.array(save_data['delta_body_pose_history']),
            'eef_pose_history': np.array(save_data['eef_pose_history']),
            'delta_eef_pose_history': np.array(save_data['delta_eef_pose_history']),
            'eef_to_body_pose_history': np.array(save_data['eef_to_body_pose_history']),
            'gripper_angle_history': np.array(save_data['gripper_angle_history']),
            'delta_gripper_angle_history': np.array(save_data['delta_gripper_angle_history']),
            'head_camera_to_init_pose_history': np.array(save_data['head_camera_to_init_pose_history']),
            'head_cam_timestamp_history': np.array(save_data['head_cam_timestamp_history']),
            'body_pose_proprio_history': np.array(save_data['body_pose_proprio_history']),
            'eef_pose_proprio_history': np.array(save_data['eef_pose_proprio_history']),
            'eef_to_body_pose_proprio_history': np.array(save_data['eef_to_body_pose_proprio_history']),
            'gripper_angle_proprio_history': np.array(save_data['gripper_angle_proprio_history']),
            'eef_link_proprio_history': np.array(save_data['eef_link_proprio_history']),
            **({
                'robot_joint_proprio_history': np.array(save_data['robot_joint_proprio_history']),
                'eef_joint_proprio_history': np.array(save_data['eef_joint_proprio_history']),
            } if self.embodiment != 'human' else {}),
            **({
                'velocity_commands_history': np.array(save_data['velocity_commands_history']),
                'imu_proprio_history': np.array(save_data['imu_proprio_history']),
            } if self.embodiment == 'g1_inspire_wb' else {}),
            **({
                'left_touch_history': np.array(save_data['left_touch_history']),
                'right_touch_history': np.array(save_data['right_touch_history']),
            } if self.enable_inspire_touch_collection else {}),
            'eef_joints_target_pose_history': np.array(save_data['eef_joints_target_pose_history']),
            'use_wrist_camera': save_data['use_wrist_camera'],
            'use_dual_wrist_camera': save_data['use_dual_wrist_camera'],
            'save_video': save_data['save_video'],
            'control_freq': save_data['control_freq'],
            'is_realtime_save': save_data['is_realtime_save'],
            'is_last_save': save_data['is_last_save']
        }
        print(f"Background: Numpy conversion completed for episode {save_data['episode_num']}")
        # Get embodiment masks
        self.get_embodiment_masks()

        # Save HDF5 file (reuse the existing HDF5 saving logic)
        self.save_to_hdf5_file(numpy_data)
        # Save videos (only for the last save of each episode to avoid too many video files)
        if numpy_data['save_video'] and numpy_data.get('is_last_save', False):
            # Generate videos by reading from the saved HDF5 file
            self.generate_videos_from_hdf5(numpy_data['save_path'], numpy_data['episode_num'], numpy_data['control_freq'], numpy_data['use_wrist_camera'])

        # Clean up the saved data from memory
        episode_num = numpy_data['episode_num']
        del save_data
        gc.collect()

        print(f"Background: Episode {episode_num} saved successfully.")

    def _encode_png_bytes(self, rgb_img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> 1D uint8 PNG bytes (as np.ndarray)."""
        # OpenCV encodes BGR by convention; keep roundtrip consistent:
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.png', bgr)
        if not ok:
            raise RuntimeError("cv2.imencode('.png') failed")
        return buf  # np.ndarray(dtype=uint8), 1D

    def _decode_png_bytes(self, png_bytes: np.ndarray) -> np.ndarray:
        """1D uint8 PNG bytes -> RGB uint8 HxWx3."""
        bgr = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2.imdecode failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _encode_jpg_bytes(self, rgb_img: np.ndarray, quality: int = 90) -> np.ndarray:
        """RGB uint8 HxWx3 -> 1D uint8 JPEG bytes."""
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            '.jpg', bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        )
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg') failed")
        return buf  # np.ndarray(dtype=uint8), 1D

    def _decode_jpg_bytes(self, jpg_bytes: np.ndarray) -> np.ndarray:
        """1D uint8 JPEG bytes -> RGB uint8 HxWx3."""
        bgr = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2.imdecode failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



    def _episode_metadata(self, numpy_data):
        """Describe this collector's output; do not infer provenance for old data."""
        hand_type = 'human' if self.embodiment == 'human' else self.inspire_hand_type
        tactile_format = 'none'
        if self.enable_inspire_touch_collection:
            width = 29 if hand_type == 'f1' else 1062
            for side in ('left', 'right'):
                touch = numpy_data[f'{side}_touch_history']
                if touch.ndim != 2 or touch.shape[1] != width:
                    raise ValueError(f'{hand_type} collector requires {width}-D {side} touch')
            tactile_format = 'f1_force_direction_proximity_29_v1' if hand_type == 'f1' else 'dftp_normalized_1062_v1'
        camera_profile = 'head-only'
        if numpy_data['use_wrist_camera']:
            camera_profile = 'head-and-wrists' if numpy_data['use_dual_wrist_camera'] else 'head-and-single-wrist'
        return {
            'schema_version': 'htd_v1',
            'coordinate_convention': 'htd_g1_legacy_v1' if self.embodiment == 'g1_inspire_wb' else 'htd_human_avp_v1',
            'hand_type': hand_type,
            'tactile_format': tactile_format,
            'camera_profile': camera_profile,
            'sim': False,
            'embodiment': self.args.embodiment,
            'compress': self.args.image_save_format != 'numpy',
            'frequency': self.args.control_freq,
        }

    def save_to_hdf5_file(self, numpy_data):
        """Save numpy data to HDF5 file - extracted from original background_save_worker"""
        metadata = self._episode_metadata(numpy_data)
        path = os.path.join(numpy_data['save_path'], f"episode_{numpy_data['episode_num']}.hdf5")
        existed = os.path.exists(path)
        with h5py.File(path, 'a') as f:
            if existed:
                for name, value in metadata.items():
                    if name not in f.attrs or f.attrs[name] != value:
                        raise ValueError(f'Episode metadata changed between batches: {name}')
            else:
                for name, value in metadata.items():
                    f.attrs[name] = value

            # Create groups only if they don't exist
            if 'observations' not in f:
                obs_group = f.create_group('observations')
                # Create resizable datasets with correct maxshape
                obs_group.create_dataset('head_cam_timestamp', data=numpy_data['head_cam_timestamp_history'], maxshape=(None,))
                image_group = obs_group.create_group('images')

                use_dual_wrist = numpy_data.get('use_dual_wrist_camera', False)
                # Save either PNG images OR raw numpy arrays, not both
                if self.args.image_save_format != "numpy":
                    # Save PNG images only (using main/wrist dataset names)
                    vlen_u8 = h5py.vlen_dtype(np.dtype('uint8'))
                    if self.args.image_save_format == "png":
                    # Encode this batch to PNG bytes
                        main_list = [self._encode_png_bytes(img) for img in numpy_data['main_cam_image_history']]
                    else:
                        main_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['main_cam_image_history']]
                    image_group.create_dataset(
                        'main',
                        data=main_list,
                        maxshape=(None,),
                        dtype=vlen_u8,
                    )
                    # wrist (optional)
                    if numpy_data['use_wrist_camera']:
                        if use_dual_wrist:
                            if len(numpy_data['wrist_left_cam_image_history']) > 0 and len(numpy_data['wrist_right_cam_image_history']) > 0:
                                if self.args.image_save_format == "png":
                                    wrist_left_list = [self._encode_png_bytes(img) for img in numpy_data['wrist_left_cam_image_history']]
                                    wrist_right_list = [self._encode_png_bytes(img) for img in numpy_data['wrist_right_cam_image_history']]
                                else:
                                    wrist_left_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['wrist_left_cam_image_history']]
                                    wrist_right_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['wrist_right_cam_image_history']]
                                image_group.create_dataset('wrist_left', data=wrist_left_list, maxshape=(None,), dtype=vlen_u8)
                                image_group.create_dataset('wrist_right', data=wrist_right_list, maxshape=(None,), dtype=vlen_u8)
                        else:
                            if len(numpy_data['wrist_cam_image_history']) > 0:
                                if self.args.image_save_format == "png":
                                    wrist_list = [self._encode_png_bytes(img) for img in numpy_data['wrist_cam_image_history']]
                                else:
                                    wrist_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['wrist_cam_image_history']]
                                image_group.create_dataset('wrist', data=wrist_list, maxshape=(None,), dtype=vlen_u8)
                else:
                    # Save raw numpy arrays only (original behavior)
                    image_group.create_dataset('main', data=numpy_data['main_cam_image_history'], maxshape=(None, numpy_data['main_cam_image_history'].shape[1], numpy_data['main_cam_image_history'].shape[2], numpy_data['main_cam_image_history'].shape[3]), compression="gzip", compression_opts=4)
                    # Only create wrist dataset if wrist camera is used and has data
                    if numpy_data['use_wrist_camera']:
                        if use_dual_wrist:
                            if len(numpy_data['wrist_left_cam_image_history']) > 0 and len(numpy_data['wrist_right_cam_image_history']) > 0:
                                image_group.create_dataset('wrist_left', data=numpy_data['wrist_left_cam_image_history'], maxshape=(None, numpy_data['wrist_left_cam_image_history'].shape[1], numpy_data['wrist_left_cam_image_history'].shape[2], numpy_data['wrist_left_cam_image_history'].shape[3]), compression="gzip", compression_opts=4)
                                image_group.create_dataset('wrist_right', data=numpy_data['wrist_right_cam_image_history'], maxshape=(None, numpy_data['wrist_right_cam_image_history'].shape[1], numpy_data['wrist_right_cam_image_history'].shape[2], numpy_data['wrist_right_cam_image_history'].shape[3]), compression="gzip", compression_opts=4)
                        else:
                            if len(numpy_data['wrist_cam_image_history']) > 0:
                                image_group.create_dataset('wrist', data=numpy_data['wrist_cam_image_history'], maxshape=(None, numpy_data['wrist_cam_image_history'].shape[1], numpy_data['wrist_cam_image_history'].shape[2], numpy_data['wrist_cam_image_history'].shape[3]), compression="gzip", compression_opts=4)
                proprio_group = obs_group.create_group('proprioceptions')
                # Proprioceptive state history (from embodiment feedback)
                proprio_group.create_dataset('body_proprio', data=numpy_data['body_pose_proprio_history'], maxshape=(None, numpy_data['body_pose_proprio_history'].shape[1]))
                proprio_group.create_dataset('eef_proprio', data=numpy_data['eef_pose_proprio_history'], maxshape=(None, numpy_data['eef_pose_proprio_history'].shape[1]))
                proprio_group.create_dataset('eef_to_body_proprio', data=numpy_data['eef_to_body_pose_proprio_history'], maxshape=(None, numpy_data['eef_to_body_pose_proprio_history'].shape[1]))
                proprio_group.create_dataset('gripper_proprio', data=numpy_data['gripper_angle_proprio_history'], maxshape=(None, numpy_data['gripper_angle_proprio_history'].shape[1]))
                proprio_other_group = proprio_group.create_group('other')
                proprio_other_group.create_dataset('eef_link_proprio', data=numpy_data['eef_link_proprio_history'], maxshape=(None, numpy_data['eef_link_proprio_history'].shape[1]))
                if self.embodiment != 'human' and 'robot_joint_proprio_history' in numpy_data and 'eef_joint_proprio_history' in numpy_data:
                    proprio_other_group.create_dataset('robot_joint_proprio', data=numpy_data['robot_joint_proprio_history'], maxshape=(None, numpy_data['robot_joint_proprio_history'].shape[1]))
                    proprio_other_group.create_dataset('eef_joint_proprio', data=numpy_data['eef_joint_proprio_history'], maxshape=(None, numpy_data['eef_joint_proprio_history'].shape[1]))
                if self.embodiment == 'g1_inspire_wb' and 'imu_proprio_history' in numpy_data:
                    proprio_other_group.create_dataset('imu_proprio', data=numpy_data['imu_proprio_history'], maxshape=(None, numpy_data['imu_proprio_history'].shape[1]))
                if self.enable_inspire_touch_collection and 'left_touch_history' in numpy_data and 'right_touch_history' in numpy_data:
                    proprio_other_group.create_dataset('left_touch', data=numpy_data['left_touch_history'], maxshape=(None, numpy_data['left_touch_history'].shape[1]))
                    proprio_other_group.create_dataset('right_touch', data=numpy_data['right_touch_history'], maxshape=(None, numpy_data['right_touch_history'].shape[1]))
            else:
                # Append new data to existing datasets
                obs_group = f['observations']
                current_size = obs_group['head_cam_timestamp'].shape[0]
                new_data_size = len(numpy_data['head_cam_timestamp_history'])

                # Resize datasets to accommodate new data
                obs_group['head_cam_timestamp'].resize((current_size + new_data_size,))
                obs_group['head_cam_timestamp'][current_size:] = numpy_data['head_cam_timestamp_history']

                use_dual_wrist = numpy_data.get('use_dual_wrist_camera', False)
                # Append either PNG images OR raw numpy arrays, not both
                if self.args.image_save_format != "numpy":
                    if self.args.image_save_format == "png":
                        main_list = [self._encode_png_bytes(img) for img in numpy_data['main_cam_image_history']]
                    else:
                        main_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['main_cam_image_history']]
                    if 'main' in obs_group['images']:
                        obs_group['images/main'].resize((current_size + new_data_size,))
                        for i, buf in enumerate(main_list):
                            obs_group['images/main'][current_size + i] = buf

                    # wrist (only if dataset exists)
                    if numpy_data['use_wrist_camera']:
                        if use_dual_wrist and 'wrist_left' in obs_group['images'] and 'wrist_right' in obs_group['images']:
                            if len(numpy_data['wrist_left_cam_image_history']) > 0 and len(numpy_data['wrist_right_cam_image_history']) > 0:
                                if self.args.image_save_format == "png":
                                    wrist_left_list = [self._encode_png_bytes(img) for img in numpy_data['wrist_left_cam_image_history']]
                                    wrist_right_list = [self._encode_png_bytes(img) for img in numpy_data['wrist_right_cam_image_history']]
                                else:
                                    wrist_left_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['wrist_left_cam_image_history']]
                                    wrist_right_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['wrist_right_cam_image_history']]
                                obs_group['images/wrist_left'].resize((current_size + new_data_size,))
                                obs_group['images/wrist_right'].resize((current_size + new_data_size,))
                                for i, buf in enumerate(wrist_left_list):
                                    obs_group['images/wrist_left'][current_size + i] = buf
                                for i, buf in enumerate(wrist_right_list):
                                    obs_group['images/wrist_right'][current_size + i] = buf
                        elif 'wrist' in obs_group['images'] and len(numpy_data['wrist_cam_image_history']) > 0:
                            if self.args.image_save_format == "png":
                                wrist_list = [self._encode_png_bytes(img) for img in numpy_data['wrist_cam_image_history']]
                            else:
                                wrist_list = [self._encode_jpg_bytes(img, quality=self.args.jpg_save_quality) for img in numpy_data['wrist_cam_image_history']]
                            obs_group['images/wrist'].resize((current_size + new_data_size,))
                            for i, buf in enumerate(wrist_list):
                                obs_group['images/wrist'][current_size + i] = buf
                else:
                    # Append raw numpy arrays only (original behavior)
                    obs_group['images/main'].resize((current_size + new_data_size, numpy_data['main_cam_image_history'].shape[1], numpy_data['main_cam_image_history'].shape[2], numpy_data['main_cam_image_history'].shape[3]))
                    obs_group['images/main'][current_size:] = numpy_data['main_cam_image_history']

                    # Only update wrist dataset if it exists and wrist camera is used
                    if numpy_data['use_wrist_camera']:
                        if use_dual_wrist and 'wrist_left' in obs_group['images'] and 'wrist_right' in obs_group['images']:
                            if len(numpy_data['wrist_left_cam_image_history']) > 0 and len(numpy_data['wrist_right_cam_image_history']) > 0:
                                obs_group['images/wrist_left'].resize((current_size + new_data_size, numpy_data['wrist_left_cam_image_history'].shape[1], numpy_data['wrist_left_cam_image_history'].shape[2], numpy_data['wrist_left_cam_image_history'].shape[3]))
                                obs_group['images/wrist_right'].resize((current_size + new_data_size, numpy_data['wrist_right_cam_image_history'].shape[1], numpy_data['wrist_right_cam_image_history'].shape[2], numpy_data['wrist_right_cam_image_history'].shape[3]))
                                obs_group['images/wrist_left'][current_size:] = numpy_data['wrist_left_cam_image_history']
                                obs_group['images/wrist_right'][current_size:] = numpy_data['wrist_right_cam_image_history']
                        elif 'wrist' in obs_group['images'] and len(numpy_data['wrist_cam_image_history']) > 0:
                            obs_group['images/wrist'].resize((current_size + new_data_size, numpy_data['wrist_cam_image_history'].shape[1], numpy_data['wrist_cam_image_history'].shape[2], numpy_data['wrist_cam_image_history'].shape[3]))
                            obs_group['images/wrist'][current_size:] = numpy_data['wrist_cam_image_history']

                # Append proprioceptive state history
                obs_group['proprioceptions/body_proprio'].resize((current_size + new_data_size, numpy_data['body_pose_proprio_history'].shape[1]))
                obs_group['proprioceptions/body_proprio'][current_size:] = numpy_data['body_pose_proprio_history']

                obs_group['proprioceptions/eef_proprio'].resize((current_size + new_data_size, numpy_data['eef_pose_proprio_history'].shape[1]))
                obs_group['proprioceptions/eef_proprio'][current_size:] = numpy_data['eef_pose_proprio_history']

                obs_group['proprioceptions/eef_to_body_proprio'].resize((current_size + new_data_size, numpy_data['eef_to_body_pose_proprio_history'].shape[1]))
                obs_group['proprioceptions/eef_to_body_proprio'][current_size:] = numpy_data['eef_to_body_pose_proprio_history']

                obs_group['proprioceptions/gripper_proprio'].resize((current_size + new_data_size, numpy_data['gripper_angle_proprio_history'].shape[1]))
                obs_group['proprioceptions/gripper_proprio'][current_size:] = numpy_data['gripper_angle_proprio_history']

                obs_group['proprioceptions/other/eef_link_proprio'].resize((current_size + new_data_size, numpy_data['eef_link_proprio_history'].shape[1]))
                obs_group['proprioceptions/other/eef_link_proprio'][current_size:] = numpy_data['eef_link_proprio_history']
                if self.embodiment != 'human' and 'robot_joint_proprio_history' in numpy_data and 'eef_joint_proprio_history' in numpy_data:
                    if 'robot_joint_proprio' in obs_group['proprioceptions/other'] and 'eef_joint_proprio' in obs_group['proprioceptions/other']:
                        obs_group['proprioceptions/other/robot_joint_proprio'].resize((current_size + new_data_size, numpy_data['robot_joint_proprio_history'].shape[1]))
                        obs_group['proprioceptions/other/robot_joint_proprio'][current_size:] = numpy_data['robot_joint_proprio_history']
                        obs_group['proprioceptions/other/eef_joint_proprio'].resize((current_size + new_data_size, numpy_data['eef_joint_proprio_history'].shape[1]))
                        obs_group['proprioceptions/other/eef_joint_proprio'][current_size:] = numpy_data['eef_joint_proprio_history']
                if self.embodiment == 'g1_inspire_wb' and 'imu_proprio_history' in numpy_data and 'imu_proprio' in obs_group['proprioceptions/other']:
                    obs_group['proprioceptions/other/imu_proprio'].resize((current_size + new_data_size, numpy_data['imu_proprio_history'].shape[1]))
                    obs_group['proprioceptions/other/imu_proprio'][current_size:] = numpy_data['imu_proprio_history']
                if self.enable_inspire_touch_collection and 'left_touch_history' in numpy_data and 'right_touch_history' in numpy_data:
                    if 'left_touch' in obs_group['proprioceptions/other'] and 'right_touch' in obs_group['proprioceptions/other']:
                        obs_group['proprioceptions/other/left_touch'].resize((current_size + new_data_size, numpy_data['left_touch_history'].shape[1]))
                        obs_group['proprioceptions/other/left_touch'][current_size:] = numpy_data['left_touch_history']
                        obs_group['proprioceptions/other/right_touch'].resize((current_size + new_data_size, numpy_data['right_touch_history'].shape[1]))
                        obs_group['proprioceptions/other/right_touch'][current_size:] = numpy_data['right_touch_history']

            if 'actions' not in f:
                action_group = f.create_group('actions')
                action_group.create_dataset('body', data=numpy_data['body_pose_history'], maxshape=(None, numpy_data['body_pose_history'].shape[1]))
                action_group.create_dataset('delta_body', data=numpy_data['delta_body_pose_history'], maxshape=(None, numpy_data['delta_body_pose_history'].shape[1]))
                action_group.create_dataset('eef', data=numpy_data['eef_pose_history'], maxshape=(None, numpy_data['eef_pose_history'].shape[1]))
                action_group.create_dataset('eef_to_body', data=numpy_data['eef_to_body_pose_history'], maxshape=(None, numpy_data['eef_to_body_pose_history'].shape[1]))
                action_group.create_dataset('delta_eef', data=numpy_data['delta_eef_pose_history'], maxshape=(None, numpy_data['delta_eef_pose_history'].shape[1]))
                action_group.create_dataset('gripper', data=numpy_data['gripper_angle_history'], maxshape=(None, numpy_data['gripper_angle_history'].shape[1]))
                action_group.create_dataset('delta_gripper', data=numpy_data['delta_gripper_angle_history'], maxshape=(None, numpy_data['delta_gripper_angle_history'].shape[1]))
                action_group.create_dataset('eef_joints_target', data=numpy_data['eef_joints_target_pose_history'], maxshape=(None, numpy_data['eef_joints_target_pose_history'].shape[1]))
                if self.embodiment == 'g1_inspire_wb' and 'velocity_commands_history' in numpy_data:
                    action_group.create_dataset('velocity_commands', data=numpy_data['velocity_commands_history'], maxshape=(None, numpy_data['velocity_commands_history'].shape[1]))
            else:
                # Resize and update existing action datasets
                action_group = f['actions']
                current_size = action_group['body'].shape[0]
                new_data_size = len(numpy_data['body_pose_history'])

                action_group['body'].resize((current_size + new_data_size, numpy_data['body_pose_history'].shape[1]))
                action_group['body'][current_size:] = numpy_data['body_pose_history']

                action_group['delta_body'].resize((current_size + new_data_size, numpy_data['delta_body_pose_history'].shape[1]))
                action_group['delta_body'][current_size:] = numpy_data['delta_body_pose_history']

                action_group['eef'].resize((current_size + new_data_size, numpy_data['eef_pose_history'].shape[1]))
                action_group['eef'][current_size:] = numpy_data['eef_pose_history']

                action_group['eef_to_body'].resize((current_size + new_data_size, numpy_data['eef_to_body_pose_history'].shape[1]))
                action_group['eef_to_body'][current_size:] = numpy_data['eef_to_body_pose_history']

                action_group['delta_eef'].resize((current_size + new_data_size, numpy_data['delta_eef_pose_history'].shape[1]))
                action_group['delta_eef'][current_size:] = numpy_data['delta_eef_pose_history']

                action_group['gripper'].resize((current_size + new_data_size, numpy_data['gripper_angle_history'].shape[1]))
                action_group['gripper'][current_size:] = numpy_data['gripper_angle_history']

                action_group['delta_gripper'].resize((current_size + new_data_size, numpy_data['delta_gripper_angle_history'].shape[1]))
                action_group['delta_gripper'][current_size:] = numpy_data['delta_gripper_angle_history']

                action_group['eef_joints_target'].resize((current_size + new_data_size, numpy_data['eef_joints_target_pose_history'].shape[1]))
                action_group['eef_joints_target'][current_size:] = numpy_data['eef_joints_target_pose_history']
                if self.embodiment == 'g1_inspire_wb' and 'velocity_commands_history' in numpy_data and 'velocity_commands' in action_group:
                    action_group['velocity_commands'].resize((current_size + new_data_size, numpy_data['velocity_commands_history'].shape[1]))
                    action_group['velocity_commands'][current_size:] = numpy_data['velocity_commands_history']

            if 'masks' not in f:
                mask_group = f.create_group('masks')
                mask_group.create_dataset('img_main', data=self.img_main_mask)
                mask_group.create_dataset('img_wrist', data=self.img_wrist_mask)
                mask_group.create_dataset('proprio_body', data=self.proprio_body_mask)
                mask_group.create_dataset('proprio_eef', data=self.proprio_eef_mask)
                mask_group.create_dataset('proprio_gripper', data=self.proprio_gripper_mask)
                mask_group.create_dataset('proprio_other', data=self.proprio_other_mask)
                mask_group.create_dataset('act_body', data=self.act_body_mask)
                mask_group.create_dataset('act_eef', data=self.act_eef_mask)
                mask_group.create_dataset('act_gripper', data=self.act_gripper_mask)
            else:
                # Update existing mask datasets (masks don't change size, so no resize needed)
                mask_group = f['masks']
                mask_group['img_main'][...] = self.img_main_mask
                mask_group['img_wrist'][...] = self.img_wrist_mask
                mask_group['proprio_body'][...] = self.proprio_body_mask
                mask_group['proprio_eef'][...] = self.proprio_eef_mask
                mask_group['proprio_gripper'][...] = self.proprio_gripper_mask
                mask_group['proprio_other'][...] = self.proprio_other_mask
                mask_group['act_body'][...] = self.act_body_mask
                mask_group['act_eef'][...] = self.act_eef_mask
                mask_group['act_gripper'][...] = self.act_gripper_mask

            if 'camera_poses' not in f:
                camera_group = f.create_group('camera_poses')
                camera_group.create_dataset('head_camera_to_init', data=numpy_data['head_camera_to_init_pose_history'], maxshape=(None, numpy_data['head_camera_to_init_pose_history'].shape[1]))
            else:
                # Resize and update existing camera pose dataset
                camera_group = f['camera_poses']
                current_size = camera_group['head_camera_to_init'].shape[0]
                new_data_size = len(numpy_data['head_camera_to_init_pose_history'])

                camera_group['head_camera_to_init'].resize((current_size + new_data_size, numpy_data['head_camera_to_init_pose_history'].shape[1]))
                camera_group['head_camera_to_init'][current_size:] = numpy_data['head_camera_to_init_pose_history']

    def wait_for_save_completion(self, timeout=None):
        """Wait for background save operations to complete"""
        if timeout is None:
            # Wait indefinitely
            while self.saving_in_progress:
                time.sleep(0.1)
        else:
            # Wait with timeout
            start_time = time.time()
            while self.saving_in_progress and (time.time() - start_time) < timeout:
                time.sleep(0.1)

    @property
    def increment_counter(self):
        self.record_episode_counter += 1
        return self.record_episode_counter
    
    def run(self):
        """Main loop for data collection and teleoperation publishing"""
        print("=" * 60)
        print("HTD Data Collector with DDS Teleoperation")
        print("=" * 60)
        print(f"Control frequency: {self.args.control_freq} Hz")
        print(f"Embodiment: {self.args.embodiment}")
        print(f"Collect data: {self.args.collect_data}")
        print(f"Collect Inspire touch: {self.args.collect_inspire_touch} (hand_type={self.inspire_hand_type})")
        print("=" * 60)
        print("\nPress Ctrl+C to stop\n")
        head_camera_streaming_thread = threading.Thread(target=self.head_camera_stream_thread, daemon=True)
        head_camera_streaming_thread.start()
        if self.args.use_wrist_camera:
            if self.use_dual_wrist_camera:
                wrist_left_thread = threading.Thread(target=self.wrist_camera_left_stream_thread, daemon=True)
                wrist_right_thread = threading.Thread(target=self.wrist_camera_right_stream_thread, daemon=True)
                wrist_left_thread.start()
                wrist_right_thread.start()
            else:
                wrist_camera_streaming_thread = threading.Thread(target=self.wrist_camera_stream_thread, daemon=True)
                wrist_camera_streaming_thread.start()

        # Main control loop
        loop_period = 1.0 / self.args.control_freq

        try:
            while self.running:
                loop_start = time.time()

                # Call data collection and publishing
                self.collect_data()

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

    def safe_mat_update(self, prev_mat, mat):
        # return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0).
        det = np.linalg.det(mat)
        if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
            return prev_mat, False
        return mat, True

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

    def rot_mat_to_rpy_zxy(self, R):
        """
        Convert a rotation matrix (RzRxRy) to Euler angles with ZXY order (first rotate with y, then x, then z).
        This method is more numerically stable, especially near singularities.
        """
        sx = R[2, 1]
        singular_threshold = 1e-6
        cx = np.sqrt(R[2, 0]**2 + R[2, 2]**2)

        if cx < singular_threshold:
            x = np.arctan2(sx, cx)
            y = np.arctan2(R[0, 2], R[0, 0])
            z = 0
        else:
            x = np.arctan2(sx, cx)
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([x, y, z])

    def close(self):
        """Clean shutdown of data collector"""
        self.running = False

        # Wait for any pending saves to complete before shutting down
        if self.args.collect_data:
            print("Waiting for background saves to complete...")
            self.wait_for_save_completion(timeout=30)  # Wait up to 30 seconds

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

        # Clean up shared memory
        if hasattr(self, 'shm'):
            try:
                self.shm.close()
                self.shm.unlink()
                print('Shared memory cleaned up')
            except Exception as e:
                print(f"Error cleaning up shared memory: {e}")

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

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        prog="HTD data collection with Apple Vision Pro and Vuer"
    )
    parser.add_argument('--embodiment',type=str, default="human", choices=["human", "g1_inspire_wb"], help='embodiment')
    parser.add_argument("--head_camera_type", type=int, default=1, help="0=realsense, 1=stereo rgb camera")
    parser.add_argument("--head_camera_res", type=str, default="480p", choices=["480p", "720p", "1080p"], help="head camera resolution: 480p, 720p, or 1080p")
    parser.add_argument("--use_wrist_camera", type=str2bool, default=False, help="whether to use wrist camera")
    parser.add_argument("--wrist_camera_res", type=str, default="480p", choices=["480p", "720p", "1080p"], help="wrist camera resolution: 480p, 720p, or 1080p")
    parser.add_argument("--desired_stream_fps", type=int, default=60, help="desired camera streaming fps")
    parser.add_argument("--control_freq", type=int, default=30, help="frequency to record human data")
    parser.add_argument("--collect_data", type=str2bool, default=True, help="whether to collect data")
    parser.add_argument("--collect_inspire_touch", type=str2bool, default=False, help="whether to collect Inspire touch data (requires --collect_data=True)")
    parser.add_argument("--inspire_hand_type", type=str, default="dftp", choices=["dftp", "f1"], help="Inspire hand type: dftp or f1 (default: dftp)")
    parser.add_argument("--manipulate_mode", type=int, default=1, help="1: right eef; 2: left eef; 3: bimanual")
    parser.add_argument('--save_video', type=str2bool, default=False, help="whether to collect save videos of camera views when storing the data")
    parser.add_argument('--image_save_format',type=str, default="jpg", choices=["numpy", "png", "jpg"], help='image save format: numpy, png, or jpg')
    parser.add_argument('--jpg_save_quality',type=int, default=95, help='image save quality: 0-100')
    # exp_name
    parser.add_argument("--exp_name", type=str, default='test')
    # if using see-through mode, need to set the following parameters
    parser.add_argument("--see_through_mode", type=str2bool, default=False, help="whether to use see-through mode")
    parser.add_argument("--see_through_mode_ip", type=str, default='127.0.0.1', help="avp ip address for see-through mode")
    parser.add_argument("--see_through_video", type=str2bool, default=False, help="if stream stereo video for see-through mode")
    parser.add_argument('--dds_interface', type=str, default=None,
                       help='Network interface name to use for DDS (e.g., enp4s0, wlxc83a35a4f24d)')

    args = parser.parse_args()
    
    # Create data collector
    htd_data_collector = HTDDataCollector(args)
    try:
        htd_data_collector.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        htd_data_collector.close()
        print("Program exited")
        
