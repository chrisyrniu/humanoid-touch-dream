from dex_retargeting import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class HandType(Enum):
    INSPIRE_HAND_DFTP = "./assets/inspire_hand/inspire_hand_dftp.yml"
    INSPIRE_HAND_F1 = "./assets/inspire_hand/inspire_hand_f1.yml"

class HandRetargeting:
    def __init__(self, hand_type: HandType):
        if hand_type in (HandType.INSPIRE_HAND_DFTP, HandType.INSPIRE_HAND_F1):
            RetargetingConfig.set_default_urdf_dir('./assets')
        config_file_path = Path(hand_type.value)

        try:
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)
                
            if 'left' not in self.cfg or 'right' not in self.cfg:
                raise ValueError("Configuration file must contain 'left' and 'right' keys.")

            left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names
            self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
            self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

            if hand_type == HandType.INSPIRE_HAND_DFTP:
                # DFTP hardware register order.
                self.left_inspire_api_joint_names  = [ 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                                                       'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint' ]
                self.right_inspire_api_joint_names = [ 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                                                       'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]

            elif hand_type == HandType.INSPIRE_HAND_F1:
                # F1 hardware register order: [little, ring, middle, index, thumb_bend, thumb_rotate]
                self.left_inspire_api_joint_names  = [ 'left_little_1_joint', 'left_ring_1_joint', 'left_middle_1_joint',
                                                       'left_index_1_joint', 'left_thumb_2_joint', 'left_thumb_1_joint' ]
                self.right_inspire_api_joint_names = [ 'right_little_1_joint', 'right_ring_1_joint', 'right_middle_1_joint',
                                                       'right_index_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]


        except FileNotFoundError:
            logger_mp.warning(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            logger_mp.warning(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            logger_mp.error(f"An error occurred: {e}")
            raise
