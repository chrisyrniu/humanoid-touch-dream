"""
Utility functions. Code adapted from HPT.
"""

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from act_utils import find_all_hdf5, BatchSampler


HTD_POLICY_EMBODIMENT = 'g1_inspire_wb'


def require_htd_policy_embodiment(root, source=None):
    """Read and validate the sole policy embodiment supported by HTD release."""
    source = source or getattr(root, 'filename', None) or 'HDF5 sample'
    if 'embodiment' not in root.attrs:
        raise ValueError(
            f"{source}: missing required HDF5 'embodiment' attribute; "
            f"expected {HTD_POLICY_EMBODIMENT!r}"
        )

    embodiment = root.attrs['embodiment']
    if isinstance(embodiment, (bytes, np.bytes_)):
        try:
            embodiment = embodiment.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source}: malformed HDF5 'embodiment' attribute; "
                "expected a UTF-8 scalar string"
            ) from exc
    if not isinstance(embodiment, (str, np.str_)):
        raise ValueError(
            f"{source}: malformed HDF5 'embodiment' attribute "
            f"of type {type(embodiment).__name__}; expected a scalar string"
        )
    embodiment = str(embodiment)
    if embodiment != HTD_POLICY_EMBODIMENT:
        raise ValueError(
            f"{source}: unsupported policy embodiment {embodiment!r}; "
            f"HTD release supports only {HTD_POLICY_EMBODIMENT!r}"
        )
    return embodiment


def compute_dict_mean(epoch_dicts):
    """
    Compute the mean of values in a list of dictionaries that may have different structures.
    For each leaf node (non-dictionary value), computes the mean of all available values across dicts.
    
    Args:
        epoch_dicts: List of dictionaries that may have different structures
        
    Returns:
        Dictionary containing mean values for all leaf nodes that appear in any input dict
    """
    if not epoch_dicts:
        return {}
    
    # Collect all unique keys across all dictionaries
    all_keys = set()
    for d in epoch_dicts:
        all_keys.update(d.keys())
    
    result = {}
    for k in all_keys:
        # Collect all values for this key
        values = []
        for d in epoch_dicts:
            if k in d:
                if isinstance(d[k], dict):
                    # For nested dictionaries, collect all corresponding dicts
                    values.append(d[k])
                else:
                    # For leaf values, include in the mean calculation
                    values.append(d[k])
        
        if not values:
            continue
        # If the values are dictionaries, recurse
        if all(isinstance(v, dict) for v in values):
            result[k] = compute_dict_mean(values)
        # If we have any numeric values, compute their mean
        elif any(isinstance(v.item(), (int, float)) for v in values):
            numeric_values = [v.item() for v in values if isinstance(v.item(), (int, float))]
            if numeric_values:
                result[k] = sum(numeric_values) / len(numeric_values)
                
    return result


def set_seed(seed):
    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_velocity_rebalance_weights(dataset_path_list, episode_ids, episode_len,
                                       vel_weight_x=1.0, vel_weight_y=1.0, vel_weight_yaw=5.0,
                                       vel_threshold=0.1):
    """Compute per-timestep sampling weights based on velocity command activity.

    For each training timestep, the weight is:
        1.0 + (has_x_vel * vel_weight_x) + (has_y_vel * vel_weight_y) + (has_yaw * vel_weight_yaw)
    so timesteps with active velocity components are sampled more frequently.
    """
    total_len = sum(episode_len)
    weights = np.ones(total_len, dtype=np.float64)
    offset = 0
    for i, ep_id in enumerate(episode_ids):
        ep_path = dataset_path_list[ep_id]
        ep_len = episode_len[i]
        try:
            with h5py.File(ep_path, 'r') as f:
                if '/actions/velocity_commands' in f:
                    vel = np.array(f['/actions/velocity_commands'])
                    n = min(ep_len, len(vel))
                    has_x = np.abs(vel[:n, 0]) > vel_threshold
                    has_y = np.abs(vel[:n, 1]) > vel_threshold
                    has_yaw = np.abs(vel[:n, 2]) > vel_threshold
                    weights[offset:offset + n] += has_x * vel_weight_x + has_y * vel_weight_y + has_yaw * vel_weight_yaw
        except Exception as e:
            print(f'[vel_rebalance] Warning: failed to read {ep_path}: {e}')
        offset += ep_len
    active = weights > 1.0
    print(f'[vel_rebalance] {active.sum()}/{total_len} timesteps have elevated weight '
          f'({100*active.mean():.1f}%), mean weight: {weights.mean():.2f}, max: {weights.max():.1f}')
    return weights


def load_data(dataset_dir_l, name_filter, emb_dict, batch_size_train, batch_size_val,
              chunk_size, skip_mirrored_data=False,
              stats_dir_l=None, sample_weights=None, train_ratio=0.95,
              main_cam_size=None, wrist_cam_size=None,
              data_aug=False, grayscale=False, randomize_color=False, min_val_num=1,
              vel_rebalance=False, vel_rebalance_weight_x=1.0, vel_rebalance_weight_y=1.0,
              vel_rebalance_weight_yaw=5.0, vel_rebalance_threshold=0.1, num_workers=8,
              prefetch_factor=1, persistent_workers=True, pin_memory=True):
    if num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    if prefetch_factor < 1:
        raise ValueError('prefetch_factor must be positive')
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [sorted(find_all_hdf5(dataset_dir, skip_mirrored_data)) for dataset_dir in dataset_dir_l]
    dataset_path_list_list = [[n for n in group if name_filter(n)] for group in dataset_path_list_list]
    print('dataset_path_list_list', dataset_path_list_list)
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    num_episodes_l = [len(group) for group in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    shuffled_episode_ids_0 = np.arange(num_episodes_0)
    val_episode_num_0 = max(min_val_num, int((1-train_ratio) * num_episodes_0))
    train_episode_ids_0 = shuffled_episode_ids_0[:-val_episode_num_0]
    val_episode_ids_0 = shuffled_episode_ids_0[-val_episode_num_0:]
    print(f'train_episode_ids_0: {train_episode_ids_0}')
    print(f'val_episode_ids_0: {val_episode_ids_0}')
    
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    print("test on ",dataset_path_list[val_episode_ids_l[0][0]])
    print(f'validation episodes: {[dataset_path_list[idx] for idx in val_episode_ids_0]}')

    _, all_episode_len, embodiment_list = get_norm_stats(dataset_path_list, embodiment_dict=emb_dict)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    print('train_episode_len_l', train_episode_len_l)
    print('val_episode_len_l', val_episode_len_l)
    print('train_episode_len', train_episode_len)
    print('val_episode_len', val_episode_len)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    embodiment_norm_stats, _, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]),
                                   embodiment_dict=emb_dict)
    print(f'Norm stats from: {stats_dir_l}')

    timestep_weights = None
    if vel_rebalance:
        print(f'[vel_rebalance] Computing per-timestep velocity rebalance weights '
              f'(wx={vel_rebalance_weight_x}, wy={vel_rebalance_weight_y}, '
              f'wyaw={vel_rebalance_weight_yaw}, thresh={vel_rebalance_threshold})...')
        timestep_weights = compute_velocity_rebalance_weights(
            dataset_path_list, train_episode_ids, train_episode_len,
            vel_weight_x=vel_rebalance_weight_x,
            vel_weight_y=vel_rebalance_weight_y,
            vel_weight_yaw=vel_rebalance_weight_yaw,
            vel_threshold=vel_rebalance_threshold,
        )

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights, timestep_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        dataset_path_list, embodiment_norm_stats, train_episode_ids, train_episode_len, chunk_size,
        main_cam_size=main_cam_size, wrist_cam_size=wrist_cam_size,
        data_aug=data_aug, grayscale=grayscale, randomize_color=randomize_color,
    )
    val_dataset = EpisodicDataset(
        dataset_path_list, embodiment_norm_stats, val_episode_ids, val_episode_len, chunk_size,
        main_cam_size=main_cam_size, wrist_cam_size=wrist_cam_size,
        data_aug=False, grayscale=grayscale, randomize_color=False,
    )
    train_num_workers = num_workers
    val_num_workers = num_workers
    print(f'Augment images: {train_dataset.data_aug}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    
    def collate_fn(samples):
        def aggregate_dict(data: list):
            if isinstance(data[0], dict):
                return {key: aggregate_dict([sample[key] for sample in data]) for key in data[0].keys()}
            else:
                return torch.stack(data)
            
        embodiment_array = np.array([sample[1] for sample in samples], dtype=np.str_)
        input_dict = dict()
        for emb in np.unique(embodiment_array):
            data_emb_list = [sample[0] for sample in samples if sample[1] == emb]
            input_dict[emb] = aggregate_dict(data_emb_list)
        
        return input_dict
        
    worker_options = {
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
    } if num_workers else {}
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=pin_memory, num_workers=train_num_workers, collate_fn=collate_fn, **worker_options)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=pin_memory, num_workers=val_num_workers, collate_fn=collate_fn, **worker_options)

    return train_dataloader, val_dataloader, embodiment_norm_stats, embodiment_list, train_dataset.is_sim

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, norm_stats, episode_ids, episode_len,
                 chunk_size, main_cam_size=None, wrist_cam_size=None,
                 data_aug=False, grayscale=False, randomize_color=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = np.array(dataset_path_list, dtype=np.bytes_)
        self.obs_keys = dict()
        self.action_keys = dict()
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(max(episode_len), chunk_size)
        self.transformations = None
        self.main_cam_size = main_cam_size
        self.wrist_cam_size = wrist_cam_size
        self.data_aug = data_aug
        self.grayscale = grayscale
        self.randomize_color = randomize_color
        if self.data_aug:
            self.transformations = [transforms.ColorJitter(hue=0.5, saturation=0.5)]
        else:
            self.transformations = None
  
        
        self.dream_keys = dict()
        for embodiment in norm_stats.keys():
            self.obs_keys[embodiment] = norm_stats[embodiment]['obs'].keys()
            self.action_keys[embodiment] = norm_stats[embodiment]['action'].keys()
            self.dream_keys[embodiment] = norm_stats[embodiment].get('dream', {}).keys()
        
        self.__getitem__(0)  # Validate the first sample during initialization.
        self.is_sim = False

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def _get_cam_size(self, cam_name):
        """Returns (height, width) target size for a camera, or None to skip resize."""
        if 'wrist' in cam_name:
            return self.wrist_cam_size
        else:
            return self.main_cam_size

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = str(self.dataset_path_list[episode_id], encoding='utf-8')
        if "flipped" in dataset_path:
            flipped_data = True
        else:
            flipped_data = False
        with h5py.File(dataset_path, 'r') as root:
            compressed = root.attrs.get('compress', False)
            embodiment = require_htd_policy_embodiment(root, dataset_path)
            episode_len = get_episode_len(root)
            image_dict = dict()
            image_mask_dict = dict()
            proprio_dict = dict()
            proprio_mask_dict = dict()
            tactile_dict = dict()
            tactile_mask_dict = dict()
            action_dict = dict()
            action_mask_dict = dict()
            key_path_map = KEY_PATH_MAP_G1_INSPIRE_WB
            for key in self.obs_keys[embodiment]:
                if 'image' in key:
                    image_dict[key] = root[key_path_map[key]][start_ts] # will convert to tensor after preprocess
                    if key in KEY_MASK_PATH_MAP:
                        image_mask_dict[key] = torch.tensor(root[KEY_MASK_PATH_MAP[key]][0])
                    else:
                        image_mask_dict[key] = torch.tensor([True])
                elif 'proprio' in key:
                    proprio_dict[key] = torch.from_numpy(root[key_path_map[key]][start_ts]).float()
                    if key in KEY_MASK_PATH_MAP:
                        proprio_mask_dict[key] = torch.tensor(root[KEY_MASK_PATH_MAP[key]])
                    else:
                        proprio_mask_dict[key] = torch.tensor([True] * proprio_dict[key].shape[-1])
                elif 'tactile' in key:
                    tactile_dict[key] = torch.from_numpy(root[key_path_map[key]][start_ts]).float()
                    if key in KEY_MASK_PATH_MAP:
                        tactile_mask_dict[key] = torch.tensor(root[KEY_MASK_PATH_MAP[key]])
                    else:
                        tactile_mask_dict[key] = torch.tensor([True] * tactile_dict[key].shape[-1])
            for key in self.action_keys[embodiment]:
                action_dict[key] = torch.from_numpy(root[key_path_map[key]][start_ts:start_ts + self.chunk_size])
                if key in KEY_MASK_PATH_MAP:
                    action_mask_dict[key] = torch.tensor(root[KEY_MASK_PATH_MAP[key]])
                else:
                    action_mask_dict[key] = torch.tensor([True] * action_dict[key].shape[-1])

            # dream labels: future obs sequences for dream heads
            dream_dict = dict()
            dream_mask_dict = dict()
            for key in self.dream_keys.get(embodiment, []):
                if key not in key_path_map:
                    continue
                dream_dict[key] = torch.from_numpy(root[key_path_map[key]][start_ts + 1:start_ts + 1 + self.chunk_size]).float()
                dream_mask_dict[key] = torch.tensor([True] * dream_dict[key].shape[-1])

            for cam_name, image in image_dict.items():
                if compressed:
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                target_size = self._get_cam_size(cam_name)
                if target_size is not None:
                    th, tw = target_size
                    if image.shape[1] != tw or image.shape[0] != th:
                        image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
                image_dict[cam_name] = image

            action_len = episode_len - start_ts
            dream_len = action_len - 1

        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1
        is_pad = torch.from_numpy(is_pad).bool()
        for key in action_dict.keys():
            orig_action = action_dict[key]
            action_dim = orig_action.shape[-1]
            padded_action = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
            padded_action[:len(orig_action)] = orig_action
            action_dict[key] = torch.from_numpy(padded_action).float()

        is_pad_dream = np.zeros(self.chunk_size)
        is_pad_dream[dream_len:] = 1
        is_pad_dream = torch.from_numpy(is_pad_dream).bool()
        for key in dream_dict.keys():
            orig_dream = dream_dict[key]
            dream_dim = orig_dream.shape[-1]
            padded_dream = np.zeros((self.chunk_size, dream_dim), dtype=np.float32)
            padded_dream[:len(orig_dream)] = orig_dream
            dream_dict[key] = torch.from_numpy(padded_dream).float()

        order = np.random.permutation(3) if self.randomize_color else None
        for key in image_dict.keys():
            image_tensor = torch.from_numpy(image_dict[key])
            # Keep transport buffers compact; only CPU transforms need floats.
            if self.grayscale or self.data_aug:
                image_tensor = image_tensor.float().div_(255.0)
            image_tensor = torch.einsum('h w c -> c h w', image_tensor)
            if flipped_data:
                image_tensor = image_tensor.flip(-1)
            if self.randomize_color:
                image_tensor = image_tensor[order,:,:]
            if self.grayscale:
                image_tensor = torch.mean(image_tensor, dim=0, keepdim=True).repeat(3,1,1)
            if self.data_aug:
                for transform in self.transformations:
                    image_tensor = transform(image_tensor)
            image_dict[key] = image_tensor.contiguous()
            del image_tensor

        
        ret_dict = {
            'obs': {**image_dict, **proprio_dict, **tactile_dict},
            'obs_mask': {**image_mask_dict, **proprio_mask_dict, **tactile_mask_dict},
            'action': action_dict,
            'action_mask': action_mask_dict,
            'is_pad': is_pad,
        }
        if dream_dict:
            ret_dict['dream'] = dream_dict
            ret_dict['dream_mask'] = dream_mask_dict
            ret_dict['is_pad_dream'] = is_pad_dream

        del image_dict, image_mask_dict, proprio_dict, proprio_mask_dict, tactile_dict, tactile_mask_dict, action_dict, action_mask_dict, dream_dict, dream_mask_dict, is_pad
        
        return ret_dict, embodiment


def parse_embodiment_keys(emb_dict:dict, embodiment:str):
    obs_keys = []
    action_keys = []
    dream_keys = []
    for key in emb_dict[embodiment]['tokenizer'].keys():
        if 'proprio' in key or 'image' in key or 'tactile' in key:
            obs_keys.append(key)
    for key in emb_dict[embodiment]['action_head'].keys():
        action_keys.append(key)
    if 'dream_head' in emb_dict[embodiment]:
        for key in emb_dict[embodiment]['dream_head'].keys():
            dream_keys.append(key)
    return obs_keys, action_keys, dream_keys


def get_norm_stats(dataset_path_list, embodiment_dict):
    all_obs_data = dict()
    all_action_data = dict()
    all_dream_data = dict()
    all_episode_len = []
    all_embodiment_stats = dict()
    embodiment_list = []
    total_emb_episode_len = dict()

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                embodiment = require_htd_policy_embodiment(root, dataset_path)
                key_path_map = KEY_PATH_MAP_G1_INSPIRE_WB
                if embodiment not in embodiment_list:
                    embodiment_list.append(embodiment)
                    print('---------discovered embodiment', embodiment, '----------')
                    obs_entry_keys, action_entry_keys, dream_entry_keys = parse_embodiment_keys(embodiment_dict, embodiment)
                    all_obs_data[embodiment] = {key: [] for key in obs_entry_keys}
                    all_action_data[embodiment] = {key: [] for key in action_entry_keys}
                    all_dream_data[embodiment] = {key: [] for key in dream_entry_keys}
                    all_embodiment_stats[embodiment] = dict()
                    total_emb_episode_len[embodiment] = 0
                for key in obs_entry_keys:
                    if 'image' not in key:
                        all_obs_data[embodiment][key].append(root[key_path_map[key]][()])
                for key in all_action_data[embodiment].keys():
                    all_action_data[embodiment][key].append(root[key_path_map[key]][()])
                for key in all_dream_data[embodiment].keys():
                    if key in key_path_map and key_path_map[key] in root:
                        all_dream_data[embodiment][key].append(root[key_path_map[key]][()])
                all_episode_len.append(len(root[key_path_map[obs_entry_keys[0]]][()]))
                total_emb_episode_len[embodiment] += all_episode_len[-1]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            raise e

    for embodiment in embodiment_list:
        obs_stats = dict()
        action_stats = dict()
        dream_stats = dict()
        for key in all_obs_data[embodiment].keys():
            if 'image' not in key:
                all_obs_data[embodiment][key] = np.concatenate(all_obs_data[embodiment][key], axis=0)
            else:
                obs_stats[key] = {"mean": None, "std": None}
        for key in all_action_data[embodiment].keys():
            all_action_data[embodiment][key] = np.concatenate(all_action_data[embodiment][key], axis=0)
        for key in all_dream_data[embodiment].keys():
            if all_dream_data[embodiment][key]:
                all_dream_data[embodiment][key] = np.concatenate(all_dream_data[embodiment][key], axis=0)

        obs_stats.update({key: {"mean": np.mean(all_obs_data[embodiment][key], axis=0), "std": np.std(all_obs_data[embodiment][key], axis=0)} for key in all_obs_data[embodiment].keys() if 'image' not in key})
        action_stats.update({key: {"mean": np.mean(all_action_data[embodiment][key], axis=0), "std": np.std(all_action_data[embodiment][key], axis=0)} for key in all_action_data[embodiment].keys()})
        for key in all_dream_data[embodiment].keys():
            data = all_dream_data[embodiment][key]
            if isinstance(data, np.ndarray) and data.size > 0:
                dream_stats[key] = {"mean": np.mean(data, axis=0), "std": np.std(data, axis=0)}
        all_embodiment_stats[embodiment] = {'obs': obs_stats, 'action': action_stats, 'dream': dream_stats}

    del all_obs_data, all_action_data, all_dream_data
    
    for embodiment in total_emb_episode_len.keys():
        print(f'-------------Embodiment {embodiment} has {total_emb_episode_len[embodiment]} steps--------------------')
    
    return all_embodiment_stats, all_episode_len, embodiment_list

KEY_PATH_MAP_G1_INSPIRE_WB = {
    'main_image': '/observations/images/main',
    'left_wrist_image': '/observations/images/wrist_left',
    'right_wrist_image': '/observations/images/wrist_right',
    'body_pose_proprio': '/observations/proprioceptions/body_proprio',
    'eef_pose_proprio': '/observations/proprioceptions/eef_proprio',
    'eef_to_body_pose_proprio': '/observations/proprioceptions/eef_to_body_proprio',
    'left_eef_to_body_pose_proprio': '/observations/proprioceptions/left_eef_to_body_proprio',
    'right_eef_to_body_pose_proprio': '/observations/proprioceptions/right_eef_to_body_proprio',
    'eef_link_proprio': '/observations/proprioceptions/other/eef_link_proprio',
    'left_eef_link_proprio': '/observations/proprioceptions/left_eef_link_proprio',
    'right_eef_link_proprio': '/observations/proprioceptions/right_eef_link_proprio',
    'robot_joint_pos_proprio': '/observations/proprioceptions/robot_joint_pos_proprio',
    'robot_joint_vel_proprio': '/observations/proprioceptions/robot_joint_vel_proprio',
    'left_eef_joint_pos_proprio': '/observations/proprioceptions/left_eef_joint_pos_proprio',
    'right_eef_joint_pos_proprio': '/observations/proprioceptions/right_eef_joint_pos_proprio',
    'body_pose': '/actions/body',
    'eef_to_body_pose': '/actions/eef_to_body',
    'left_eef_to_body_pose': '/actions/left_eef_to_body',
    'right_eef_to_body_pose': '/actions/right_eef_to_body',
    'eef_link_pose': '/actions/eef_link',
    'left_eef_link_pose': '/actions/left_eef_link',
    'right_eef_link_pose': '/actions/right_eef_link',
    'velocity_commands': '/actions/velocity_commands',
    'left_eef_finger_force_proprio': '/observations/proprioceptions/left_eef_finger_force_proprio',
    'right_eef_finger_force_proprio': '/observations/proprioceptions/right_eef_finger_force_proprio',
    # dream head force keys (map to the same observation paths as proprio)
    'left_eef_finger_force': '/observations/proprioceptions/left_eef_finger_force_proprio',
    'right_eef_finger_force': '/observations/proprioceptions/right_eef_finger_force_proprio',
    'left_tactile': '/observations/proprioceptions/other/left_touch',
    'right_tactile': '/observations/proprioceptions/other/right_touch',
    'left_pinky_tactile': '/observations/tactile/left_pinky_tactile',
    'right_pinky_tactile': '/observations/tactile/right_pinky_tactile',
    'left_ring_tactile': '/observations/tactile/left_ring_tactile',
    'right_ring_tactile': '/observations/tactile/right_ring_tactile',
    'left_middle_tactile': '/observations/tactile/left_middle_tactile',
    'right_middle_tactile': '/observations/tactile/right_middle_tactile',
    'left_index_tactile': '/observations/tactile/left_index_tactile',
    'right_index_tactile': '/observations/tactile/right_index_tactile',
    'left_thumb_tactile': '/observations/tactile/left_thumb_tactile',
    'right_thumb_tactile': '/observations/tactile/right_thumb_tactile',
    'left_palm_tactile': '/observations/tactile/left_palm_tactile',
    'right_palm_tactile': '/observations/tactile/right_palm_tactile',
    'left_tactile_cap': '/observations/proprioceptions/other/left_touch',
    'right_tactile_cap': '/observations/proprioceptions/other/right_touch',
    'left_pinky_tactile_cap': '/observations/tactile/left_pinky_tactile',
    'right_pinky_tactile_cap': '/observations/tactile/right_pinky_tactile',
    'left_ring_tactile_cap': '/observations/tactile/left_ring_tactile',
    'right_ring_tactile_cap': '/observations/tactile/right_ring_tactile',
    'left_middle_tactile_cap': '/observations/tactile/left_middle_tactile',
    'right_middle_tactile_cap': '/observations/tactile/right_middle_tactile',
    'left_index_tactile_cap': '/observations/tactile/left_index_tactile',
    'right_index_tactile_cap': '/observations/tactile/right_index_tactile',
    'left_thumb_tactile_cap': '/observations/tactile/left_thumb_tactile',
    'right_thumb_tactile_cap': '/observations/tactile/right_thumb_tactile',
    'left_palm_tactile_cap': '/observations/tactile/left_palm_tactile',
    'right_palm_tactile_cap': '/observations/tactile/right_palm_tactile',
}

# missing keys default to true
KEY_MASK_PATH_MAP = {
    'main_image': '/masks/img_main',
    'wrist_image': '/masks/img_wrist',
    'body_pose_proprio': '/masks/proprio_body',
    'eef_pose_proprio': '/masks/proprio_eef',
    'eef_to_body_pose_proprio': '/masks/proprio_eef',
    'gripper_proprio': '/masks/proprio_gripper',
    'delta_eef_pose': '/masks/act_eef',
    'delta_body_pose': '/masks/act_body',
    'delta_gripper': '/masks/act_gripper',
    'eef_pose': '/masks/act_eef',
    'body_pose': '/masks/act_body',
    'gripper': '/masks/act_gripper',
}

def get_episode_len(hdfile):
    for key in hdfile.keys():
        if isinstance(hdfile[key], h5py.Group):
            res = get_episode_len(hdfile[key])
            return res
        else:
            return hdfile[key][()].shape[0]
