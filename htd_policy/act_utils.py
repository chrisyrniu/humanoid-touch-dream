"""
Code adapted from https://github.com/MarkFzp/humanplus
"""


import numpy as np
import torch
import os
import h5py
import fnmatch
import cv2
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, 
                 chunk_size, policy_class,width=None, height=None, normalize_resnet=False,data_aug=False,
                 observation_name=[],feature_loss=False,grayscale=False,randomize_color=False,
                 randomize_index=None,randomize_data_degree=0,randomize_data=False, aggregate_image=True,
                 action_name=['actions'], is_sim=False,
                 main_cam_size=None, wrist_cam_size=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(max(episode_len), chunk_size)
        self.policy_class = policy_class
        self.transformations = None
        self.width = width
        self.height = height
        self.main_cam_size = main_cam_size
        self.wrist_cam_size = wrist_cam_size
        self.data_aug = data_aug
        self.feature_loss = feature_loss
        self.randomize_data = randomize_data
        self.randomize_data_radian = randomize_data_degree/180*np.pi
        self.randomize_index = randomize_index
        self.grayscale = grayscale
        self.randomize_color = randomize_color
        self.aggregate_image = aggregate_image
        # self.precompute_image_features = precompute_image_features
        if self.data_aug:
            #has nothing to do with the deployment of the model 
            self.transformations = [
                transforms.ColorJitter(hue=0.5,saturation=0.5),
                # transforms.Pad(padding=[int(self.width * 0.05), int(self.height * 0.05)], padding_mode='edge'),
                # transforms.RandomCrop(size=[self.height,self.width])]
            ]
        else:
            self.transformations = None
  
        self.normalize_resnet = normalize_resnet
        if self.normalize_resnet:
            #need to normalize the image to the same mean and std as the resnet model during depolyment
            self.normalize_resnet_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        self.observation_name = observation_name
        self.action_name = action_name
        self.is_sim = bool(is_sim)


    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        action_mask_data = None
        if "flipped" in dataset_path:
            flipped_data = True
        else:
            flipped_data = False
        with h5py.File(dataset_path, 'r') as root:
            is_sim = bool(root.attrs.get('sim', self.is_sim))
            compressed = root.attrs.get('compress', False)
            episode_len = root[f'/{self.action_name[0]}'].shape[0]
            action_start = start_ts if is_sim else max(0, start_ts - 1)
            action_data = []
            action_mask = []
            for name in self.action_name:
                act_array = root[f'/{name}'][action_start:action_start + self.chunk_size]
                action_data.append(act_array)
                item_name = name.split('/')[-1]
                if f'/masks/act_{item_name}' in root:
                    action_mask.append(root[f'/masks/act_{item_name}'][()])
                else:
                    action_mask.append(np.ones(act_array.shape[-1], dtype=bool))
            action = np.concatenate(action_data, axis=-1)
            action_mask = np.concatenate(action_mask, axis=-1)
            action_mask = action_mask if action.shape[-1] == action_mask.shape[-1] else None
            # get observation at start_ts only
            if len(self.observation_name) > 0:
                observation_data = []
                for name in self.observation_name:
                    if name=='imu_orn':
                        observation_data.append(root[f'/observations/{name}'][()][start_ts, :2]) # only take the first 2 elements, row and pitch
                    else:
                        observation_data.append(root[f'/observations/{name}'][start_ts])
                qpos = np.concatenate(observation_data, axis=-1)
            else:
                qpos = np.zeros(40)
            
            image_dict = dict()
            if self.feature_loss:
                image_dict_future = dict()
            for cam_name in self.camera_names:
                hdf5_name = 'main' if cam_name in ('main_left', 'main_right') else cam_name
                if hdf5_name not in image_dict:
                    image_dict[hdf5_name] = root[f'/observations/images/{hdf5_name}'][start_ts]
                    if self.feature_loss:
                        dummy_index = min(start_ts+self.chunk_size, episode_len - 1)
                        image_dict_future[hdf5_name] = root[f'/observations/images/{hdf5_name}'][dummy_index]
                    
            
            image_groups = [image_dict]
            if self.feature_loss:
                image_groups.append(image_dict_future)
            has_split_main = 'main' in image_dict
            for images in image_groups:
                for cam_name, image in images.items():
                    if compressed:
                        image = cv2.imdecode(image, 1)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    main_size = self.main_cam_size
                    if main_size is None and self.width is not None and self.height is not None:
                        main_size = (self.height, self.width)
                    target_size = main_size
                    if 'wrist' in cam_name and self.wrist_cam_size is not None:
                        target_size = self.wrist_cam_size
                    elif cam_name != 'main' and has_split_main and main_size is not None:
                        target_size = (main_size[0], main_size[1] // 2)
                    if target_size is not None:
                        target_h, target_w = target_size
                        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    images[cam_name] = np.array(image)

        action_len = len(action)
        padded_action = np.zeros((self.chunk_size, action.shape[-1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1
        is_pad[:action_len][np.all(action == 0., axis=1)] = 1  # filter out zero actions

        # new axis for different cameras
        if self.aggregate_image:
            all_cam_images = []
            if self.feature_loss:
                all_cam_images_future = []
            for cam_name in self.camera_names:
                if cam_name == 'main_left':
                    full_img = image_dict['main']
                    w_half = full_img.shape[1] // 2
                    all_cam_images.append(full_img[:, :w_half, :])
                elif cam_name == 'main_right':
                    full_img = image_dict['main']
                    w_half = full_img.shape[1] // 2
                    all_cam_images.append(full_img[:, w_half:, :])
                else:
                    all_cam_images.append(image_dict[cam_name])
                if self.feature_loss:
                    if cam_name == 'main_left':
                        full_img_f = image_dict_future['main']
                        w_half = full_img_f.shape[1] // 2
                        all_cam_images_future.append(full_img_f[:, :w_half, :])
                    elif cam_name == 'main_right':
                        full_img_f = image_dict_future['main']
                        w_half = full_img_f.shape[1] // 2
                        all_cam_images_future.append(full_img_f[:, w_half:, :])
                    else:
                        all_cam_images_future.append(image_dict_future[cam_name])
            
            if len({image.shape for image in all_cam_images}) == 1:
                all_cam_images = np.stack(all_cam_images, axis=0)
                if self.feature_loss:
                    all_cam_images_future = np.stack(all_cam_images_future, axis=0)

                # construct observations
                image_data = torch.from_numpy(all_cam_images)
                if self.feature_loss:
                    image_data_future = torch.from_numpy(all_cam_images_future)
                    image_data = torch.cat([image_data, image_data_future], dim=0) #just cat the images together for feature loss
        
                
                qpos_data = torch.from_numpy(qpos).float()
                action_data = torch.from_numpy(padded_action).float()
                is_pad = torch.from_numpy(is_pad).bool()
                action_mask_data = torch.from_numpy(action_mask).bool()

                # channel last
                image_data = torch.einsum('k h w c -> k c h w', image_data)
                # Keep transport buffers compact; only CPU transforms need floats.
                if self.grayscale or self.data_aug:
                    image_data = image_data / 255.0
                if flipped_data:
                    image_data = image_data.flip(-1)
                if self.randomize_color:
                    order = np.random.permutation(3)
                    image_data = image_data[:,order,:,:] # randomize color
            

                if self.grayscale:
                    image_data = torch.mean(image_data, dim=1, keepdim=True).repeat(1,3,1,1)
                if self.data_aug:
                    for transform in self.transformations:
                        image_data = transform(image_data) # apply data augmentation
                image_data = image_data.contiguous()
            else:
                # Different camera resolutions cannot share one stacked image tensor.
                image_data = {}
                order = np.random.permutation(3) if self.randomize_color else None
                for cam_id, camera_name in enumerate(self.camera_names):
                    image_tensor = torch.from_numpy(all_cam_images[cam_id]).permute(2, 0, 1)
                    if self.feature_loss:
                        future_tensor = torch.from_numpy(all_cam_images_future[cam_id]).permute(2, 0, 1)
                        image_tensor = torch.stack((image_tensor, future_tensor), dim=0)
                    if self.grayscale or self.data_aug:
                        image_tensor = image_tensor / 255.0
                    if flipped_data:
                        image_tensor = image_tensor.flip(-1)
                    if order is not None:
                        image_tensor = image_tensor[..., order, :, :]
                    if self.grayscale:
                        image_tensor = image_tensor.mean(dim=-3, keepdim=True).expand_as(image_tensor)
                    if self.data_aug:
                        for transform in self.transformations:
                            image_tensor = transform(image_tensor)
                    image_data[camera_name] = image_tensor.contiguous()
                qpos_data = torch.from_numpy(qpos).float()
                action_data = torch.from_numpy(padded_action).float()
                is_pad = torch.from_numpy(is_pad).bool()
                action_mask_data = torch.from_numpy(action_mask).bool()

            if self.randomize_data:
                #randomize the action data
                randomize_amount = torch.rand(len(self.randomize_index))*2*self.randomize_data_radian - self.randomize_data_radian
                action_data[:,self.randomize_index] += randomize_amount
                qpos_data[self.randomize_index] += randomize_amount
        else:
            image_data = dict()
            order = np.random.permutation(3) if self.randomize_color else None
            for camera_name in self.camera_names:
                image_tensor = torch.from_numpy(image_dict[camera_name])
                if self.feature_loss:
                    future_image_tensor = torch.from_numpy(image_dict_future[camera_name])
                    image_tensor = torch.cat([image_tensor, future_image_tensor], dim=0)
                if self.grayscale or self.data_aug:
                    image_tensor = image_tensor / 255.0
                image_tensor = torch.einsum('h w c -> c h w', image_tensor)
                if flipped_data:
                    image_tensor = image_tensor.flip(-1)
                # if not self.randomize_color:
                #     image_tensor = image_tensor[[2,1,0],:,:] # check whether data is rgb or bgr
                else:
                    image_tensor = image_tensor[order,:,:]
                if self.grayscale:
                    image_tensor = torch.mean(image_tensor, dim=0, keepdim=True).repeat(3,1,1)
                if self.data_aug:
                    for transform in self.transformations:
                        image_tensor = transform(image_tensor)
                image_data[camera_name] = image_tensor.contiguous()
                
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()
            action_mask_data = torch.from_numpy(action_mask).bool()
            
            if self.randomize_data:
                #randomize the action data
                randomize_amount = torch.rand(len(self.randomize_index))*2*self.randomize_data_radian - self.randomize_data_radian
                action_data[:,self.randomize_index] += randomize_amount
                qpos_data[self.randomize_index] += randomize_amount
            
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        return image_data, qpos_data, action_data, is_pad, action_mask_data


def get_norm_stats(dataset_path_list,observation_name=['qpos'], action_name=['actions']):
    all_qpos_data = []
    all_action_data = []
    all_action_v_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                action_data = []
                for name in action_name:
                    action_data.append(root[f'/{name}'][()])
                action = np.concatenate(action_data, axis=-1)
                original_action_shape = action.shape
                try:
                    observation_data = []
                    for name in observation_name:
                        if name=='imu_orn':
                            observation_data.append(root[f'/observations/{name}'][()][..., :2]) # only take the first 2 elements, row and pitch
                        else:
                            observation_data.append(root[f'/observations/{name}'][()])
                    qpos = np.concatenate(observation_data, axis=-1)
                except:
                    qpos = np.zeros([original_action_shape[0],40])

      
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_action_v_data.append(torch.from_numpy(action[1:, :]-action[:-1, :]))
        all_episode_len.append(len(action))
    
    all_action_data = torch.cat(all_action_data, dim=0) 
    all_action_v_data = torch.cat(all_action_v_data, dim=0)
    
    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping    

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    
    all_action_v_max = all_action_v_data.max(dim=0).values.float()
    all_action_v_min = all_action_v_data.min(dim=0).values.float()
    

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
        
    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
            "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
            "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
            "example_qpos": qpos,
            "all_action_v_max": all_action_v_max.numpy(),
            "all_action_v_min": all_action_v_min.numpy()}
  
    return stats, all_episode_len, all_qpos_data.size(-1), all_action_data.size(-1)

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir, followlinks=True):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights, timestep_weights=None):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])

    group_cdfs = None
    if timestep_weights is not None:
        group_cdfs = []
        for i in range(len(episode_len_l)):
            lo, hi = sum_dataset_len_l[i], sum_dataset_len_l[i + 1]
            w = timestep_weights[lo:hi].astype(np.float64)
            w /= w.sum()
            group_cdfs.append((lo, np.cumsum(w)))

    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            if group_cdfs is not None:
                lo, cdf = group_cdfs[episode_idx]
                step_idx = lo + np.searchsorted(cdf, np.random.random())
            else:
                step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, 
              chunk_size, skip_mirrored_data=False, load_pretrain=False, 
              policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.95,
              width=None, height=None, normalize_resnet=False, data_aug=False,observation_name=None,
              feature_loss=False, grayscale=False, randomize_color=False,
              randomize_index=None,randomize_data_degree=0,randomize_data=False,aggregate_image=True,
              action_name=['actions'], precomputed_features=False, min_val_num=1, is_sim=False,
              num_workers=8, main_cam_size=None, wrist_cam_size=None, pin_memory=True):
    if num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    dataset_path_list = sorted(dataset_path_list)
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    # shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    shuffled_episode_ids_0 = np.arange(num_episodes_0) # we don't shuffle so that the validation episodes are consistent across runs, you can change this
    val_episode_num_0 = max(min_val_num, int(num_episodes_0 * (1 - train_ratio)))
    train_episode_ids_0 = shuffled_episode_ids_0[:-val_episode_num_0]
    val_episode_ids_0 = shuffled_episode_ids_0[-val_episode_num_0:]
    
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    print("test on ",dataset_path_list[val_episode_ids_l[0][0]])
    print(f'validation episodes: {[dataset_path_list[idx] for idx in val_episode_ids_0]}')
    
    # obtain normalization stats for qpos and action
    _, all_episode_len, qpos_dim, action_dim = get_norm_stats(dataset_path_list,observation_name=observation_name, action_name=action_name)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _, _, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]),observation_name=observation_name, action_name=action_name)
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)


    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, 
                                    policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=data_aug,
                                    observation_name=observation_name,feature_loss=feature_loss,grayscale=grayscale,randomize_color=randomize_color,
                                    randomize_index=randomize_index,randomize_data_degree=randomize_data_degree,randomize_data=randomize_data,
                                    aggregate_image=aggregate_image, action_name=action_name, is_sim=is_sim,
                                    main_cam_size=main_cam_size, wrist_cam_size=wrist_cam_size)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, 
                                  policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=False,
                                  observation_name=observation_name,feature_loss=feature_loss,grayscale=grayscale,randomize_color=False,
                                  aggregate_image=aggregate_image, action_name=action_name, is_sim=is_sim,
                                  main_cam_size=main_cam_size, wrist_cam_size=wrist_cam_size)
    train_num_workers = num_workers
    val_num_workers = num_workers
    print(f'Augment images: {train_dataset.data_aug}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    worker_options = {'prefetch_factor': 2} if num_workers else {}
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=pin_memory, num_workers=train_num_workers, **worker_options)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=pin_memory, num_workers=val_num_workers, **worker_options)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim, qpos_dim, action_dim


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
