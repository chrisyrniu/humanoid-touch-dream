import argparse
import h5py
import numpy as np
import fnmatch
import os

if __package__:
    from .processing_utils import atomic_process_hdf5
else:
    from processing_utils import atomic_process_hdf5

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir, followlinks=False):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return sorted(hdf5_files)

def replace_dataset(f, dataset_path, new_data):
    """
    Replace a dataset with new data, handling shape changes.
    If the shape changed, delete and recreate the dataset.
    """
    if dataset_path in f:
        old_shape = f[dataset_path].shape
        new_shape = new_data.shape
        if old_shape == new_shape:
            # Shape unchanged, can use in-place assignment
            f[dataset_path][...] = new_data
        else:
            # Shape changed, need to delete and recreate
            del f[dataset_path]
            f.create_dataset(dataset_path, data=new_data)
    else:
        # Dataset doesn't exist, create it
        f.create_dataset(dataset_path, data=new_data)

def process_g1(f):
    """Split the retained combined G1 fields without changing their values."""
    expected_widths = {
        'actions/eef_to_body': 12,
        'actions/eef_joints_target': 150,
        'observations/proprioceptions/eef_to_body_proprio': 12,
        'observations/proprioceptions/other/eef_link_proprio': 30,
        'observations/proprioceptions/other/eef_joint_proprio': 24,
        'observations/proprioceptions/other/robot_joint_proprio': 58,
    }
    frames = None
    for path, width in expected_widths.items():
        dataset = f[path]
        if len(dataset.shape) != 2 or dataset.shape[1] != width or dataset.shape[0] == 0:
            raise ValueError(f'{path}: expected a nonempty [N, {width}] array, got {dataset.shape}')
        if frames is not None and dataset.shape[0] != frames:
            raise ValueError(f'{path}: inconsistent episode frame count')
        frames = dataset.shape[0]
    # right and left eef to body pose
    eef_to_body_pose = np.array(f['actions/eef_to_body'])
    eef_to_body_pose = eef_to_body_pose.reshape(-1, 2, 6)
    right_eef_to_body_pose = eef_to_body_pose[:, 0, :]
    left_eef_to_body_pose = eef_to_body_pose[:, 1, :]
    # (T, 6)
    replace_dataset(f, 'actions/right_eef_to_body', right_eef_to_body_pose)
    replace_dataset(f, 'actions/left_eef_to_body', left_eef_to_body_pose)
    print(f'  ✓ Successfully processed eef_to_body_pose')
    
    # right and left eef link pose
    eef_joints_target = np.array(f['actions/eef_joints_target'])
    eef_joints_target = eef_joints_target.reshape(-1, 2, 25, 3)
    right_eef_link_pose = np.concatenate([
        eef_joints_target[:, 0, 4, :], # thumb
        eef_joints_target[:, 0, 9, :], # index
        eef_joints_target[:, 0, 14, :], # middle
        eef_joints_target[:, 0, 19, :], # ring
        eef_joints_target[:, 0, 24, :], # pinky
    ], axis=1)
    left_eef_link_pose = np.concatenate([
        eef_joints_target[:, 1, 4, :], # thumb
        eef_joints_target[:, 1, 9, :], # index
        eef_joints_target[:, 1, 14, :], # middle
        eef_joints_target[:, 1, 19, :], # ring
        eef_joints_target[:, 1, 24, :] # pinky
    ], axis=1)
    # (T, 15)
    replace_dataset(f, 'actions/right_eef_link', right_eef_link_pose)
    replace_dataset(f, 'actions/left_eef_link', left_eef_link_pose)
    eef_link_pose = np.concatenate([right_eef_link_pose, left_eef_link_pose], axis=1) # (T, 30)
    replace_dataset(f, 'actions/eef_link', eef_link_pose)
    print(f'  ✓ Successfully processed eef_link_pose')

    # right and left eef to body proprio
    eef_to_body_pose_proprio = np.array(f['observations/proprioceptions/eef_to_body_proprio'])
    eef_to_body_pose_proprio = eef_to_body_pose_proprio.reshape(-1, 2, 6)
    right_eef_to_body_pose_proprio = eef_to_body_pose_proprio[:, 0, :]
    left_eef_to_body_pose_proprio = eef_to_body_pose_proprio[:, 1, :]
    # (T, 6)
    replace_dataset(f, 'observations/proprioceptions/right_eef_to_body_proprio', right_eef_to_body_pose_proprio)
    replace_dataset(f, 'observations/proprioceptions/left_eef_to_body_proprio', left_eef_to_body_pose_proprio)
    print(f'  ✓ Successfully processed eef_to_body_pose_proprio')

    # right and left eef link proprio
    eef_link_proprio = np.array(f['observations/proprioceptions/other/eef_link_proprio'])
    eef_link_proprio = eef_link_proprio.reshape(-1, 2, 15)
    right_eef_link_proprio = eef_link_proprio[:, 0, :]
    left_eef_link_proprio = eef_link_proprio[:, 1, :]
    # (T, 15)
    replace_dataset(f, 'observations/proprioceptions/right_eef_link_proprio', right_eef_link_proprio)
    replace_dataset(f, 'observations/proprioceptions/left_eef_link_proprio', left_eef_link_proprio)
    print(f'  ✓ Successfully processed eef_link_proprio')

    # right and left eef joint state proprio
    eef_joint_state_proprio = np.array(f['observations/proprioceptions/other/eef_joint_proprio'])
    eef_joint_state_proprio = eef_joint_state_proprio.reshape(-1, 4, 6)
    right_eef_joint_pos_proprio = eef_joint_state_proprio[:, 0, :]
    left_eef_joint_pos_proprio = eef_joint_state_proprio[:, 1, :]
    right_eef_finger_force_proprio = eef_joint_state_proprio[:, 2, :]
    left_eef_finger_force_proprio = eef_joint_state_proprio[:, 3, :]
    # (T, 6)
    replace_dataset(f, 'observations/proprioceptions/right_eef_joint_pos_proprio', right_eef_joint_pos_proprio)
    replace_dataset(f, 'observations/proprioceptions/left_eef_joint_pos_proprio', left_eef_joint_pos_proprio)
    replace_dataset(f, 'observations/proprioceptions/right_eef_finger_force_proprio', right_eef_finger_force_proprio)
    replace_dataset(f, 'observations/proprioceptions/left_eef_finger_force_proprio', left_eef_finger_force_proprio)
    print(f'  ✓ Successfully processed eef_joint_state_proprio')

    # robot joint state proprio
    robot_joint_state_proprio = np.array(f['observations/proprioceptions/other/robot_joint_proprio'])
    robot_joint_state_proprio = robot_joint_state_proprio.reshape(-1, 2, 29)
    robot_joint_pos_proprio = robot_joint_state_proprio[:, 0, :]
    robot_joint_vel_proprio = robot_joint_state_proprio[:, 1, :]
    # (T, 29)
    replace_dataset(f, 'observations/proprioceptions/robot_joint_pos_proprio', robot_joint_pos_proprio)
    replace_dataset(f, 'observations/proprioceptions/robot_joint_vel_proprio', robot_joint_vel_proprio)
    print(f'  ✓ Successfully processed robot_joint_state_proprio')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Process G1 Inspire whole-body HDF5 data')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Offline working copy containing HDF5 episodes; each successful file is atomically replaced')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.dataset_dir):
        parser.error('--dataset_dir must be an existing directory')
    hdf5_files = find_all_hdf5(args.dataset_dir, skip_mirrored_data=True)
    if not hdf5_files:
        print('No eligible HDF5 episodes found')
        return 1
    errors = 0
    for index, path in enumerate(hdf5_files, 1):
        print(f'[{index}/{len(hdf5_files)}] Processing: {path}')
        try:
            atomic_process_hdf5(path, process_g1)
        except Exception as exc:
            errors += 1
            print(f'Error processing {path}: {type(exc).__name__}: {exc}')
    print(f'Processing summary: files={len(hdf5_files)} successful={len(hdf5_files) - errors} errors={errors}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
