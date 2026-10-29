import argparse
import h5py
import numpy as np
import fnmatch
import os

if __package__:
    from .processing_utils import atomic_process_hdf5
else:
    from processing_utils import atomic_process_hdf5

# ---------------------------------------------------------------------------
# F1 hand tactile constants
# ---------------------------------------------------------------------------
# Raw processed layout per hand: 29 dims
#   5 fingers (little, ring, middle, index, thumb) x 4: normal, tangential, direction, proximity
#   3 palms (palm_left, palm_center, palm_right) x 3: normal, tangential, direction
F1_FINGER_NAMES = ['little', 'ring', 'middle', 'index', 'thumb']
F1_PALM_NAMES = ['palm_left', 'palm_center', 'palm_right']
F1_RAW_DIM_PER_HAND = 29
F1_FINGER_RAW_DIM = 4   # normal, tangential, direction, proximity
F1_PALM_RAW_DIM = 3     # normal, tangential, direction
F1_FINGER_NORM_DIM = 5  # normal/20, tangential/20, sin(dir), cos(dir), proximity
F1_PALM_NORM_DIM = 4    # normal/20, tangential/20, sin(dir), cos(dir)
F1_NORM_DIM_PER_HAND = len(F1_FINGER_NAMES) * F1_FINGER_NORM_DIM + len(F1_PALM_NAMES) * F1_PALM_NORM_DIM  # 37
F1_FORCE_SCALE = 20.0
F1_NO_CONTACT_DIR = -1.0

# finger name -> dftp-style tactile key name
F1_FINGER_TO_TACTILE_KEY = {
    'little': 'pinky',
    'ring': 'ring',
    'middle': 'middle',
    'index': 'index',
    'thumb': 'thumb',
}

# ---------------------------------------------------------------------------
# DFTP hand tactile constants
# ---------------------------------------------------------------------------
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

def normalize_f1_touch(raw):
    """Normalize F1 tactile data from raw (T, 29) to processed (T, 37).

    Per-finger (x5): [normal/20, tangential/20, sin(dir), cos(dir), proximity]
    Per-palm   (x3): [normal/20, tangential/20, sin(dir), cos(dir)]
    Direction sentinel -1 (no contact) maps to (sin=0, cos=0).
    """
    T = raw.shape[0]
    out = np.zeros((T, F1_NORM_DIM_PER_HAND), dtype=np.float32)
    ri = 0  # read index into raw
    wi = 0  # write index into out
    for _ in range(len(F1_FINGER_NAMES)):
        normal = raw[:, ri]
        tangential = raw[:, ri + 1]
        direction = raw[:, ri + 2]
        proximity = raw[:, ri + 3]
        contact = direction != F1_NO_CONTACT_DIR
        out[:, wi]     = normal / F1_FORCE_SCALE
        out[:, wi + 1] = tangential / F1_FORCE_SCALE
        out[:, wi + 2] = np.where(contact, np.sin(direction), 0.0)
        out[:, wi + 3] = np.where(contact, np.cos(direction), 0.0)
        out[:, wi + 4] = proximity
        ri += F1_FINGER_RAW_DIM
        wi += F1_FINGER_NORM_DIM
    for _ in range(len(F1_PALM_NAMES)):
        normal = raw[:, ri]
        tangential = raw[:, ri + 1]
        direction = raw[:, ri + 2]
        contact = direction != F1_NO_CONTACT_DIR
        out[:, wi]     = normal / F1_FORCE_SCALE
        out[:, wi + 1] = tangential / F1_FORCE_SCALE
        out[:, wi + 2] = np.where(contact, np.sin(direction), 0.0)
        out[:, wi + 3] = np.where(contact, np.cos(direction), 0.0)
        ri += F1_PALM_RAW_DIM
        wi += F1_PALM_NORM_DIM
    return out


def process_f1_tactile(f):
    """Process F1 hand tactile data: normalize and split into per-finger datasets."""
    left_raw = np.array(f['observations/proprioceptions/other/left_touch'])
    right_raw = np.array(f['observations/proprioceptions/other/right_touch'])

    if (left_raw.ndim != 2 or right_raw.ndim != 2
            or left_raw.shape[0] == 0 or left_raw.shape[0] != right_raw.shape[0]
            or any(raw.shape[1] not in (F1_RAW_DIM_PER_HAND, F1_NORM_DIM_PER_HAND)
                   for raw in (left_raw, right_raw))):
        raise ValueError('F1 touch must be aligned nonempty [N, 29] or [N, 37] arrays for both hands')
    # 37-D input is already normalized. Rebuilding splits is safe after an
    # interrupted historical conversion and must never divide forces twice.
    left_norm = normalize_f1_touch(left_raw.astype(np.float32)) if left_raw.shape[1] == F1_RAW_DIM_PER_HAND else left_raw
    right_norm = normalize_f1_touch(right_raw.astype(np.float32)) if right_raw.shape[1] == F1_RAW_DIM_PER_HAND else right_raw

    replace_dataset(f, 'observations/proprioceptions/other/left_touch', left_norm)
    replace_dataset(f, 'observations/proprioceptions/other/right_touch', right_norm)

    # split into per-finger datasets matching dftp key names
    wi = 0
    for fname in F1_FINGER_NAMES:
        tkey = F1_FINGER_TO_TACTILE_KEY[fname]
        replace_dataset(f, f'observations/tactile/left_{tkey}_tactile',
                        left_norm[:, wi:wi + F1_FINGER_NORM_DIM])
        replace_dataset(f, f'observations/tactile/right_{tkey}_tactile',
                        right_norm[:, wi:wi + F1_FINGER_NORM_DIM])
        wi += F1_FINGER_NORM_DIM

    # concatenate all 3 palm zones into a single palm tactile dataset
    palm_start = wi
    palm_end = palm_start + len(F1_PALM_NAMES) * F1_PALM_NORM_DIM
    replace_dataset(f, 'observations/tactile/left_palm_tactile',
                    left_norm[:, palm_start:palm_end])
    replace_dataset(f, 'observations/tactile/right_palm_tactile',
                    right_norm[:, palm_start:palm_end])

    print(f'  ✓ Successfully processed F1 tactile ({left_raw.shape[0]} frames, '
          f'{left_raw.shape[1]}d -> {F1_NORM_DIM_PER_HAND}d per hand)')


def process_dftp_tactile(f):
    """Process DFTP hand tactile data: split into per-finger datasets."""
    # fingerone=pinky, fingertwo=ring, fingerthree=middle, fingerfour=index, fingerfive=thumb
    fingerone_end = INSPIRE_TOUCH_FIELD_SIZES['fingerone_tip_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerone_top_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerone_palm_touch']
    fingertwo_end = fingerone_end + INSPIRE_TOUCH_FIELD_SIZES['fingertwo_tip_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingertwo_top_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingertwo_palm_touch']
    fingerthree_end = fingertwo_end + INSPIRE_TOUCH_FIELD_SIZES['fingerthree_tip_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerthree_top_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerthree_palm_touch']
    fingerfour_end = fingerthree_end + INSPIRE_TOUCH_FIELD_SIZES['fingerfour_tip_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerfour_top_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerfour_palm_touch']
    fingerfive_end = fingerfour_end + INSPIRE_TOUCH_FIELD_SIZES['fingerfive_tip_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerfive_top_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerfive_middle_touch'] + INSPIRE_TOUCH_FIELD_SIZES['fingerfive_palm_touch']
    palm_end = fingerfive_end + INSPIRE_TOUCH_FIELD_SIZES['palm_touch']

    left_touch = np.array(f['observations/proprioceptions/other/left_touch'])
    right_touch = np.array(f['observations/proprioceptions/other/right_touch'])
    replace_dataset(f, 'observations/proprioceptions/other/left_touch', left_touch)
    replace_dataset(f, 'observations/proprioceptions/other/right_touch', right_touch)

    replace_dataset(f, 'observations/tactile/left_pinky_tactile', left_touch[:, :fingerone_end])
    replace_dataset(f, 'observations/tactile/right_pinky_tactile', right_touch[:, :fingerone_end])
    replace_dataset(f, 'observations/tactile/left_ring_tactile', left_touch[:, fingerone_end:fingertwo_end])
    replace_dataset(f, 'observations/tactile/right_ring_tactile', right_touch[:, fingerone_end:fingertwo_end])
    replace_dataset(f, 'observations/tactile/left_middle_tactile', left_touch[:, fingertwo_end:fingerthree_end])
    replace_dataset(f, 'observations/tactile/right_middle_tactile', right_touch[:, fingertwo_end:fingerthree_end])
    replace_dataset(f, 'observations/tactile/left_index_tactile', left_touch[:, fingerthree_end:fingerfour_end])
    replace_dataset(f, 'observations/tactile/right_index_tactile', right_touch[:, fingerthree_end:fingerfour_end])
    replace_dataset(f, 'observations/tactile/left_thumb_tactile', left_touch[:, fingerfour_end:fingerfive_end])
    replace_dataset(f, 'observations/tactile/right_thumb_tactile', right_touch[:, fingerfour_end:fingerfive_end])
    replace_dataset(f, 'observations/tactile/left_palm_tactile', left_touch[:, fingerfive_end:palm_end])
    replace_dataset(f, 'observations/tactile/right_palm_tactile', right_touch[:, fingerfive_end:palm_end])

    print(f'  ✓ Successfully processed DFTP tactile')


def process_tactile(f, hand_type='auto'):
    """Validate the two hands and create or rebuild all tactile splits."""
    left = f['observations/proprioceptions/other/left_touch']
    right = f['observations/proprioceptions/other/right_touch']
    if (len(left.shape) != 2 or len(right.shape) != 2
            or left.shape[0] != right.shape[0] or left.shape[0] == 0):
        raise ValueError('Tactile inputs must be aligned nonempty [N, D] arrays for both hands')
    widths = (left.shape[1], right.shape[1])
    dftp_width = sum(INSPIRE_TOUCH_FIELD_SIZES.values())
    inferred = ('f1' if all(width in (F1_RAW_DIM_PER_HAND, F1_NORM_DIM_PER_HAND) for width in widths)
                else 'dftp' if widths == (dftp_width, dftp_width) else None)
    if inferred is None or hand_type not in ('auto', inferred):
        raise ValueError(f'Tactile widths {widths} do not match requested hand type {hand_type}')
    declared = f.attrs.get('hand_type')
    if isinstance(declared, bytes):
        declared = declared.decode('utf-8')
    if declared is not None and str(declared).lower() != inferred:
        raise ValueError(f'hand_type metadata {declared!r} conflicts with tactile widths {widths}')
    if inferred == 'f1':
        process_f1_tactile(f)
        f.attrs['tactile_format'] = 'f1_normalized_37_v1'
    else:
        process_dftp_tactile(f)
        f.attrs['tactile_format'] = 'dftp_normalized_1062_v1'
    f.attrs['hand_type'] = inferred


def main(argv=None):
    parser = argparse.ArgumentParser(description='Process Inspire hand tactile data (DFTP or F1)')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Offline working copy containing HDF5 episodes; each successful file is atomically replaced')
    parser.add_argument('--hand_type', type=str, default='auto', choices=['auto', 'dftp', 'f1'],
                        help='Hand type: auto-detect from data, or require dftp/f1')
    parser.add_argument('--skip_mirrored', action='store_true', default=True,
                        help='Skip mirrored data files (default)')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.dataset_dir):
        parser.error('--dataset_dir must be an existing directory')
    hdf5_files = find_all_hdf5(args.dataset_dir, skip_mirrored_data=args.skip_mirrored)
    if not hdf5_files:
        print('No eligible HDF5 episodes found')
        return 1
    errors = 0
    for index, path in enumerate(hdf5_files, 1):
        print(f'[{index}/{len(hdf5_files)}] Processing: {path}')
        try:
            atomic_process_hdf5(path, lambda f: process_tactile(f, args.hand_type))
        except Exception as exc:
            errors += 1
            print(f'Error processing {path}: {type(exc).__name__}: {exc}')
    print(f'Processing summary: files={len(hdf5_files)} successful={len(hdf5_files) - errors} errors={errors}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
