# Policy training

Train HTD, MXT, and ACT policies from your processed G1 demonstrations.
HTD enables tactile and force dreaming with `train_mxt.py --touch_dreaming`.

## Installation

From the repository root:

```bash
conda create -n htd_policy python=3.10 pip
conda activate htd_policy
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e htd_policy
cd htd_policy
```

Training requires CUDA. Use the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/)
for another GPU/driver combination. Training uses BF16 by default; add
`--no-bf16` to use FP32 on GPUs without BF16 support. The first run downloads
pretrained ResNet18 weights unless they are already cached.

All commands below run from `htd_policy/`. In a new terminal, activate
`htd_policy` and enter that directory from the repository root.

## Prepare your data

Collect and process episodes using the [data guide](../data_collect/README.md).
Point `--dataset_dir` at a directory containing the processed episodes:

```text
my_task/
└── g1_inspire_wb/
    └── YYYYMMDD_HHMMSS/
        ├── episode_1.hdf5
        └── episode_2.hdf5
```

The loader finds episodes recursively, sorts their paths, and holds out the
last episodes for validation. Session directories can stay as recorded;
folder names do not define the train/validation split. The examples use `--min_val_num 1` and
require at least two episodes. Adjust this and `--train_ratio` for your dataset.
Normalization uses all supplied episodes; `--stats_dir /path/to/statistics_data`
selects another corpus, such as training-only episodes.

## Train HTD

For DFTP hands with head and wrist cameras:

```bash
python train_mxt.py --touch_dreaming \
  --dataset_dir /path/to/processed/my_task \
  --ckpt_dir ../runs/run --task_name my_task --min_val_num 1
```

The run is saved to `runs/run_my_task_HTD/` at the repository root.
`--ckpt_dir` is a prefix: the trainer appends the task and method. Choose a new
prefix for a fresh run. `--task_name` is a label, not a model selector.

EMA collapse metrics exclude padded future targets.

To select a profile, add `--embodiment_config_path` and
`--trunk_config_path` with a matching pair:

| Profile | Embodiment YAML | Trunk YAML |
| --- | --- | --- |
| DFTP non-walk, 16-dim (default) | `configs/htd/dftp_nonwalk_latent16.yaml` | `configs/htd/trunk_nonwalk.yaml` |
| DFTP non-walk, 64-dim | `configs/htd/dftp_nonwalk_latent64.yaml` | `configs/htd/trunk_nonwalk.yaml` |
| DFTP walk, 16-dim | `configs/htd/dftp_walk_latent16.yaml` | `configs/htd/trunk_walk.yaml` |
| DFTP walk, 64-dim | `configs/htd/dftp_walk_latent64.yaml` | `configs/htd/trunk_walk.yaml` |
| F1, head + wrists | `configs/htd/f1.yaml` | `configs/htd/trunk_f1.yaml` |

DFTP non-walk profiles use head and wrist cameras. Walk profiles use only the
head camera and give more weight to velocity prediction. Each DFTP tactile
region uses the listed latent dimension and two tokens; F1 predicts raw tactile values.

Model settings are in [configs/htd/](configs/htd/); training options are listed by
`python train_mxt.py --help`. Keep custom YAMLs with the run for deployment.

## MXT training

For DFTP hands, omit `--touch_dreaming` and select a model without dream heads:

```bash
python train_mxt.py \
  --dataset_dir /path/to/processed/my_task \
  --ckpt_dir ../runs/run --task_name my_task --min_val_num 1 \
  --embodiment_config_path configs/mxt/touch.yaml \
  --trunk_config_path configs/htd/trunk_nonwalk.yaml --chunk_size 60 \
  --main_cam_width 1280 --main_cam_height 480 \
  --wrist_cam_width 640 --wrist_cam_height 480
```

Use `configs/mxt/visual.yaml` to omit touch. Both configurations use head and
wrist cameras. The run is saved to `runs/run_my_task_MXT/`.

## ACT training

For DFTP hands:

```bash
python train_act.py --decoder_only --config configs/act/touch.yaml \
  --dataset_dir /path/to/processed/my_task \
  --ckpt_dir ../runs/run --task_name my_task --min_val_num 1 \
  --hidden_dim 512 --chunk_size 60
```

Use `configs/act/visual.yaml` to omit touch. The corresponding
`touch_walk.yaml` and `visual_walk.yaml` use head-only images. The run is saved
to `runs/run_my_task_HIT/`. Run `python train_act.py --help` for other options.

## Logging, resume, and rollout

Logging is local. For W&B, run `wandb login`, set
`WANDB_ENTITY` to your account or team, and add `--wandb`. The project is `htd`;
`--wandb_name YOUR_PROJECT` selects another project.

To resume HTD/MXT, repeat the original command with `--auto_resume`, keeping the
same prefix, model, dataset, and statistics settings. `--num_steps` is the final
step index, not the number of additional updates.

For deployment, keep the checkpoint, saved configuration, and normalization
statistics together in the run directory and follow the [rollout guide](../policy_rollout/README.md).
