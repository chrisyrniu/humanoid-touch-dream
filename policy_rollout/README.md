# Policy rollout

Run a trained HTD, MXT, or ACT (HIT) policy alongside the
[robot controller](../robot_control/README.md). HTD and MXT use `--policy_class MXT`;
ACT rollout requires a model trained with `--decoder_only` (HIT) and uses
`--policy_class HIT`. `--touch_dreaming` is only needed during training.

## Install

Complete the [policy installation](../htd_policy/README.md#installation).
Prepare CycloneDDS and the SDK checkouts from
[controller setup](../robot_control/README.md#install-from-the-repository-root),
and the camera utilities from [collection setup](../data_collect/README.md#requirements-and-installation).
Use a Linux graphical desktop for camera selection and keyboard input.
Then run these commands from the repository root:

```bash
export HTD_ROOT="$(pwd)"
export HTD_DEPS="$(dirname "$HTD_ROOT")/htd-deps"
conda activate htd_policy
python -m pip install -r "$HTD_ROOT/policy_rollout/requirements-hardware.txt"
```

Install the selected hand SDKs into this same `htd_policy` environment using the
[SDK commands in collection setup](../data_collect/README.md#requirements-and-installation):
Unitree and Inspire SDKs for DFTP, or the Inspire F1 SDK for F1.
If your SDK checkouts are elsewhere, set `HTD_DEPS` to their parent directory.

## Prepare the checkpoint and hardware

`--model_path` must point to your training output directory with
`policy_last.ckpt`, `all_configs.json`, and `dataset_stats.pkl` kept together.
Use the original normalization statistics. For HTD/MXT, use the same
embodiment/trunk YAMLs as training, including custom latent dimensions. Rollout
resizes camera images to the input dimensions recorded in the training configuration.

Start the matching hand driver and G1 controller using the
[hardware startup instructions](../robot_control/README.md#hardware-handoff-and-startup).
Complete that handoff before launching the controller; it can move the robot
before its `s` prompt.

Match the cameras and hand type to training:

- The head camera must appear as `3D USB Camera`. At the default `480p`, it
  supplies a combined 1280×480 stereo image, rotated 180 degrees by rollout.
- With `--use_wrist_camera True`, connect two `Global Shutter Camera` devices.
  Label the previewed camera `l` or `r` followed by Enter; the other gets the
  remaining side. Each wrist image is 640×480 at `480p`.
- Match the hand driver, controller's `--hand_type`, and rollout's
  `--inspire_hand_type` (`dftp` or `f1`) to the installed hands and trained model.

Add `--dds_interface YOUR_INTERFACE` if needed. It selects the rollout interface
when `CYCLONEDDS_URI` is unset; the controller and hand driver need matching DDS
network settings.

## Run HTD or MXT

Open a terminal at the repository root. To run your checkpoint from the
[HTD training example](../htd_policy/README.md#train-htd) with DFTP hands and
head/wrist cameras:

```bash
conda activate htd_policy
cd policy_rollout
OMP_NUM_THREADS=1 python policy_rollout.py \
  --model_path ../runs/run_my_task_HTD \
  --policy_class MXT --embodiment g1_inspire_wb \
  --inspire_hand_type dftp \
  --use_wrist_camera True --send_velocity_commands False \
  --device cuda:0 --control_freq 30 --inference_interval 45 \
  --embodiment_config_path ../htd_policy/configs/htd/dftp_nonwalk_latent16.yaml \
  --trunk_config_path ../htd_policy/configs/htd/trunk_nonwalk.yaml
```

For another [model configuration](../htd_policy/README.md#train-htd) or
[MXT baseline](../htd_policy/README.md#mxt-training), select its run directory
and training YAMLs. Set `--use_wrist_camera False` for head-only models and
`--inspire_hand_type f1` for F1 hands. Enable `--send_velocity_commands True`
to send walking commands from a model trained to predict them.

## Run ACT (HIT)

The [ACT training example](../htd_policy/README.md#act-training) already uses
`--decoder_only` to train HIT. To deploy its checkpoint, start at the repository root:

```bash
conda activate htd_policy
cd policy_rollout
OMP_NUM_THREADS=1 python policy_rollout.py \
  --model_path ../runs/run_my_task_HIT --policy_class HIT \
  --inspire_hand_type dftp \
  --use_wrist_camera True --send_velocity_commands False \
  --device cuda:0 --control_freq 30 --inference_interval 45
```

The observation and action configuration is loaded from the saved training
metadata. For a head-only walking model, set `--use_wrist_camera False` and
`--send_velocity_commands True`.

## Operate rollout

Confirm live camera, robot, hand, and required tactile feedback. Rollout uses
the latest received observations without rejecting stale feedback.
To start, press **`1`**, wait for the controller reset to finish, then press **`2`**.

| Key | Action |
| --- | --- |
| `1` | Request a controller reset and restart the action chunk. |
| `2` | Start after reset, or resume a paused rollout. |
| `3` | Pause command publishing. |
| Ctrl+C | Exit rollout. |

Pausing or exiting rollout leaves the robot controller running with the last
received targets. Press **`2`** to resume a paused rollout.

Keep `--inference_interval` positive and no greater than the checkpoint's action
horizon (60 in the example).

See [starting, resetting, and stopping the controller](../robot_control/README.md#start-reset-and-stop-the-controller).

Add `--dream_touch True` to record an HTD model's dream predictions and
corresponding sensor observations under `MODEL_DIR/dream_touch_results/TIMESTAMP/`.
