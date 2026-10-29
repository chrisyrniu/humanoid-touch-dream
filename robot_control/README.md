# G1 robot and Inspire hand control

Run this controller and the matching Inspire hand driver alongside either the
[data collector](../data_collect/README.md) or
[policy rollout](../policy_rollout/README.md). Use one command source at a time.
Policy training does not require this environment.

## Required hardware and assets

Use a compatible G1 with Inspire DFTP Ethernet hands (the paper setup), or F1
hands with one USB/RS485 adapter per hand. Select the G1 wrist variant separately
from the hand type:

| Controller option | Included robot description |
| --- | --- |
| `--g1_type 0` (default) | `assets/g1/g1_body29_hand14_wrist4010.urdf` |
| `--g1_type 1` | `assets/g1/g1_body29_hand14_wrist5010.urdf` |

Robot descriptions, meshes, and the `inspire_hand_dftp.yml` / `inspire_hand_f1.yml`
retargeting profiles are included under `assets/`; keep their relative paths.
The controller loads the bundled
[student_policy_jit.pt](g1_wbc_policy/student_policy_jit.pt) with its matching
[g1_student_htd.yaml](g1_wbc_policy/g1_student_htd.yaml), including its gains,
command ranges, and waist limits. Downloading the WBC submodule is not required.
This WBC checkpoint is separate from the manipulation policy trained in `htd_policy/`.

## Install from the repository root

Use a dedicated Python 3.10 environment; this numerical stack requires NumPy 1.x.
Start in the cloned repository root:

```bash
export HTD_ROOT="$(pwd)"
export HTD_DEPS="$(dirname "$HTD_ROOT")/htd-deps"
conda create -n htd_robot_control python=3.10 pinocchio=3.1.0 \
  casadi=3.6.7 numpy=1.26.4 pip -c conda-forge
conda activate htd_robot_control
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "numpy==1.26.4" \
  "pytransform3d>=3.5.0" "nlopt>=2.6.1,<2.8.0" "trimesh>=4.4.0" \
  "anytree>=2.12.0" "pyyaml>=6.0.0" "lxml>=5.2.2"
python -m pip install --no-deps -e "$HTD_ROOT/robot_control/dex-retargeting"
python -m pip install -e "$HTD_ROOT/robot_control"
```

`--no-deps` preserves conda's Pinocchio/CasADi bindings. `pip check` may report
that the PyPI package `pin` is missing; verify the installed bindings below.

Install the [Unitree SDK](https://github.com/unitreerobotics/unitree_sdk2_python)
and download the hand SDK. Skip a clone if that checkout already exists.

```bash
mkdir -p "$HTD_DEPS"
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  "$HTD_DEPS/unitree_sdk2_python"
python -m pip install "numpy==1.26.4" -e "$HTD_DEPS/unitree_sdk2_python"
git clone https://github.com/chrisyrniu/inspire_hands_unitree_sdk.git \
  "$HTD_DEPS/inspire_hands_unitree_sdk"
```

If `cyclonedds==0.10.2` cannot find its native library, follow the Unitree SDK's
Cyclone DDS build instructions and set `CYCLONEDDS_HOME` before retrying.

Install **one** hand SDK per environment. For DFTP:

```bash
python -m pip install "numpy==1.26.4" \
  -e "$HTD_DEPS/inspire_hands_unitree_sdk/dftp/inspire_hand_sdk"
```

For F1, use this instead:

```bash
python -m pip install "numpy==1.26.4" "pymodbus>=3.10,<4" \
  -e "$HTD_DEPS/inspire_hands_unitree_sdk/f1"
```

DFTP requires PyModbus 3.6.9; F1 requires 3.10 or newer, below version 4. Keep
these installations separate. The collection and rollout guides describe
installing the selected SDK in their environments too.

Check the numerical bindings without opening hardware:

```bash
python -c 'import numpy, pinocchio; from pinocchio import casadi; import dex_retargeting; print(numpy.__version__, pinocchio.__version__)'
```

## Hardware handoff and startup

**Controller startup moves the robot toward its default pose before the `s`
prompt.** Prepare the robot's support/stop procedure and release its built-in
motion controller for low-level motor control before launch. See the
`MotionSwitcherClient` handoff in [Unitree's G1 example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py);
this controller does not switch the robot's mode itself.

Follow the [Unitree SDK network setup](https://github.com/unitreerobotics/unitree_sdk2_python#usage).
The controller and hand drivers use DDS domain 0 with automatic interface
selection. If the host has multiple network interfaces, edit
`ChannelConfigAutoDetermine` in the installed SDK's
[channel_config.py](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/core/channel_config.py):
replace `autodetermine="true"` with `name="YOUR_ROBOT_INTERFACE"`, using the
interface connected to the robot. Restart the controller and hand drivers after
changing it. Set the collector/rollout's `--dds_interface` separately; the
controller itself has no such option.

Use separate terminals for the hand driver, controller, and collector/rollout.
In each controller terminal, start from the repository root and run:

```bash
conda activate htd_robot_control
cd robot_control
```

### DFTP hands

Match the IPs in [inspire_dftp.py](hand_drivers/inspire_dftp.py) to the rig:
left `192.168.123.210`, right `192.168.123.211`, Modbus device ID 1.
These are configured in the file, not through CLI options.

In the first terminal, start the hand driver:

```bash
python hand_drivers/inspire_dftp.py
```

In the second, start the controller. Choose `--g1_type` from the table above:

```bash
python g1_inspire_wb_controller.py --g1_type 0 --hand_type dftp
```

Enter `s` followed by Enter when prompted. Then start
[data collection](../data_collect/README.md#dftp-with-head-and-wrist-cameras)
or [policy rollout](../policy_rollout/README.md) in its own environment.

### F1 hands

Use these commands in place of the DFTP commands. Ensure the user can access
the serial device files and identify the left/right adapters. The driver uses
hand ID 1 and 115200 baud; override with `--id-left`, `--id-right`, or `--baudrate`
if needed. `--touch` enables tactile feedback.

First terminal:

```bash
python hand_drivers/inspire_f1.py \
  --port-left /dev/ttyUSB0 --port-right /dev/ttyUSB1 --touch
```

Second terminal (this example uses the 5010-wrist description):

```bash
python g1_inspire_wb_controller.py --g1_type 1 --hand_type f1
```

Enter `s` when prompted. Select `--inspire_hand_type f1` in collection/rollout,
and use the F1 training configuration for F1 data.

### Start, reset, and stop the controller

| Input | Behavior |
| --- | --- |
| `s` + Enter | Starts the main loop after initialization |
| Remote button index `[2]`, collector reset gesture, or rollout reset key | Starts a three-second reset; opens the hands and returns to the default arm/torso targets |
| Remote button index `[3]` or Ctrl+C | Exits the main loop and requests damping |

The remote indices refer to the SDK button array; confirm their physical labels.
Wait for reset completion before recalibrating the collector or starting rollout.

Pausing or exiting collection/rollout leaves this controller running with the
latest received targets. Command publishing can resume without a reset.

The default control and feedback rates are 50 Hz, matching the supplied WBC
configuration. See the [collection controls](../data_collect/README.md#calibration-episodes-and-shutdown)
and [rollout controls](../policy_rollout/README.md#operate-rollout).