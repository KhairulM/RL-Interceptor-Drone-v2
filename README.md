# RLInterceptorDrone

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fork of [OmniDrones](https://github.com/btx0424/OmniDrones) focused on training a multi-rotor **interceptor** drone via reinforcement learning. The repository ships an `Intercept` task in which a *pursuer* drone learns to catch a moving *evader* that follows scripted trajectories (hover, linear, circular).

> Heavy lifting (simulation, robot models, RL stack) comes from upstream OmniDrones. This README only covers what is needed to install the environment and train/play the `Intercept` task.

## Prerequisites

- Linux (tested on Ubuntu) with an NVIDIA RTX 5090
- NVIDIA driver compatible with **Isaac Sim 5.1.0**
- **Python 3.11** (Isaac Sim 5.1 ships and is tested with 3.11)
- A working installation of [NVIDIA Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html)

## Installation

The project is laid out as a standard Python package at the repo root. Install it into a Python 3.11 virtual environment that has access to Isaac Sim's Python modules.

### 1. Clone

```bash
git clone git@github.com:KhairulM/RL-Interceptor-Drone-v2.git RLInterceptorDrone
cd RLInterceptorDrone
git submodule update --init --recursive
```

### 2. Create the virtual environment

```bash
python3.11 -m venv --prompt="interceptor" .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Isaac Sim Python packages

Follow the [official Isaac Sim 5.1 pip install instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html). In short, with the venv active:

```bash
pip install isaacsim==5.1.* --extra-index-url https://pypi.nvidia.com
```

### 4. Install this package (editable)

From the repo root:

```bash
pip install -e .
```

This installs `omni_drones` and its Python dependencies (`hydra-core`, `omegaconf`, `torchrl`, `tensordict`, `wandb`, `numpy`, `scipy`, `imageio`, `moviepy`, ...).

### 5. Verify

```bash
python -c "import omni_drones; print(omni_drones.__file__)"
```

It should print a path inside this repository.

## Optional: Crazyswarm2 isolated environment (recommended)

Use a separate Python **3.10** virtual environment for Crazyswarm2 and Crazyflie tooling.
This avoids dependency conflicts with the main Isaac Sim / interceptor environment.

### Why separate envs?

- The main interceptor stack is Python 3.11 + Isaac Sim.
- `cflib` currently pulls dependency versions (for example `packaging~=25`) that can conflict with Isaac-related pins.
- Keeping Crazyswarm2 in its own env makes both workflows reproducible.

### Setup steps (from repo root)

```bash
# 1) Create and activate a dedicated env
python3.10 -m venv --prompt crazyswarm .venv-crazyswarm
source .venv-crazyswarm/bin/activate

# 2) Upgrade packaging tools
pip install -U pip setuptools wheel

# 3) Install common Python deps required by ROS/Crazyswarm Python tools
pip install Jinja2 pyyaml typeguard

# 4) Install local cflib from this repo vendored source
cd libs/crazyflie-lib-python
pip install .

# 5) Install math/geometry helpers used in this project
pip install rowan transforms3d

# 6) Validate environment consistency
pip check
```

You should see `No broken requirements found.` at the end.

### Install Crazyswarm2 source tree

From the workspace root:

```bash
mkdir -p crazyswarm2_ws/src
cd crazyswarm2_ws/src
git clone https://github.com/IMRCLab/crazyswarm2 --recursive
```

### Build Crazyswarm2 (ROS 2 Humble)

With `.venv-crazyswarm` active:

```bash
cd ~/Projects/RLInterceptorDrone
source .venv-crazyswarm/bin/activate

# Source ROS 2 first (required for colcon/ament)
source /opt/ros/humble/setup.zsh

# Required Python build deps used by ROS package parsing/generation
pip install -U catkin_pkg lark-parser

# ROS Humble expects EmPy 3.x (4.x breaks rosidl templates)
pip uninstall -y em empy
pip install empy==3.3.4

cd crazyswarm2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

After a successful build, source overlays before running nodes:

```bash
cd ~/Projects/RLInterceptorDrone
source .venv-crazyswarm/bin/activate
source /opt/ros/humble/setup.zsh
source crazyswarm2_ws/install/setup.zsh
```

### Notes

- If you see resolver warnings immediately after `pip install -U pip setuptools wheel`, run `pip check` and install missing packages as reported.
- Keep `.venv` (interceptor) and `.venv-crazyswarm` (Crazyswarm2) separate; do not mix installs between them.
- If `colcon build` fails with `No module named 'catkin_pkg'`, install `catkin_pkg` in `.venv-crazyswarm`.
- If `colcon build` fails with `AttributeError: module 'em' has no attribute 'BUFFERED_OPT'`, pin `empy==3.3.4` in `.venv-crazyswarm`.

## Running the Intercept task

The `Intercept` task lives at [omni_drones/envs/single/intercept.py](omni_drones/envs/single/intercept.py) and is configured via [cfg/task/Intercept.yaml](cfg/task/Intercept.yaml). A pursuer drone (`RateController`) learns to catch an evader (`LeePositionController`) that follows scripted hover/linear/circular trajectories.

All commands below assume the venv is activated and you are at the repo root.

### Train (headless, recommended)

```bash
cd scripts
python train.py task=Intercept headless=true env.num_envs=256
```

Useful overrides (Hydra syntax — `key=value`, dotted paths allowed):

| Override | Purpose |
|---|---|
| `task=Intercept` | Select the Intercept task config |
| `algo=mappo` | RL algorithm (default; see [cfg/algo/](cfg/algo/)) |
| `env.num_envs=256` | Parallel environments (start small; see note below) |
| `task.action_transform=rate` | Use the body-rate action transform (default) |
| `task.action_transform=null` | Disable the rate transform (raw rotor commands) |
| `headless=true` | Run without the Isaac Sim GUI (faster) |
| `total_frames=10_000_000` | Cap training frames |
| `wandb.mode=disabled` | Disable Weights & Biases logging |
| `seed=0` | RNG seed |

Outputs (checkpoints, configs, videos) are written under `scripts/outputs/<date>/<time>/` by Hydra.

### Play / evaluate a trained policy

Use `scripts/play.py` with a checkpoint produced by training. The checkpoint
override key is `checkpoint`.

```bash
cd scripts
python play.py task=Intercept headless=false \
    checkpoint=outputs/<date>/<time>/checkpoint_final.pt
```

Set `headless=false` to watch the rollout in the Isaac Sim viewport.

#### Evaluate OOD: random evader trajectory only

To test generalization when the policy was trained on linear + zigzag, force
the evader to use only the random strategy at evaluation time:

```bash
cd scripts
python play.py task=Intercept headless=false \
    checkpoint=outputs/<date>/<time>/checkpoint_final.pt \
    'task.evader.trajectory_types=[random]'
```

Notes:

- Quote the list override in zsh (`'task.evader.trajectory_types=[random]'`) to avoid shell glob expansion.
- Use `headless=true` for faster, non-visual batch evaluation.
- For mixed evaluation, use e.g. `'task.evader.trajectory_types=[linear,zigzag,random]'`.

### Tuning the Intercept scenario

Common knobs in [cfg/task/Intercept.yaml](cfg/task/Intercept.yaml):

- `pursuer.model` / `pursuer.controller` — drone model and low-level controller of the interceptor
- `evader.model` / `evader.controller` — drone model and controller of the target
- `evader.trajectory_types` — list sampled per env (`linear`, `zigzag`, `velocity_obstacle`, `random`)
- `evader.speed_range`, `evader.spawn_distance_range`, `evader.bounds`, `evader.boundary_mode` — evader motion limits
- `success_radius`, `reset_thres`, `reward_distance_scale` — reward / termination shaping
- `env.num_envs`, `env.max_episode_length`, `env.env_spacing` — sim batching

### Notes

- On RTX 5090 / Isaac Sim 5.1, the verified-stable Intercept setup uses a **homogeneous Hummingbird pursuer + Hummingbird evader** scene (already the default). Mixing different drone models in the same scene at high `env.num_envs` has been observed to crash the PhysX GPU backend.
- Start with `env.num_envs=256` and scale up only after confirming stability on your hardware.

## Repository layout

```
cfg/                Hydra configs (task/, algo/, base/)
omni_drones/        Python package: envs, robots, controllers, learning, sensors, utils
scripts/            Entry points: train.py, play.py (Hydra apps reading cfg/)
examples/           Standalone demo scripts
docs/               Sphinx documentation source
```

## Upstream

This repo descends from [btx0424/OmniDrones](https://github.com/btx0424/OmniDrones) (configured as the `upstream` remote). For the broader feature set, multi-agent benchmarks, and original documentation, see the [upstream docs](https://omnidrones.readthedocs.io/en/latest/).

## Citation

If you use this work, please cite the original OmniDrones paper:

```bibtex
@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control},
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Acknowledgement

Built on top of [OmniDrones](https://github.com/btx0424/OmniDrones) and [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html). Some abstractions are inspired by [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
