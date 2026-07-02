#!/usr/bin/env bash
set -euo pipefail

# Launch Crazyswarm2 with the UDP-compatible Python backend.
# Usage:
#   ./scripts/launch_crazyswarm_udp.sh [script] [backend]
# Examples:
#   ./scripts/launch_crazyswarm_udp.sh
#   ./scripts/launch_crazyswarm_udp.sh hello_world cflib

SCRIPT_NAME="${1:-hello_world}"
BACKEND="${2:-cflib}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv-crazyswarm"
WS_PATH="$ROOT_DIR/crazyswarm2_ws"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Missing virtual environment: $VENV_PATH"
  echo "Create it first following README instructions."
  exit 1
fi

if [[ ! -f "$WS_PATH/install/setup.bash" ]]; then
  echo "Missing ROS workspace overlay: $WS_PATH/install/setup.bash"
  echo "Build the workspace first with colcon."
  exit 1
fi

cd "$WS_PATH"

source "$VENV_PATH/bin/activate"

# ROS setup scripts can reference unset shell vars; relax nounset while sourcing.
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

# ROS Python entrypoints may use system shebangs; prepend venv site-packages.
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.10/site-packages:${PYTHONPATH:-}"

exec ros2 launch crazyflie_examples launch.py script:="$SCRIPT_NAME" backend:="$BACKEND"
