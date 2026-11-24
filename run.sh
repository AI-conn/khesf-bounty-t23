#!/bin/bash
# One-command, offline, zero-dependency run.
set -euo pipefail
cd "$(dirname "$0")"
python3 kinematics_auth.py
