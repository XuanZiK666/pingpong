#!/usr/bin/env python3
"""Launcher for VLASH inference on x1 robot.

Registers the worobots X1RobotConfig with draccus before invoking vlash.run,
so that `robot.type: x1_robot` in the YAML is recognized.

Usage:
    python run_inference.py --config_path=inference_pingpong.yaml
"""

import sys
from pathlib import Path

# Make worobots importable
worobots_path = str(Path(__file__).resolve().parent / "worobots")
if worobots_path not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Register X1RobotConfig so draccus can resolve `type: x1_robot`
from worobots.x1_robot import X1RobotConfig  # noqa: F401

# Now invoke vlash run (uses lerobot's parser.wrap())
from vlash.run import main

if __name__ == "__main__":
    main()