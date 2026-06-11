# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper for tracer training.

The shared root ``train.py`` now owns both physical and tracer training. This
wrapper keeps older commands working by defaulting to ``--training_mode=tracer``.
"""

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train import main


def has_training_mode(argv):
    return any(arg == "--training_mode" or arg.startswith("--training_mode=") for arg in argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not has_training_mode(argv):
        argv = ["--training_mode=tracer", *argv]
    main(argv)
