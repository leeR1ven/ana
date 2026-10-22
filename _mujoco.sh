#!/bin/bash
cd /root/ana
python3 -m pip install -q mujoco 2>&1 | tail -2
python3 -c "import mujoco; print('mujoco', mujoco.__version__)"
python3 -c "import sys; sys.path.insert(0,'.'); from agi.dog_tasks import run_skill; print('dog_tasks ok')"
