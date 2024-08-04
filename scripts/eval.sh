#!/bin/bash

MUJOCO_GL=${MUJOCO_GL:-egl} python cambrian/ml/trainer.py --eval \
    hydra.sweeper.params='${clear:}' +hydra.sweeper.optim.max_batch_size=null hydra/sweeper=basic \
    $@
