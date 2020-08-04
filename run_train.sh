#!/bin/bash

EXE="train.py"

ARGS=" --epochs 1"
ARGS+=" --freeze-conv"
ARGS+=" --lr 0.01"
# ARGS+=" --resume"
ARGS+=" --test"
ARGS+=" --train"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
