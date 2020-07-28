#!/bin/bash

EXE="train.py"

ARGS=" --epochs 1"
ARGS+=" --lr 0.1"
# ARGS+=" --resume"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
