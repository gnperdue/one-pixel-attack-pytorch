#!/bin/bash

EXE="train.py"

ARGS=" --epochs 1"
ARGS+=" --lr 0.01"
# ARGS+=" --resume"
ARGS+=" --use-feature-extractor"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
