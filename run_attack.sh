#!/bin/bash

EXE="attack.py"

ARGS=" --model resnet18"
ARGS+=" --pixels 1"
ARGS+=" --maxiter 100"
ARGS+=" --popsize 400"
ARGS+=" --samples 100"
ARGS+=" --targeted"
ARGS+=" --save ./results/results.pkl"
ARGS+=" --verbose"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
