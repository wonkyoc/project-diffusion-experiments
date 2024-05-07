#!/bin/bash


#EXP_DIR=$(dirname $(readlink -f $0))
HOME=/home/wonkyoc/git
MODEL_DIR=$HOME/models
EXP_DIR=$HOME/diffusion-experiments
SDCPP_DIR=$HOME/stable-diffusion.cpp
DIFF_DIR=$HOME/diffusers

# hyperparameters
THREADS=24
STEPS=1
PROMPT="an oil surrealist painting of a dreamworld on a seashore where clocks and watches appear to be inexplicably limp and melting in the desolate landscape. a table on the left, with a golden watch swarmed by ants. a strange fleshy creature in the center of the painting"


run_diffusers() {
    echo "Run Diffusers"
    python $EXP_DIR/01-parallelism/main.py \
        --prompt "$PROMPT" \
        --threads $THREADS \
        --steps $STEPS
}

run_sdcpp() {
    echo "Run stable-diffusion.cpp"
    $SDCPP_DIR/build/bin/sd -m $MODEL_DIR/sd-v1-5.ckpt \
        --sampling-method euler_a \
        --prompt "$PROMPT" \
        --threads $THREADS    \
        --steps $STEPS   \
        -v --color
}

run_diffusers
run_sdcpp

