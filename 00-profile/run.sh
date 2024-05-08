#!/bin/bash


#EXP_DIR=$(dirname $(readlink -f $0))
#HOME=/home/wonkyoc/git
HOME=/p/alpha
MODEL_DIR=$HOME/models
EXP_DIR=$HOME/project-diffusion-experiments
SDCPP_DIR=$HOME/stable-diffusion.cpp
DIFF_DIR=$HOME/diffusers
LOG_DIR=$HOME/project-diffusion-experiments/00-profile/results

# hyperparameters
THREADS=(1 2 4 8 12 20 24)
#PIN="taskset -c 4-7"
STEPS=1
PROMPT="an oil surrealist painting of a dreamworld on a seashore where clocks and watches appear to be inexplicably limp and melting in the desolate landscape. a table on the left, with a golden watch swarmed by ants. a strange fleshy creature in the center of the painting"


run_diffusers() {
    echo "Run Diffusers"
    $PIN python $EXP_DIR/01-parallelism/main.py \
        --prompt "$1" \
        --threads $2 \
        --steps $3 > $LOG_DIR/diffusers-t$2.log
}

run_sdcpp() {
    echo "Run stable-diffusion.cpp"
    $PIN $SDCPP_DIR/build/bin/sd -m $MODEL_DIR/sd-v1-5.ckpt \
        --sampling-method euler_a \
        --prompt "$1" \
        --threads $2    \
        --steps $3   \
        -v --color > $LOG_DIR/sdcpp-t$2.log
}

for t in ${THREADS[@]}; do
    run_diffusers $PROMPT $t $STEPS
    run_sdcpp $PROMPT $t $STEPS
done
