#!/bin/bash


#EXP_DIR=$(dirname $(readlink -f $0))
HOME=/Users/wonkyoc/git
MODEL_DIR=$HOME/models
EXP_DIR=$HOME/project-diffusion-experiments
#SDCPP_DIR=$HOME/stable-diffusion.cpp
#DIFF_DIR=$HOME/diffusers

# hyperparameters
THREADS=(4)
BATCHS=(1)
DEVICES=("mps")
STEPS=10
ITER=1
PROMPT="an oil surrealist painting of a dreamworld on a seashore where clocks and watches appear to be inexplicably limp and melting in the desolate landscape. a table on the left, with a golden watch swarmed by ants. a strange fleshy creature in the center of the painting"


run_diffusers() {
    echo "Run Diffusers"
    python $EXP_DIR/01-parallelism/main.py \
        --prompt "$PROMPT" \
        --threads "$1" \
        --steps $STEPS \
        --iteration $ITER \
        --batch_size "$2" \
        --device "$3"

}

run_sdcpp() {
    echo "Run stable-diffusion.cpp"
    ${SDCPP_DIR}/build/bin/sd -m "$MODEL_DIR/sd-v1-5.ckpt" \
        --sampling-method euler_a \
        --prompt "$PROMPT" \
        --threads $THREADS    \
        --steps $STEPS   \
        -v --color
}

#for d in "${DEVICES[@]}"; do
#    for b in "${BATCHS[@]}"; do
#        for t in "${THREADS[@]}"; do
#            run_diffusers $t $b $d
#            #run_sdcpp
#            sleep 5
#        done
#    done
#done

# 1 CPU + 1 GPU
run_diffusers 4 1 "cpu" &
run_diffusers 4 3 "mps"
