#!/bin/bash


# xzl: win, cuda 
# env: gitbash, win10, vs2022 build

#EXP_DIR=$(dirname $(readlink -f $0))
#HOME=/home/wonkyoc/git
#HOME=/p/alpha
HOME=/d/workspace-stable-diffusion

SDCPP_DIR=$HOME/stable-diffusion.cpp
#SDCPP_DIR=/d/workspace-ggml/stable-diffusion.cpp
MODEL_DIR=/d/workspace-ggml/stable-diffusion.cpp/assets


EXP_DIR=$HOME/project-diffusion-experiments
DIFF_DIR=$HOME/diffusers
LOG_DIR=$HOME/project-diffusion-experiments/00-profile/results

# hyperparameters
#THREADS=(1 2 4 8 12 20 24)
THREADS=(1 2 4)

#PIN="taskset -c 4-7"
STEPS=1
PROMPT="an oil surrealist painting of a dreamworld on a seashore where clocks and watches appear to be inexplicably limp and melting in the desolate landscape. a table on the left, with a golden watch swarmed by ants. a strange fleshy creature in the center of the painting"


run_diffusers() {
    echo "Run Diffusers"
    $PIN python $EXP_DIR/01-parallelism/main.py \
        --prompt "$PROMPT" \
        --threads $1 \
        --steps $2 > $LOG_DIR/$HOSTNAME-diffusers-t$1.log
}

run_sdcpp() {
    echo "Run stable-diffusion.cpp"
	mkdir -p $EXP_DIR/results
	
    $PIN $SDCPP_DIR/build-cuda/bin/Release/sd.exe -m $MODEL_DIR/sd-v1-4.ckpt \
        --sampling-method euler_a \
        --prompt "$PROMPT" \
        --threads $1    \
        --steps $2   \
        -v > $LOG_DIR/$HOSTNAME-sdcpp-debug-t$1.log
}

# dont vary threads, but only batch count
run_sdcpp_gpu() {
    echo "Run stable-diffusion.cpp"
	mkdir -p $EXP_DIR/results
	
    for b in 1 2 4
    do 
        $PIN $SDCPP_DIR/build-cuda/bin/Release/sd.exe -m $MODEL_DIR/sd-v1-4.ckpt \
            --sampling-method euler_a \
            --prompt "$PROMPT" \
            --batch-count $b    \
            --steps $1   \
            -v > $LOG_DIR/$HOSTNAME-sdcpp-gpu-b-$b.log
    done
}


# for t in ${THREADS[@]}; do
#     #run_diffusers $t $STEPS
#     # run_sdcpp $t $STEPS
# done

run_sdcpp_gpu $STEPS