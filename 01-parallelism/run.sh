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
    t=$(date "+%Y%m%d.%H%M%S")

    mkdir -p "$t"
    
    /Users/wonkyoc/miniconda3/bin/python $EXP_DIR/01-parallelism/main.py \
        --prompt "$PROMPT" \
        --threads "$1" \
        --steps $STEPS \
        --iteration $ITER \
        --batch_size "$2" \
        --num_cpu_instances "$3" \
        --num_gpu_instances "$4" \
        --log_dir $t

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

cores=1
images=1
proc=1   # must <= images


bs=$((images / proc)) # images per process
tr=$((cores / proc))  # thr per process

#for d in "${DEVICES[@]}"; do
#    for b in "${BATCHS[@]}"; do
#        for t in "${THREADS[@]}"; do
#            run_diffusers $t $b $d
#            #run_sdcpp
#            sleep 5
#        done
#    done
#done

#for i in $(seq 1 $proc);
#do
#    run_diffusers $tr $bs "cpu" $proc >> $HOSTNAME-diffusers-instance-$i.stdout  2>&1 &
#    # >> test-mps-bs2.stdout  2>&1 &
#    #run_diffusers $tr $bs "cpu" &
#done
#run_diffusers 1 1 "mps" &
#wait


# run_diffusers $tr $bs $num_cpu_instances $num_gpu_instances
run_diffusers $tr $bs 2 2
#>> $HOSTNAME-diffusers-instance-$i.stdout  2>&1 &
echo "All done"
