#!/bin/bash

for ((i = 4; i < 6; i++)); do
    taskset -c $i python main.py &
done 
