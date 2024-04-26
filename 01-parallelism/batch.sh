#!/bin/bash

for ((i = 0; i < 24; i++)); do
    taskset -c $i python main.py &
done 
