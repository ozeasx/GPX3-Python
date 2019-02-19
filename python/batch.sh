#!/bin/bash
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 0 -t3f 0 -g 1000 -n 30 -o ../results/TFFTFF/ &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results/TFFTTF/ &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results/TFFTFT/ &

nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 0 -t1f 1 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results/TTFTTF/ &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 1 -t1f 1 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results/TFTTFT/ &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 1 -t1f 1 -t2f 1 -t3f 1 -g 1000 -n 30 -o ../results/TTTTTT/ &
