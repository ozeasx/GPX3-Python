#!/bin/bash
./main.py -c 1 -r 0.5 -k 3 ../tsplib/vm1084.tsp -t1 1 -t2 0 -t3 0 -F False -g 1000 -n 30 -o ../results3/TFF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/vm1084.tsp -t1 0 -t2 1 -t3 0 -F False -g 1000 -n 30 -o ../results3/FTF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/vm1084.tsp -t1 0 -t2 0 -t3 1 -F False -g 1000 -n 30 -o ../results3/FFT

./main.py -c 1 -r 0.5 -k 3 ../tsplib/vm1084.tsp -t1 1 -t2 1 -t3 0 -F False -g 1000 -n 30 -o ../results3/TTF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/vm1084.tsp -t1 1 -t2 0 -t3 1 -F False -g 1000 -n 30 -o ../results3/TFT
./main.py -c 1 -r 0.5 -k 3 ../tsplib/vm1084.tsp -t1 0 -t2 1 -t3 1 -F False -g 1000 -n 30 -o ../results3/FTT

./main.py -c 1 -r 0.5 -k 3 ../tsplib/vm1084.tsp -t1 1 -t2 1 -t3 1 -F False -g 1000 -n 30 -o ../results3/TTT
