#!/bin/bash
# ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/u1060.tsp -t1 1 -t2 0 -t3 0 -F False -g 1000 -n 30 -o ../results4/TFF
# ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/u1060.tsp -t1 0 -t2 1 -t3 0 -F False -g 1000 -n 30 -o ../results4/FTF
# ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/u1060.tsp -t1 0 -t2 0 -t3 1 -F False -g 1000 -n 30 -o ../results4/FFT
#
# ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/u1060.tsp -t1 1 -t2 1 -t3 0 -F False -g 1000 -n 30 -o ../results4/TTF
# ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/u1060.tsp -t1 1 -t2 0 -t3 1 -F False -g 1000 -n 30 -o ../results4/TFT
# ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/u1060.tsp -t1 0 -t2 1 -t3 1 -F False -g 1000 -n 30 -o ../results4/FTT

#./main.py -c 1 -r 0.5 -k 3 ../tsplib/u1060.tsp -t1 1 -t2 1 -t3 1 -F False -g 1000 -n 30 -o ../results4/TTT

#./main.py -c 1 -r 0.5 -k 3 ../tsplib/u1060.tsp -t1 1 -t2 1 -t3 1 -E False -F False -g 1000 -n 30 -o ../results4/TTTe

./main.py -c 1 -k 3 -r 0.1 ../tsplib/d493.tsp -L True -o ../results/relax/L -n 5
./main.py -c 1 -k 3 -r 0.1 ../tsplib/d493.tsp -t3 True -t3f True -o ../results/relax/t3f -n 5
