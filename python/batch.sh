#!/bin/bash
./main.py -c 1 -r 0.5 -k 3 ../tsplib/ulysses16.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 0 -t3f 0 -g 100 -n 5 -o ../results3/TFF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/ulysses16.tsp -t1 0 -t2 1 -t3 0 -t1f 0 -t2f 1 -t3f 0 -g 100 -n 5 -o ../results3/FTF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/ulysses16.tsp -t1 0 -t2 0 -t3 1 -t1f 0 -t2f 0 -t3f 1 -g 100 -n 5 -o ../results3/FFT

./main.py -c 1 -r 0.5 -k 3 ../tsplib/ulysses16.tsp -t1 1 -t2 1 -t3 0 -t1f 1 -t2f 1 -t3f 0 -g 100 -n 5 -o ../results3/TTF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/ulysses16.tsp -t1 1 -t2 0 -t3 1 -t1f 1 -t2f 0 -t3f 1 -g 100 -n 5 -o ../results3/TFT
./main.py -c 1 -r 0.5 -k 3 ../tsplib/ulysses16.tsp -t1 0 -t2 1 -t3 1 -t1f 0 -t2f 1 -t3f 1 -g 100 -n 5 -o ../results3/FTT

./main.py -c 1 -r 0.5 -k 3 ../tsplib/ulysses16.tsp -t1 1 -t2 1 -t3 1 -t1f 1 -t2f 1 -t3f 1 -g 100 -n 5 -o ../results3/TTT
