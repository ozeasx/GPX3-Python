#!/bin/bash
./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 0 -t3f 0 -g 1000 -n 30 -o ../results2/TFF_TFF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 0 -t2 1 -t3 0 -t1f 0 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/FTF_FTF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 0 -t2 0 -t3 1 -t1f 0 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/FFT_FFT

./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 0 -t1f 1 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/TTF_TTF
./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 1 -t1f 1 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/TFT_TFT
./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 1 -t1f 1 -t2f 1 -t3f 1 -g 1000 -n 30 -o ../results2/TTT_TTT

./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 0 -t3f 0 -g 1000 -n 30 -o ../results2/TFF_TFFm
./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 0 -t2 1 -t3 0 -t1f 0 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/FTF_FTFm
./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 0 -t2 0 -t3 1 -t1f 0 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/FFT_FFTm

./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 0 -t1f 1 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/TTF_TTFm
./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 1 -t1f 1 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/TFT_TFTm
./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 1 -t1f 1 -t2f 1 -t3f 1 -g 1000 -n 30 -o ../results2/TTT_TTTm
