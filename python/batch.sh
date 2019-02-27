#!/bin/bash
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 0 -t3f 0 -g 1000 -n 30 -o ../results2/TFF_TFF &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 0 -t2 1 -t3 0 -t1f 0 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/FTF_FTF &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 0 -t2 0 -t3 1 -t1f 0 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/FFT_FFT &

nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 0 -t1f 1 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/TTF_TTF &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 1 -t1f 1 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/TFT_TFT &
nohup ./main.py -c 1 -r 0.5 -k 3 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 1 -t1f 1 -t2f 1 -t3f 1 -g 1000 -n 30 -o ../results2/TTT_TTT &

nohup ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 0 -t1f 1 -t2f 0 -t3f 0 -g 1000 -n 30 -o ../results2/TFF_TFFm &
nohup ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 0 -t2 1 -t3 0 -t1f 0 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/FTF_FTFm &
nohup ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 0 -t2 0 -t3 1 -t1f 0 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/FFT_FFTm &

nohup ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 0 -t1f 1 -t2f 1 -t3f 0 -g 1000 -n 30 -o ../results2/TTF_TTFm &
nohup ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 0 -t3 1 -t1f 1 -t2f 0 -t3f 1 -g 1000 -n 30 -o ../results2/TFT_TFTm &
nohup ./main.py -c 1 -r 0.5 -k 3 -m 0.02 ../tsplib/eil101.tsp -t1 1 -t2 1 -t3 1 -t1f 1 -t2f 1 -t3f 1 -g 1000 -n 30 -o ../results2/TTT_TTTm &
