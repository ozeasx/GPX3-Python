#!/bin/bash

./main.py -c 1 -k 3 -e 1 -g 500 -n 5 -o ../results/c1k3e1/ ../tsplib/eil101.tsp
./main.py -c 1 -r 0.8 -k 3 -e 1 -g 500 -n 5 -o ../results/c1r08k3e1/ ../tsplib/eil101.tsp
./main.py -c 1 -m 0.02 -k 3 -e 1 -g 500 -n 5 -o ../results/c1m002k3e1/ ../tsplib/eil101.tsp
./main.py -c 1 -m 0.02 -r 0.8 -k 3 -e 1 -g 500 -n 5 -o ../results/c1m002r08k3e1/ ../tsplib/eil101.tsp
./main.py -c 1 -m 0.02 -r 0.8 -e 1 -g 500 -n 5 -o ../results/c1m002r08e1/ ../tsplib/eil101.tsp

./main.py -c 1 -k 3 -r 0.8 -f3 True -g 500 -n 5 -o ../results/f1f2f3/ ../tsplib/eil101.tsp
./main.py -c 1 -k 3 -r 0.8 -f2 False -f3 True -g 500 -n 5 -o ../results/f1f3/ ../tsplib/eil101.tsp
./main.py -c 1 -k 3 -r 0.8 -f2 False -g 500 -n 5 -o ../results/f1/ ../tsplib/eil101.tsp
./main.py -c 1 -k 3 -r 0.8 -f1 False -f3 True -g 500 -n 5 -o ../results/f2f3/ ../tsplib/eil101.tsp
./main.py -c 1 -k 3 -r 0.8 -f1 False -f2 False -f3 True -g 500 -n 5 -o ../results/f3/ ../tsplib/eil101.tsp
