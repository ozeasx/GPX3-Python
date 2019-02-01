#!/bin/bash

nohup ./test.py -p 250 -n 30 -o ../reports/eil101.csv ../tsplib/eil101.tsp
nohup ./test.py -p 250 -n 30 -o ../reports/berlin52.csv ../tsplib/berlin52.tsp
nohup ./test.py -p 250 -n 30 -o ../reports/ulysses16.csv ../tsplib/ulysses16.tsp
