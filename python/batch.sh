#!/bin/bash

date

nohup ./test.py -p 1000 -n 30 -o ../reports/ulysses16.csv ../tsplib/ulysses16.tsp
nohup ./test.py -p 1000 -n 30 -o ../reports/berlin52.csv ../tsplib/berlin52.tsp
nohup ./test.py -p 1000 -n 30 -o ../reports/eil101.csv ../tsplib/eil101.tsp

nohup ./test.py -p 1000 -n 30 -o ../reports/berlin52_2opt.csv ../tsplib/berlin52.tsp -M 2opt
nohup ./test.py -p 1000 -n 30 -o ../reports/eil101_2opt.csv ../tsplib/eil101.tsp -M 2opt

date
