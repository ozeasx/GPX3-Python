#!/bin/bash

nohup ./test.py -p 1000 ../tsplib/ulysses16.tsp > ../reports/ulysses16.out &
nohup ./test.py -p 1000 ../tsplib/berlin52.tsp > ../reports/berlin52.out &
nohup ./test.py -p 1000 ../tsplib/eil101.tsp > ../reports/eil101.out &

nohup ./test.py -p 1000 ../tsplib/berlin52.tsp -M 2opt > ../reports/berlin52_2opt.out &
nohup ./test.py -p 1000 ../tsplib/eil101.tsp -M 2opt > ../reports/eil101_2opt.out &
