#!/bin/bash

./main.py -c 1 -m 0.02 -k 3 -e 3 -r 0.8 -f3 True -F a -g 500 -n 5 -o ../result/vrpA/ ../cvrp/F-n45-k4.vrp
./main.py -c 1 -m 0.02 -k 3 -e 3 -r 0.8 -f3 True -F b -g 500 -n 5 -o ../result/vrpB/ ../cvrp/F-n45-k4.vrp
./main.py -c 1 -m 0.02 -k 3 -e 3 -r 0.8 -f3 True -F c -g 500 -n 5 -o ../result/vrpC/ ../cvrp/F-n45-k4.vrp
./main.py -c 1 -m 0.02 -k 3 -e 3 -r 0.8 -f3 True -F d -g 500 -n 5 -o ../result/vrpD/ ../cvrp/F-n45-k4.vrp
./main.py -c 1 -m 0.02 -k 3 -e 3 -r 0.8 -f3 True -F e -g 500 -n 5 -o ../result/vrpE/ ../cvrp/F-n45-k4.vrp
