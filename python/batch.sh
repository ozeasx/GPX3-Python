#!/bin/bash

./main.py -c 1 -M nn -R 0.5 -m 0.05 -P True -e 2 -r 0.5 -f3 True -i True -n 5 -o ../result2/vrpA/ ../cvrp/B-n31-k5.vrp &
./main.py -c 1 -M nn -R 0.5 -m 0.05 -P True -e 2 -r 0.5 -f3 True -i True -n 5 -o ../result2/vrpB/ ../cvrp/B-n34-k5.vrp &
./main.py -c 1 -M nn -R 0.5 -m 0.05 -P True -e 2 -r 0.5 -f3 True -i True -n 5 -o ../result2/vrpC/ ../cvrp/B-n35-k5.vrp &
./main.py -c 1 -M nn -R 0.5 -m 0.05 -P True -e 2 -r 0.5 -f3 True -i True -n 5 -o ../result2/vrpD/ ../cvrp/B-n38-k6.vrp &
./main.py -c 1 -M nn -R 0.5 -m 0.05 -P True -e 2 -r 0.5 -f3 True -i True -n 5 -o ../result2/vrpE/ ../cvrp/B-n41-k6.vrp &
