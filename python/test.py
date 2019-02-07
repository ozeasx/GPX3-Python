#!/usr/bin/python
# ozeasx@gmail.com

import time
import os
import sys
import argparse
from tsp import TSPLIB
from chromosome import Chromosome
from itertools import combinations
from collections import defaultdict
import multiprocessing
from gpx import GPX
import functions
import csv
import numpy as np


# Argument parser
p = argparse.ArgumentParser(description="Tester")
p.add_argument("I", help="TSP instance file", type=str)
p.add_argument("-M", choices=['random', '2opt'], default='random',
               help="Method to generate inicial population")
p.add_argument("-p", help="Population size", type=int, default=100)
p.add_argument("-n", help="Number of iterations",
               type=int, default=1)
p.add_argument("-o", help="Results output csv file", type=str)
# Parser
args = p.parse_args()

#  Arguments assertions
assert os.path.isfile(args.I), "File " + args.I + " doesn't exist"
assert 0 < args.n <= 100, "Invalid iteration limit [0,100]"

# TSP and GPX instances
tsp = TSPLIB(args.I)
optima = tsp.best_solution.dist

# To ensure unique populations across tests
all_pop = set()


# Create solutions combinations
def gen_pop(size, dimension, data, method='random'):
    print "Creating population..."
    population = set()
    if method == 'random':
        # Populate with unique individuals
        for i in xrange(size):
            c = Chromosome(dimension, data)
            # Avoid duplicated
            while c in all_pop:
                c = Chromosome(dimension, data)
            # Calc dist
            c.dist = data.tour_dist(c.tour)
            population.add(c)
            all_pop.add(c)
    # Generate with 2opt (hard for small tsp)
    elif method == '2opt':
        for i in xrange(size):
            c = Chromosome(dimension, data)
            c.dist = data.tour_dist(c.tour)
            c = functions.two_opt(c, data)
            # Avoid duplicated
            while c in all_pop:
                c = Chromosome(dimension, data)
                c.dist = data.tour_dist(c.tour)
                c = functions.two_opt(c, data)
            population.add(c)
            all_pop.add(c)
    print "Done"
    # Return population
    return population


def recombine(pair):
    gpx = GPX(tsp)
    gpx.f1_test = f1
    gpx.f2_test = True
    gpx.f3_test = False
    gpx.ff1_test = False
    c1, c2 = gpx.recombine(*pair)
    return gpx.counters


# Test
def test(population, stats):
    # Multiprocessing
    # pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    # result = pool.map(recombine, combinations(population, 2))
    # pool.close()
    # pool.join()
    result = list()
    for pair in combinations(population, 2):
        result.append(recombine(pair))
    # Create keys and list indexes
    stats['feasible_1'].append(0)
    stats['feasible_2'].append(0)
    stats['feasible_3'].append(0)
    stats['infeasible'].append(0)
    stats['fusion'].append(0)
    stats['fusion_1'].append(0)
    stats['fusion_2'].append(0)
    stats['fusion_3'].append(0)
    stats['unsolved'].append(0)
    stats['bad_child'].append(0)
    stats['inf_tour'].append(0)
    stats['parents_dist'].append(0)
    stats['children_dist'].append(0)
    stats['improved'].append(0)
    stats['parents_improvement'].append(0)
    stats['failed'].append(0)
    stats['recombinations'].append(0)
    stats['total_time'].append(0)

    # Consolidate results
    for gpx_counter in result:
        for key in gpx_counter:
            stats[key][-1] += gpx_counter[key]
            # Count improvements
        if gpx_counter['children_dist'] < gpx_counter['parents_dist']:
            stats['improved'][-1] += 1
    # Calc improvement
    stats['parents_improvement'][-1] += (1 - stats['children_dist'][-1]
                                         / stats['parents_dist'][-1])
    # Store total recombinations
    stats['recombinations'][-1] += len(result)
    # Return results
    print stats
    return stats


# Counter
counter = dict()

# Tests arrange
conf = [(True, False, False), (True, True, False),
        (True, False, True), (True, True, True)]

# Execution
for n in xrange(args.n):

    print "Test ", n+1, "/", args.n, "started.."
    # Outter time
    n_time = time.time()
    # Same unique population across tests
    population = gen_pop(args.p, tsp.dimension, tsp, args.M)
    for i, f in enumerate(conf):
        # Set rules
        f1, f2, f3 = f
        # Inner time
        i_time = time.time()
        if n == 0:
            counter[i] = defaultdict(list)
        test(population, counter[i])
        # Inner time
        counter[i]['total_time'][-1] = time.time() - i_time

    # Outter time
    print "Done in ",  time.time() - n_time


# Calc avg, var and std
print "Consolidanting results"
result = dict()
for key in counter:
    result[key] = defaultdict(int)
    for field in counter[key]:
        result[key][field + '_mean'] = np.mean(counter[key][field])
        # result[key][field + '_var'] = np.var(counter[key][field])
        # result[key][field + '_std'] = np.std(counter[key][field])

# Write data to csv file and stdout
if args.o:
    with open(args.o, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(result[0].keys())
        for key in result:
            writer.writerow(result[key].values())
    print "Results wrote to ", args.o
# Write to stdout
else:
    writer = csv.writer(sys.stdout)
    writer.writerow(result[0].keys())
    for key in result:
        writer.writerow(result[key].values())
