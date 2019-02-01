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

all_pop = set()


# Create solutions combinations
def gen_pop(size, dimension, data, method='random'):
    print "Creating population..."
    population = set()
    if method == 'random':
        # Populate with unique individuals
        while len(population) < size:
            population.add(Chromosome(dimension, data))
        # Calc distances
        for c in population:
            c.dist = data.tour_dist(c.tour)
    # Generate with 2opt
    elif method == '2opt':
        while len(population) < size:
            c = Chromosome(dimension, data)
            c.dist = data.tour_dist(c.tour)
            population.add(functions.two_opt(c, data))
    print "Done"
    return population


def recombine(pair):
    gpx = GPX(tsp, f1, f2, f3)
    c1, c2 = gpx.recombine(*pair)
    return gpx.counters


# Test
def test(population):
    # To store statistics
    stats = dict()
    # Multiprocessing
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    result = pool.map(recombine, combinations(population, 2))
    pool.close()
    pool.join()
    # Consolidate results
    for counter in result:
        stats['feasible_1'] += counter['feasible_1']
        stats['feasible_2'] += counter['feasible_2']
        stats['feasible_3'] += counter['feasible_3']
        stats['infeasible'] += counter['infeasible']
        stats['fusion_1'] += counter['fusion_1']
        stats['fusion_2'] += counter['fusion_2']
        stats['fusion_3'] += counter['fusion_3']
        stats['unsolved'] += counter['unsolved']
        stats['inf_tour'] += counter['inf_tour']
        stats['bad_child'] += counter['bad_child']
        stats['failed'] += counter['failed']
        stats['parents_dist'] += counter['parents_dist']
        stats['children_dist'] += counter['children_dist']
        if stats['children_dist'] < stats['parents_dist']:
            stats['improved'] += 1
    # Calc improvement
    parents = float(stats['parents_dist'])
    children = float(stats['children_dist'])

    stats['parents_improvement'] += 1 - parents / children
    # Return results
    return stats


# Counter
counter = dict()
for i in xrange(4):
    counter[i] = defaultdict(int)

# Tests arrange
tests = [(True, False, False), (True, True, False),
         (True, False, True), (True, True, True)]

# Execution
for n in xrange(args.n):

    start_time = time.time()

    population = gen_pop(args.p, tsp.dimension, tsp, args.M)

    print "Test ", n+1, "/", args.n, "started.."
    for i, f in enumerate(tests):
        f1, f2, f3 = f
        test(population, counter[i])
    print "Done in ", time.time() - start_time


# Calc averages
print "Consolidanting results"
for key in counter:
    for field in counter[key]:
        counter[key][field] /= float(args.n)

# Write data to csv file and stdout
if args.o:
    with open(args.o, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(counter[0].keys())
        for key in counter:
            writer.writerow(counter[key].values())
    print "Results wrote to ", args.o
# Write to stdout
else:
    writer = csv.writer(sys.stdout)
    writer.writerow(counter[0].keys())
    for key in counter:
        writer.writerow(counter[key].values())
