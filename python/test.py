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
import copy

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
    gpx = GPX(tsp)
    gpx.f1_test = f1
    gpx.f2_test = f2
    gpx.f3_test = f3
    c1, c2 = gpx.recombine(*pair)
    return gpx.counters


# Test
def test(population, out):
    # Multiprocessing
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    result = pool.map(recombine, combinations(population, 2))
    pool.close()
    pool.join()
    # result = list()
    # for pair in combinations(population, 2):
    #     result.append(recombine(pair))
    # Consolidate results
    for c in result:
        out['failed'] += c['failed']
        out['feasible_1'] += c['feasible_1']
        out['feasible_2'] += c['feasible_2']
        out['feasible_3'] += c['feasible_3']
        out['infeasible'] += c['infeasible']
        out['fusion'] += c['fusion']
        out['fusion_1'] += c['fusion_1']
        out['fusion_2'] += c['fusion_2']
        out['fusion_3'] += c['fusion_3']
        out['unsolved'] += c['unsolved']
        out['inf_tour'] += c['inf_tour']
        out['bad_child'] += c['bad_child']
        if c['parents_sum'] - c['children_sum'] > 0:
            out['improved'] += 1
        out['parents_sum'] += c['parents_sum']
        out['children_sum'] += c['children_sum']
    diff = float(out['parents_sum'] - out['children_sum'])
    parents = float(out['parents_sum'])
    out['parents_improvement'] += diff / parents * 100
    out['optima_improvement'] += diff / optima / len(result) * 100
    out['recombinations'] = len(result)



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
