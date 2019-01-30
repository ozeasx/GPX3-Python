#!/usr/bin/python
# ozeasx@gmail.com

import os
import argparse
from tsp import TSPLIB
from chromosome import Chromosome
from itertools import combinations
from collections import defaultdict
import multiprocessing
from gpx import GPX
import functions

# Argument parser
p = argparse.ArgumentParser(description="Tester")
p.add_argument("I", help="TSP instance file", type=str)
p.add_argument("-M", choices=['random', '2opt'], default='random',
               help="Method to generate inicial population")
p.add_argument("-p", help="Population size", type=int, default=100)
# Parser
args = p.parse_args()
# Assert instance file
assert os.path.isfile(args.I), "File " + args.I + " doesn't exist"

# TSP and GPX instances
tsp = TSPLIB(args.I)


# Create solutions combinations
def couple_formation(size, dimension, data, method='random'):
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
    return set(combinations(population, 2))


def recombine(couple):
    p1, p2, f1, f2, f3 = couple
    gpx = GPX(tsp)
    gpx.f1_test = f1
    gpx.f2_test = f2
    gpx.f3_test = f3
    c1, c2 = gpx.recombine(p1, p2)
    return gpx.counters


# Test
def test(couples, f1, f2, f3):
    print "Test started..."
    pop = set()
    for pair in couples:
        pair = list(pair)
        pair.extend([f1, f2, f3])
        pop.add(tuple(pair))
    # Multiprocessing
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(recombine, pop)
    # Consolidate results
    counter = defaultdict(int)
    for c in result:
        counter['feasible_1'] += c['feasible_1']
        counter['feasible_2'] += c['feasible_2']
        counter['feasible_3'] += c['feasible_3']
        counter['infeasible'] += c['infeasible']
        counter['fusions'] += c['fusions']
        counter['fusions_1'] += c['fusions_1']
        counter['fusions_2'] += c['fusions_2']
        counter['fusions_3'] += c['fusions_3']
        counter['unsolved'] += c['unsolved']
        counter['inf_tours'] += c['inf_tours']
        counter['parents_sum'] += c['parents_sum']
        counter['children_sum'] += c['children_sum']
        if c['parents_sum'] - c['children_sum'] > 0:
            counter['improved'] += 1
    print "Test finished with dataset ", args.I
    print "Partitioning ------------------------------------------------------"
    print "\tFeasible 1: ", counter['feasible_1']
    print "\tFeasible 2: ", counter['feasible_2']
    print "\tFeasible 3: ", counter['feasible_3']
    print "\tInfeasible: ", counter['infeasible']
    print "\tFusions: ", counter['fusions']
    print "\tFusions 1: ", counter['fusions_1']
    print "\tFusions 2: ", counter['fusions_2']
    print "\tFusions 3: ", counter['fusions_3']
    print "\tUnsolved: ", counter['unsolved']
    print "\tInfeasible tour: ", counter['inf_tours']
    print "Improved: ", counter['improved'], "/", len(result)
    diff = counter['parents_sum'] - counter['children_sum']
    parents = counter['parents_sum']
    best = tsp.best_solution.dist
    print "Total improvement/parents: ", diff / parents * 100, "%"
    print "Total improvement/optima: ", diff / len(result) / best * 100, "%"


couples = couple_formation(args.p, tsp.dimension, tsp, args.M)

test(couples, True, False, False)
test(couples, True, True, False)
test(couples, True, False, True)
test(couples, True, True, True)
