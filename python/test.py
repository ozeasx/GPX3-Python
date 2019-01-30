#!/usr/bin/python
# ozeasx@gmail.com

from tsp import TSPLIB
from chromosome import Chromosome
from itertools import combinations
import multiprocessing
from gpx import GPX

# TSP and GPX instances
tsp = TSPLIB("../tsplib/ulysses16.tsp")
gpx = GPX(tsp)


# Create solutions combinations
def couple_formation(size, dimension, data):
    print "Creating population..."
    population = set()
    for i in xrange(size):
        c = Chromosome(dimension, data)
        while c in population:
            c = Chromosome(dimension, data)
        c.dist = data.tour_dist(c.tour)
        population.add(c)
    print "Done"
    return combinations(population, 2)


def recombine(couple):
    p1, p2, f1, f2, f3 = couple
    print p1, p2, f1, f2, f3
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
        pair = tuple(pair)
        pop.add(pair)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(recombine, pop)
    print result
    # print "Test finished..."
    # print "Partitioning ------------------------------------------------------"
    # print "\tFeasible type 1: ", gpx.counters['feasible_1']
    # print "\tFeasible type 2: ", gpx.counters['feasible_2']
    # print "\tFeasible type 3: ", gpx.counters['feasible_3']
    # print "\tInfeasible: ", gpx.counters['infeasible']
    # print "\tFusions: ", gpx.counters['fusions']
    # print "\tUnsolved: ", gpx.counters['unsolved']
    # print "\tInfeasible tour: ", gpx.counters['inf_tours']
    # print "Improved: ", len(result)-result.count(0), "/", len(result)


couples = couple_formation(10, 16, tsp)

test(couples, True, False, False)
