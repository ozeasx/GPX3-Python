#!/usr/bin/python
# ozeasx@gmail.com

import sys
from chromosome import Chromosome
from tsp import TSPLIB
from gpx import GPX

counter = 1
couples = set()


def newcouple(size):
    def unique():
        c1 = Chromosome(size)
        c2 = Chromosome(size)
        while c1 == c2:
            c2 = Chromosome(size)
        return frozenset([c1, c2])

    couple = unique()
    while couple in couples:
        couple = unique()
    couples.add(couple)

    c1, c2 = couple
    c1.dist = tsp.tour_dist(c1.tour)
    c2.dist = tsp.tour_dist(c2.tour)
    return c1, c2


tsp = TSPLIB("../tsplib/berlin52.tsp")
size_limit = 8388840  # Notebook
# size_limit = 16777448  # Desktop

while sys.getsizeof(couples) < size_limit:
    print "\rtesting %i" % counter,  sys.getsizeof(couples), size_limit,
    gpx = GPX(tsp)
    gpx.test_1 = True
    gpx.test_2 = True
    gpx.test_3 = True
    gpx.fusion_on = False
    p1, p2 = newcouple(10)
    gpx.recombine(p1, p2)
    cond1 = gpx.counters['feasible']
    cond2 = gpx.counters['inf_tour']
    cond3 = gpx.counters['infeasible']
    if cond1 and cond2 and not cond3:
        if all(len(gpx.info['simple_a'][i]['in']) <= 3
                for i in gpx.info['feasible'][0]):
            print p1.tour, p2.tour
            break
    counter += 1
