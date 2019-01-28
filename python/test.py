#!/usr/bin/python
# ozeasx@gmail.com

from tsp import TSPLIB
from chromosome import Chromosome
from itertools import combinations
import multiprocessing
from gpx import GPX

# Snipet code to test a lot of random cases
tsp = TSPLIB("../tsplib/berlin52.tsp")
gpx = GPX(tsp)

# p1 = Chromosome(16)
# p1.dist = tsp.tour_dist(p1.tour)

# r1 = mut.two_opt(p1, tsp)

# print p1.tour
# print p1.dist
# print r1.tour
# print r1.dist

gpx.f1_test = True
gpx.f2_test = True
gpx.f3_test = False


def couple_formation(q, dimension, data):
    print "Creating population..."
    population = set()
    for i in xrange(q):
        c = Chromosome(dimension, data)
        while c in population:
            c = Chromosome(dimension, data)
        c.dist = data.tour_dist(c.tour)
        population.add(c)
    print "Done"
    print "Creating couples..."
    couples = set()
    for pair in combinations(population, 2):
        couples.add(pair)
    print "Done..."
    return couples


def recombine(couple):
    p1, p2 = tuple(couple)
    c1, c2 = gpx.recombine(p1, p2)
    return (p1.dist + p2.dist) - (c1.dist + c2.dist)


def test(pop, dimension, data):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    couples = couple_formation(pop, dimension, data)
    print "Recombinations started..."
    result = pool.map(recombine, couples)
    print "Improved: ", len(result)-result.count(0), "/", len(result)


test(1000, 10, tsp)
