#!/usr/bin/python
# ozeasx@gmail.com

import random
from collections import deque
from chromosome import Chromosome


# Nearest neighbour algorithm
def nn(data, method):

    # Tour
    tour = list()
    # Available nodes
    nodes = set(range(1, data.dimension + 1))
    # Choose first random node
    tour.append(random.sample(nodes, 1)[0])
    # Remove added client from available nodes
    nodes.remove(tour[-1])
    # Add nearest nodes
    dist = 0
    while nodes:
        n, d = data.get_nearest(tour[-1], nodes)
        tour.append(n)
        dist += d

    c = Chromosome(tour, dist)
    if method == 'nn2opt':
        c = two_opt(c, data)
    return c


# 2-opt adapted from
# https://en.wikipedia.org/wiki/2-opt
# https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
# http://pedrohfsd.com/2017/08/09/2opt-part1.html
# https://rawgit.com/pedrohfsd/TSP/develop/2opt.js


def two_opt(chromosome, data, limit=True):
    # Initial tour
    best_tour = list(chromosome.tour)
    # Get tour dist
    best_dist = chromosome.dist
    # Get dimension
    dimension = chromosome.dimension
    # Begin with no improvement
    improved = True
    # Tested inversions
    tested = set()
    # Stop when no improvement is made
    while improved:
        improved = False
        for i in xrange(dimension - 1):
            for j in xrange(i + 1, dimension):
                # Do not invert whole tour
                if i == 0:
                    if j-i == dimension - 1:
                        continue
                # Create edges swap in advance
                join_a = sorted([sorted([best_tour[i-1], best_tour[i]]),
                                 sorted([best_tour[j], best_tour[(j+1) %
                                                                 dimension]])])

                join_b = sorted([sorted([best_tour[i-1], best_tour[j]]),
                                 sorted([best_tour[i], best_tour[(j+1) %
                                                                 dimension]])])

                # List of lists to tuple
                join_a = tuple(v for sub in join_a for v in sub)
                join_b = tuple(v for sub in join_b for v in sub)

                # Avoid duplicated tests
                if (frozenset([join_a, join_b]) in tested
                        or join_a == join_b):
                    continue

                # Store cases to not be tested again
                tested.add(frozenset([join_a, join_b]))

                # Calc distances
                join_a_dist = data.ab_cycle_dist(join_a)
                join_b_dist = data.ab_cycle_dist(join_b)

                # Verify if swap is shorter
                if join_b_dist < join_a_dist:
                    # 2opt swap
                    new_tour = best_tour[0:i]
                    new_tour.extend(reversed(best_tour[i:j + 1]))
                    new_tour.extend(best_tour[j+1:])
                    best_tour = new_tour
                    best_dist = best_dist - join_a_dist + join_b_dist
                    improved = not limit
                    # break

    # Make sure 2opt is doing its job
    assert best_dist <= chromosome.dist, "Something wrong..."
    # Rotate solution to begin with 1
    p = best_tour.index(1)
    best_tour = deque(best_tour)
    best_tour.rotate(-p)

    # Return solution
    return Chromosome(best_tour, best_dist)
