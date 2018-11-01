#!/usr/bin/python
# ozeasx@gmail.com

from collections import deque
from chromosome import Chromosome

# 2-opt adapted from
# https://en.wikipedia.org/wiki/2-opt
# https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
# http://pedrohfsd.com/2017/08/09/2opt-part1.html
# https://rawgit.com/pedrohfsd/TSP/develop/2opt.js


def two_opt(chromosome, data):
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
                    improved = True

    # Rotate solution to begin with 1
    # assert len(set(best_tour)) == data.dimension
    p = best_tour.index(1)
    best_tour = deque(best_tour)
    best_tour.rotate(-p)

    # Return solution
    return Chromosome(best_tour, best_dist)
