#!/usr/bin/python
# ozeasx@gmail.com

from vrp_chromosome import VRP_Chromosome as Chromosome
from itertools import combinations


# Nearest neighbour algorithm
def nearest_neighbor(data):
    tour = list()
    visited = set()
    over_capacity = set()
    nodes = set(range(2, data.dimension + 1))

    # Return nearest node from i
    def next(i):
        last_dist = int("inf")
        nearest = None
        dist = None
        for j in nodes - visited - over_capacity:
            dist = data.dist(sorted([i, j]))
            if dist < last_dist:
                nearest = j
                last_dist = dist
        return nearest, dist


    for i in range(data.trucks):
        # append depot
        tour.append(1)
        demand = 0
        while next(last)
            test = next(last)
            demand +=
            tour.append()




# Run 2opt over vrp solution
def vrp_2opt(chromosome, data):
    new_tour = list()
    dist = 0

    for key, route in chromosome.routes.items():
        t, d = two_opt(route, data.tour_dist(route), data)
        new_tour.extend(t)
        dist += d

    return Chromosome(new_tour, None, dist)


# 2-opt adapted from
# https://en.wikipedia.org/wiki/2-opt
# https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
# http://pedrohfsd.com/2017/08/09/2opt-part1.html
# https://rawgit.com/pedrohfsd/TSP/develop/2opt.js ""
# Initial tour
def two_opt(tour, dist, data):
    # Inicial tour
    best_tour = list(tour)
    # Get tour dist
    best_dist = dist
    # Get dimension
    dimension = len(tour)
    # Begin with no improvement
    improved = True
    # Tested inversions
    tested = set()
    # Stop when no improvement is made
    while improved:
        improved = False
        for i in xrange(1, dimension - 1):
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

    # Make sure 2opt is doing its job
    assert best_dist <= dist, "Something wrong..."
    # Return solution
    return best_tour, best_dist
