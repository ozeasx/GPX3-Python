#!/usr/bin/python
# ozeasx@gmail.com

import copy
import random
from vrp_chromosome import VRP_Chromosome as Chromosome


# 2opt switch
def two_opt(c, data, limit=True):
    if data.type == 'tsp':
        tour, dist = two_opt_tsp(c.tour, c.dist, data, limit)
    elif data.type == 'vrp':
        tour, dist = two_opt_vrp(c.tour, c.dist, data, limit)
    return Chromosome(tour, dist)


# nn switch
def nn(data, method):
    if data.type == 'tsp':
        tour, dist = nn_tsp(data, method)
    elif data.type == 'vrp':
        tour, dist = nn_vrp(data, method)
    return Chromosome(tour, dist)


# 2-opt adapted from
# https://en.wikipedia.org/wiki/2-opt
# https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
# http://pedrohfsd.com/2017/08/09/2opt-part1.html
# https://rawgit.com/pedrohfsd/TSP/develop/2opt.js
def two_opt_tsp(tour, dist, data, limit=False):
    # Inicial tour
    best_tour = list(tour)

    # Distance
    best_dist = dist

    # Get dimension
    dimension = len(best_tour)

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
                if i == 0 and j == dimension - 1:
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
                if frozenset([join_a, join_b]) in tested or join_a == join_b:
                    continue

                # Store cases to not be tested again
                tested.add(frozenset([join_a, join_b]))

                # Calc distances
                join_a_dist = data.ab_dist(join_a)
                join_b_dist = data.ab_dist(join_b)

                # Verify if swap is shorter
                if join_b_dist < join_a_dist:
                    # 2opt swap
                    new_tour = best_tour[0:i]
                    new_tour.extend(reversed(best_tour[i:j + 1]))
                    new_tour.extend(best_tour[j+1:])
                    best_tour = new_tour
                    best_dist = best_dist - join_a_dist + join_b_dist
                    improved = not limit

    # Make sure 2opt is doing its job
    assert best_dist <= dist, "Something wrong..."

    # Return solution
    return best_tour, best_dist


# Nearest neighbour algorithm
def nn_tsp(data, method):
    # Tour
    tour = list()
    # Available nodes
    nodes = set(range(1, data.dimension + 1))
    # Choose first random node
    tour.append(random.sample(nodes, 1)[0])
    # Remove added client from available nodes
    nodes.remove(tour[-1])
    # Initialize distance
    dist = 0
    # Add nearest nodes
    while nodes:
        nearest, d = data.get_nearest(tour[-1], nodes)
        # Append tour
        tour.append(nearest)
        # Update distance
        dist += d
        # Remove added client from available nodes
        nodes.remove(tour[-1])

    # NN + 2opt
    if method == 'nn2opt':
        tour, dist = two_opt(tour, dist, data)

    # Return created solution
    return tour, dist


# Run 2opt over vrp solution
def two_opt_vrp(c, data, limit):
    tour = list()
    dist = 0

    for key, route in c.routes.items():
        t, d = two_opt_tsp(route, data.tour_dist(route), data, limit)
        tour.extend(t)
        dist += d

    return tour, dist


# Nearest neighbour algorithm
def nn_vrp(data, method):

    # Tour and available nodes
    tour = list()
    nodes = set(range(2, data.dimension + 1))

    # Create a route for each truck
    for i in range(data.trucks):
        # append depot
        tour.append(1)
        # append random truck route client
        tour.append(random.sample(nodes, 1)[0])
        # Remove added client from nodes
        nodes.remove(tour[-1])
        # Initialize route demand
        demand = data.demand(tour[-1])
        # Nodes that exceed truck capacity
        over_capacity = set()
        # Add nearest nodes
        while nodes:
            test = data.get_nearest(tour[-1], nodes - over_capacity)
            # Test if are nodes available
            if test is None:
                break
            # Test candidate demand
            elif demand + data.demand(test) <= data.capacity:
                tour.append(test)
                nodes.remove(test)
                demand += data.demand(test)
            # Update nodes that exceed capacity
            else:
                over_capacity.add(test)

    # Return only a complete solution
    if len(nodes) == 0:
        dist = data.tour_dist(tour)
        if method == "nn2opt":
            tour, dist = two_opt_vrp(tour, dist, data)
        # Return created solution
        return tour, dist
    else:
        return None


# Fix vrp solutions
def fix(c, data):
    routes = copy.deepcopy(c.routes)
    demand = copy.deepcopy(c.load)
    extra_demand = set()

    # Remove clients which extrapolate truck capacity
    for key in c.load:
        if c.load[key] > data.capacity:
            demand[key] = 0
            for node in c.routes[key]:
                demand[key] += data.demand(node)
                if demand[key] > data.capacity:
                    routes[key].remove(node)
                    demand[key] -= data.demand(node)
                    extra_demand.add(node)

    # Insert extra demand nodes
    limit = len(extra_demand) * 2
    while limit and extra_demand:
        limit -= 1
        for key in routes:
            if len(extra_demand) == 0:
                break
            test = next(iter(extra_demand))
            # test = sorted(extra_demand)[-1]
            if demand[key] + data.demand(test) <= data.capacity:
                best = float("inf")
                for i, node in enumerate(routes[key]):
                    r = [routes[key][i-1]] + [test] + [routes[key][i]]
                    dist = data.route_dist(r)
                    if dist < best:
                        best = dist
                        index = i
                if index == 0:
                    index = -1
                routes[key].insert(index, test)
                demand[key] += data.demand(test)
                extra_demand.remove(test)

    # Insert remaining nodes
    if len(extra_demand):
        while extra_demand:
            for key in routes:
                routes[key].append(extra_demand.pop())
                if len(extra_demand) == 0:
                    break

    # Build and return new tour
    tour = list()
    for key in routes:
        tour.extend(routes[key])

    return Chromosome(tour, None, data.tour_dist(tour))
