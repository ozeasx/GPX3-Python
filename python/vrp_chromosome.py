#!/usr/bin/python
# ozeasx@gmail.com

import random
from collections import deque
from graph import Graph
from chromosome import Chromosome


class VRP_Chromosome(Chromosome):
    def __init__(self, tour, trucks=None):
        # Create random tour based on given dimension
        if isinstance(tour, int):
            # Assert valid truck number for tour dimension
            assert tour - 1 >= trucks, "Invalid dimension/truck relation"
            # Random tour
            self._tour = range(2, tour + 1)
            random.shuffle(self._tour)
            # Insert depot at random
            points = range(0, tour - 1)
            depots = random.sample(points, trucks)
            depots.sort()
            for i, d in enumerate(depots):
                self._tour.insert(d + i, 1)
            self._tour = tuple(self._tour)
            # Truck number
            self._trucks = trucks
        # User defined tour
        elif isinstance(tour, (list, tuple, deque)):
            # If OK, set tour and truck number
            self._tour = tuple(tour)
            self._trucks = self._tour.count(1)

        # Verify if it is a valid VRP tour
        for i, j in zip(self._tour[:-1], self._tour[1:]):
            # Check adjacent depots
            if i == j:
                assert False, "Invalid VRP tour"
        assert self._tour[0] != self._tour[-1], "Invalid VRP tour"

        # Dimension
        self._dimension = len(self._tour) - self._trucks + 1
        # Load
        self._load = None
        # undirected graph and edges representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)
        self._undirected_edges = Graph.gen_undirected_edges(self._tour)

    # Get trucks
    @property
    def trucks(self):
        return self._trucks

    # Get load
    @property
    def load(self):
        return self._load

    # Set load
    @load.setter
    def load(self, value):
        self._load = value

    # Given a VRP Chromosome, returns a TSP Chromosome
    def to_tsp(self):
        # TSP tour
        tsp_tour = list()

        # Ghost depots numbering
        ghost = self._dimension + 1

        first = False
        for i in self._tour:
            if i == 1:
                # If first depot, append 1
                if not first:
                    first = True
                    tsp_tour.append(i)
                    continue
                # Ghost depots
                else:
                    tsp_tour.append(ghost)
                    ghost += 1
            else:
                tsp_tour.append(i)

        return Chromosome(tsp_tour)

    # Given a vrp in TSP format, returns a VRP Chromosome
    @staticmethod
    def tsp2vrp(tsp, trucks):
        vrp_tour = list()
        dimension = len(tsp.tour) - trucks + 1
        ghost_depots = range(dimension + 1, dimension + trucks + 2)
        for i in tsp.tour:
            if i in ghost_depots:
                vrp_tour.append(1)
            else:
                vrp_tour.append(i)

        return VRP_Chromosome(vrp_tour)


# Test section
if __name__ == '__main__':
    vrp1 = VRP_Chromosome([2, 1, 3, 1, 4, 5, 1])

    print "VRP Tour, ", vrp1.tour
    print "VRP Trucks, ", vrp1.trucks
    print "VRP Dimension, ", vrp1.dimension

    tsp1 = VRP_Chromosome.vrp2tsp(vrp1)

    print
    print "TSP Tour, ", tsp1.tour
    print "TSP Dimension, ", tsp1.dimension
    print

    vrp2 = VRP_Chromosome.tsp2vrp(tsp1, 3)

    print
    print "VRP Tour, ", vrp2.tour
    print "VRP Trucks, ", vrp2.trucks
    print "VRP Dimension, ", vrp2.dimension
