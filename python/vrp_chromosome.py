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
            # Dimension
            self._dimension = tour
            # Truck number
            self._trucks = trucks
            # Random tour
            self._tour = range(2, tour + 1)
            random.shuffle(self._tour)
            # Insert depot
            points = range(0, tour - 1)
            depots = random.sample(points, trucks)
            depots.sort()
            for i, v in enumerate(depots):
                self._tour.insert(v + i, 1)
            self._tour = tuple(self._tour)
        # User defined tour
        elif isinstance(tour, (list, tuple, deque)):
            # Verify if it is a valid VRP tour
            depots = list()
            for index, node in enumerate(tour):
                if node == 1:
                    depots.append(index)
            for i, j in zip(depots[:-1], depots[1:]):
                if j - i == 1:
                    assert True, "Invalid VRP tour"
            # If OK, set tour and truck number
            self._tour = tuple(tour)
            self._trucks = len(depots)

        # undirected graph and edges representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)
        self._undirected_edges = Graph.gen_undirected_edges(self._tour)

        # Load
        self._load = None

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


# Test section
if __name__ == '__main__':
    p1 = VRP_chromosome(4, 2)
    print p1.tour
