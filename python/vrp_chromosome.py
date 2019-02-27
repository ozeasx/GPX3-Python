#!/usr/bin/python
# ozeasx@gmail.com

import random
from collections import defaultdict
from graph import Graph
from chromosome import Chromosome


class VRP_Chromosome(Chromosome):
    def __init__(self, **kwargs):
        # Create random tour based on given dimension
        if len(kwargs) == 1 and 'dimension' in kwargs:
            Chromosome.__init__(dimension=kwargs['dimension'])
            self._trucks = 1
        elif all(p in kwargs for p in ('dimension', 'trucks')):
            # Assert valid truck number for tour dimension
            assert kwargs['trucks'] < kwargs['dimension']
            # Random tour
            self._tour = range(2, kwargs['dimension'] + 1)
            random.shuffle(self._tour)
            # Insert first depot
            self._tour.insert(0, 1)
            # Insert depot at random
            points = range(2, kwargs['dimension'])  # Insert positions
            # Random insert positions
            depots = random.sample(points, kwargs['trucks'] - 1)
            depots.sort()
            for index, depot in enumerate(depots):
                self._tour.insert(depot + index, 1)
            self._tour = tuple(self._tour)
            # Truck number
            self._trucks = kwargs['trucks']
            # Dist
            self._dist = None
        # User defined tour
        elif all(p in kwargs for p in ('tour', 'dist')):
            # If OK, set tour and truck number
            self._dist = kwargs['dist']
            self._tour = tuple(kwargs[''])
            self._trucks = self._tour.count(1)
        else:
            print "Dimension or tour needed"
            exit()

        # VRP solutions assertions
        if self._trucks > 1:
            for i, j in zip(self._tour[:-1], self._tour[1:]):
                # Check adjacent depots
                if i == j:
                    assert False, "Invalid VRP tour"
            assert self._tour[0] != self._tour[-1], "Invalid VRP tour"
            assert self._tour.count(1) == self._trucks, "Invalid VRP tour"
            assert len(self._tour) == len(set(self._tour)) + self._trucks - 1

            # Store routes
            self._routes = self._tour
            if self._trucks > 1:
                self._routes = defaultdict(list)
                aux = 0
                for index, client in enumerate(self._tour):
                    if index is not 0 and client == 1:
                        aux += 1
                    self._routes[aux].append(client)

        # Dimension
        self._dimension = len(self._tour) - self._trucks + 1
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

    # Get petals
    @property
    def routes(self):
        return self._routes

    # Set load
    @load.setter
    def load(self, value):
        self._load = value

    # Get best known tour
    @property
    def best_solution(self):
        return self._best_solution

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
                    tsp_tour.append(1)
                    continue
                # Ghost depots
                else:
                    tsp_tour.append(ghost)
                    ghost += 1
            else:
                tsp_tour.append(i)

        assert len(tsp_tour) == len(set(tsp_tour)), tsp_tour

        # return VRP_Chromosome(tsp_tour)
        return VRP_Chromosome(tsp_tour, 1, self._dist)

    # Convert back to a VRP Chromosome
    def to_vrp(self, dimension):
        vrp_tour = list()
        ghost_depots = dimension + 1
        for i in self.tour:
            if i >= ghost_depots:
                vrp_tour.append(1)
            else:
                vrp_tour.append(i)

        # return VRP_Chromosome(vrp_tour)
        return VRP_Chromosome(vrp_tour, 1, self._dist)


# Test section
if __name__ == '__main__':
    vrp1 = VRP_Chromosome(4, 3)

    print "VRP Tour, ", vrp1.tour
    print "VRP Trucks, ", vrp1.trucks
    print "VRP Dimension, ", vrp1.dimension

    tsp1 = vrp1.to_tsp()

    print
    print "TSP Tour, ", tsp1.tour
    print "TSP Dimension, ", tsp1.dimension
    print

    vrp2 = tsp1.to_vrp(vrp1.dimension)

    print
    print "VRP Tour, ", vrp2.tour
    print "VRP Trucks, ", vrp2.trucks
    print "VRP Dimension, ", vrp2.dimension
