#!/usr/bin/python
# ozeasx@gmail.com

import random
from collections import deque, defaultdict
from graph import Graph
from chromosome import Chromosome


class VRP_Chromosome(Chromosome):
    def __init__(self, tour, trucks=None, dist=None):
        # Create random tour based on given dimension
        if isinstance(tour, int):
            # Assert valid truck number for tour dimension
            assert tour - 1 >= trucks, "Invalid dimension/truck relation"
            # Random tour
            self._tour = range(2, tour + 1)
            random.shuffle(self._tour)
            # Insert first depot
            self._tour.insert(0, 1)
            # Insert depot at random
            points = range(2, tour)  # Insert positions
            depots = random.sample(points, trucks-1)  # Random insert positions
            depots.sort()
            for index, depot in enumerate(depots):
                self._tour.insert(depot + index, 1)
            self._tour = tuple(self._tour)
            # Truck number
            self._trucks = trucks
            # Dist
            self._dist = None
        # User defined tour
        elif isinstance(tour, (list, tuple, deque)):
            # If OK, set tour and truck number
            self._dist = dist
            self._tour = tuple(tour)
            self._trucks = self._tour.count(1)

        # Verify if it is a valid VRP tour
        for i, j in zip(self._tour[:-1], self._tour[1:]):
            # Check adjacent depots
            if i == j:
                assert False, "Invalid VRP tour"
        assert self._tour[0] != self._tour[-1], "Invalid VRP tour"
        assert self._tour.count(1) == self._trucks, "Invalid VRP tour"
        assert len(self._tour) == len(set(self._tour)) + self._trucks - 1

        # Dimension
        self._dimension = len(self._tour) - self._trucks + 1
        # undirected graph and edges representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)
        self._undirected_edges = Graph.gen_undirected_edges(self._tour)

        # Store routes
        self._routes = self._tour
        if self._trucks > 1:
            self._routes = defaultdict(list)
            aux = 0
            for index, client in enumerate(self._tour):
                if index is not 0 and client == 1:
                    aux += 1
                self._routes[aux].append(client)

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

    # Set a new best tour and write to file
    @best_solution.setter
    def best_solution(self, solution):

        # Set best solution
        if self._best_solution is None:
            self._best_solution = solution
        elif (solution.dist < self._best_solution.dist
              and all(load <= self._capacity for load in solution.load)):
            self._best_solution = solution
        else:
            return

        # Write new solution to file
        with open(self._instance_name + ".opt.tour.new", 'w') as best:
            best.write("NAME : " + self._name + ".opt.tour.new\n")
            best.write("COMMENT : Length " + str(solution.dist)
                                           + ", ozeasx@gmail.com\n")
            best.write("TYPE : TOUR\n")
            best.write("DIMENSION : " + str(self._dimension) + "\n")
            best.write("TOUR_SECTION\n")
            for node in solution.tour:
                best.write(str(node) + "\n")
            best.write("-1\n")
            best.write("EOF\n")

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
