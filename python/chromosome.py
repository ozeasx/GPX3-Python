#!/usr/bin/python
# ozeasx@gmail.com

import random
from graph import Graph
from collections import deque


class Chromosome(object):
    # Constructor
    def __init__(self, *args):
        if len(args):
            # Random tour
            if isinstance(args[0], int) and len(args) == 1:
                tour = range(2, args[0] + 1)
                random.shuffle(tour)
                self._tour = tuple([1] + tour)
                self._dimension = args[0]
            # Defined tour
            elif isinstance(args[0], (tuple, list, deque)):
                assert len(args[0]) == len(set(args[0]))
                self._tour = tuple(args[0])
                self._dimension = len(args[0])
            # Distance
            if len(args) == 2 and isinstance(args[1], (int, float)):
                self._dist = args[1]
            else:
                self._dist = None

        # Check if the needed attributes were created
        if not all(p in self.__dict__ for p in ('_tour', '_dimension',
                                                '_dist')):
            print("Cannot create (Chromosome) object")
            exit()

        # undirected graph and edges representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)
        self._undirected_edges = Graph.gen_undirected_edges(self._tour)

    # Equality (== operator)
    def __eq__(self, other):
        return hash(self) == hash(other)

    # Inequality (!=)
    def __ne__(self, other):
        return not self.__eq__(other)

    # Identity
    def __hash__(self):
        return hash(self._undirected_edges)

    # Get tour
    @property
    def tour(self):
        return self._tour

    # Get tour distance
    @property
    def dist(self):
        return self._dist

    # Get fitness
    @property
    def fitness(self):
        return self._fitness

    # Get dimension
    @property
    def dimension(self):
        return self._dimension

    # Get undirected graph
    @property
    def undirected_graph(self):
        return Graph(self._undirected_graph)

    # Get undirected edges
    @property
    def undirected_edges(self):
        return self._undirected_edges

    # Distance setter
    @dist.setter
    def dist(self, value):
        self._dist = value

    # Fitness setter
    @fitness.setter
    def fitness(self, value):
        self._fitness = value


# Test section
if __name__ == '__main__':
    c = Chromosome(5, 1)
    print(c.__dict__)
