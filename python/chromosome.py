#!/usr/bin/python
# ozeasx@gmail.com

import numpy as np
from collections import deque
from graph import Graph


class Chromosome(object):
    # Constructor
    def __init__(self, tour=None, dist=None):
        # Create random tour based on given dimension
        if isinstance(tour, int):
            nodes = range(2, tour + 1)
            self._tour = list(np.random.choice(nodes, len(nodes),
                                               replace=False))
            self._tour.insert(0, 1)
            self._tour = tuple(self._tour)
        # User defined tour
        elif isinstance(tour, list) or isinstance(tour, deque):
            self._tour = tuple(tour)

        # Tour distance
        if dist:
            self._dist = dist

        # Fitness
        self._fitness = None

        # Number of cities
        self._dimension = len(self.tour)

        # undirected graph representaition
        if self.tour:
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

    def __str__(self):
        return str(self.fitness)

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

    # Set fitness
    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    # Calculate solutions dissimilarity
    def __sub__(self, other):
        # edges_1 = Graph.gen_undirected_edges(self.tour)
        # edges_2 = Graph.gen_undirected_edges(other.tour)
        # return 1 - len(edges_1 & edges_2)/float(len(edges_1 | edges_2))
        return 1 - abs(self.dist - other.dist)/(self.dist + other.dist)
