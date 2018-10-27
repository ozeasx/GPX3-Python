#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
from collections import defaultdict
from collections import deque
import numpy as np
import copy
import sys
from graph import Graph

#sys.setrecursionlimit(1500)

# Infeasible value compared to feasible partitions
INFEASIBLE_WEIGHT = 0.4

class Chromosome:
    # Constructor
    def __init__(self, tour):
        self.set(tour)

    # Chromosome setup
    def set(self, tour):
        # Creates tour
        # Random tour based on int number
        if isinstance(tour, int):
            V = range(2, tour + 1)
            tour = list(np.random.choice(V, len(V), replace=False))
            tour.insert(0, 1)
        # User defined
        self._tour = list(tour)

        self._undirected_graph = Graph.gen_undirected_graph(self._tour)

    # Get tour
    def get_tour(self):
        return self._tour

    # Get undirected graph
    def get_undirected_graph(self):
        return self._undirected_graph

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start):
        visited, stack, ab_cycle = set(), [start], deque()
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                ab_cycle.append(vertex)
                visited.add(vertex)
                stack.extend(graph[vertex] - visited)
        return visited, ab_cycle

    # Find partitions using dfs
    def _partition(self, graph_a, graph_b):
        # Simetric diference
        graph = graph_a ^ graph_b
        # Vertice set
        vertices = dict()
        # AB cycles
        ab_cycles = dict()
        loop = set(graph)
        index = 1
        while loop:
            vertices[index], ab_cycles[index] = self._dfs(graph, loop.pop())
            # Normalize AB cycles to begin with solution A
            if ab_cycles[index][0] in graph_b:
                if ab_cycles[index][1] in graph_b[ab_cycles[index][0]]:
                    ab_cycles[index].rotate(-1)
            ab_cycles[index] = list(ab_cycles[index])
            loop -= vertices[index]
            index += 1
        return vertices, ab_cycles

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, partitions, tour):
        simple_tour = defaultdict(deque)
        simple_graph = defaultdict(dict)

        # TODO: Optimize this
        aux_tour = list(tour)
        aux_tour.append(tour[0])

        # Identify entrance and exit vertices
        for i, j in zip(aux_tour[:-1], aux_tour[1:]):
            for key in partitions:
                # Entrance
                if i not in partitions[key] and j in partitions[key]:
                    #simple_tour[key].append(j)
                    simple_tour[key].extend([i, j])
                # Exit
                if i in partitions[key] and j not in partitions[key]:
                    #simple_tour[key].extend([i, 'c'])
                    simple_tour[key].extend([i, j])
                    simple_tour[key].append('c')

        # Covert tour to simple graph
        for key in simple_tour:
            # rotate by 'c'
            p = list(reversed(simple_tour[key])).index('c')
            simple_tour[key].rotate(p)
            #print simple_tour[key]
            simple_tour[key] = list(simple_tour[key])
            simple_graph[key] = defaultdict(set)
            # Converts permutation to graph
            for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
                if not (i == 'c' or j == 'c'):
                    simple_graph[key][i].add(j)
                    simple_graph[key][j].add(i)
            simple_graph[key] = dict(simple_graph[key])

        return dict(simple_graph)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, simple_g_a, simple_g_b):

        feasible_1 = set()
        feasible_2 = set()
        infeasible = set()

        for key in simple_g_a:
            if simple_g_a[key] == simple_g_b[key]:
                feasible_1.add(key)
            else:
                infeasible.add(key)

        return feasible_1, feasible_2, infeasible

    def _fusion(self, vertices, sga, tour_a, sgb, tour_b, infeasible):

        feasible_1 = set()
        feasible_2 = set()
        fused = set()

        n = 2
        while n < len(infeasible):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(infeasible, n):
                # Count common edges
                aux = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    aux += len(Graph(sga[i]) & Graph(sga[j]))

                fusion = list(fusion)
                fusion.append(aux)
                candidates.append(fusion)

            # Sort by common edges value
            candidates.sort(key = lambda fusion: fusion[n], reverse = True)
            for fusion in candidates:
                fusion.pop(-1)
            candidates = [tuple(fusion) for fusion in candidates]

            # Try fusions
            for fusion in candidates:
                union = dict()
                # Test to determine if partition is fused already
                if not any(i in fused for i in fusion):
                    union[fusion] = vertices[fusion[0]]
                    for i in fusion[1:]:
                        union[fusion] |= vertices[i]

                    sg_a = self._gen_simple_graph(union, tour_a)
                    sg_b = self._gen_simple_graph(union, tour_b)

                    # Check if fusion is feasible
                    f1, f2, _ = self._return_feasible(sg_a, sg_b)

                    # Update information
                    if fusion in f1 or fusion in f2:
                        for i in fusion:
                            infeasible.remove(i)
                            fused.add(i)
                            feasible_1.update(f1)
                            feasible_2.update(f2)

            # Increment fusion size
            n += 1

        # Fuse all remaining partitions
        if len(infeasible) > 1:
            feasible_1.add(tuple(infeasible))
            infeasible.clear()

        return feasible_1, feasible_2

    #def _build(self, vertices, feasible_1, feasible_2, infeasible):
    #    feasible_1.update(feasible_2)

    # Partition Crossover
    def __mul__(self, other):
        # Tours
        tour_a = list(self._tour)
        tour_b = list(other._tour)
        tour_c = list(other._tour)

        # Undirected union graph (G*)
        undirected_graph = self._undirected_graph | other._undirected_graph

        for vertice in undirected_graph:
            # Create ghost nodes for degree 4 nodes
            if len(undirected_graph[vertice]) == 4:
                tour_a.insert(tour_a.index(vertice) + 1, -vertice)
                tour_b.insert(tour_b.index(vertice) + 1, -vertice)
                tour_c.insert(tour_c.index(vertice), -vertice)
            # Remove degree 2 nodes
            if len(undirected_graph[vertice]) == 2:
                tour_a.remove(vertice)
                tour_b.remove(vertice)
                tour_c.remove(vertice)

        # Recreate graphs
        undirected_a = Graph.gen_undirected_graph(tour_a)
        undirected_b = Graph.gen_undirected_graph(tour_b)
        undirected_c = Graph.gen_undirected_graph(tour_c)

        # Partitioning schemes m, n
        vertices_m, ab_cycles_m = self._partition(undirected_a, undirected_b)
        vertices_n, ab_cycles_n = self._partition(undirected_a, undirected_c)

        # Generate simple graphs for each partitioning scheme for each tour
        simple_graph_a_m = self._gen_simple_graph(vertices_m, tour_a)
        simple_graph_b_m = self._gen_simple_graph(vertices_m, tour_b)

        simple_graph_a_n = self._gen_simple_graph(vertices_n, tour_a)
        simple_graph_c_n = self._gen_simple_graph(vertices_n, tour_c)

        # Test simple graphs to identify feasible partitions
        feasible_1_m, feasible_2_m, infeasible_m = \
            self._return_feasible(simple_graph_a_m, simple_graph_b_m)

        feasible_1_n, feasible_2_n, infeasible_n = \
            self._return_feasible(simple_graph_a_n, simple_graph_c_n)

        # Score partitions scheme
        score_m = (len(feasible_1_m) + len(feasible_2_m) +
                   len(infeasible_m) * INFEASIBLE_WEIGHT)

        score_n = (len(feasible_1_n) + len(feasible_2_n) +
                   len(infeasible_n) * INFEASIBLE_WEIGHT)

        # Choose better partitioning scheme
        if score_m >= score_n:
            feasible_1 = feasible_1_m
            feasible_2 = feasible_2_m
            infeasible = infeasible_m
            vertices = vertices_m
            ab_cycles = ab_cycles_m
            simple_graph_a = simple_graph_a_m
            simple_graph_b = simple_graph_b_m
        else:
            feasible_1 = feasible_1_n
            feasible_2 = feasible_2_n
            infeasible = infeasible_n
            vertices = vertices_n
            ab_cycles = ab_cycles_n
            tour_b = tour_c
            simple_graph_a = simple_graph_a_n
            simple_graph_b = simple_graph_c_n

        # Fusion
        f1, f2 = self._fusion(vertices, simple_graph_a, tour_a, simple_graph_b,
                              tour_b, infeasible)
        feasible_1.update(f1)
        feasible_2.update(f2)

        if True:
            print "Tour 1: ", tour_a
            print "Tour 2: ", tour_b
            print
            #print simple_graph_1
            #print simple_graph_2
            print
            print ab_cycles
            print
            print "Feasible 1: ", feasible_1
            print "Feasible 2: ", feasible_2
            print "Infeasible: ", infeasible

        return (feasible_1, feasible_2, infeasible, simple_graph_a,
                simple_graph_b, ab_cycles, tour_a, tour_b)

def test(size, limit):
    for x in xrange(limit):
        p1 = Chromosome(size)
        p2 = Chromosome(size)
        while (p1.get_undirected_graph() == p2.get_undirected_graph()):
            p2 = Chromosome(size)
        f1, f2, infeasible, sga, sgb, ab_cycles, tour_a, tour_b = p1 * p2
        print '\r', x,
        #if any(x < 0 for x in tour_a):
        #    continue
        if (len(f2) >= 4):
            print
            print "Count: ", x
            print "Tour 1: ", tour_a
            print "Tour 2: ", tour_b
            print
            print "Partitions: ", ab_cycles
            print
            print "Feasible 1: ", f1
            print "Feasible 2: ", f2
            print "Infeasible: ", infeasible
            break

#test(1000, 1000)

# Whitley2010-F1
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32])
#p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
#                7,8,5,6,4,3,22,21,24,23,2])
#p2 = Chromosome([1,2,23,24,21,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,
#                20,25,26,27,28,12,11,31,32])

# Hains 2011F-2.3
#p1 = Chromosome((1,2,3,4,5,6,7,8,9,10,11,12,13,14))
#p2 = Chromosome((1,13,12,10,9,7,6,8,5,4,11,3,2,14))

# Whitley2011-F1
#p1 = Chromosome((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
#                 25,26,27,28))
#p2 = Chromosome((1,3,2,5,4,6,7,11,12,10,13,14,15,17,16,19,18,20,21,9,8,22,23,25,
#                 24,27,26,28))

# F1 2i
#p1 = Chromosome((1, 7, 5, 6, 2, 10, 9, 3, 8, 4))
#p2 = Chromosome((1, 3, 8, 9, 7, 10, 2, 4, 5, 6))

#F2 3i
#p1 = Chromosome((1,10,4,6,2,8,5,9,3,7))
#p2 = Chromosome((1,3,2,9,7,5,6,10,4,8))

#F6 2f1
#p1 = Chromosome([1,3,6,7,10,4,8,11,2,5,9])
#p2 = Chromosome([1,4,8,2,11,5,9,6,7,10,3])

#F7 2i
#p1 = Chromosome([1,7,4,8,6,5,2,3])
#p2 = Chromosome([1,5,2,8,6,7,4,3])

#F8 3i
#p1 = Chromosome([1,2,3,4,5,6,7,8,9])
#p2 = Chromosome([1,8,6,4,2,9,7,5,3])

#F9 1f, 3i (1f, 1f2, 2i) execption
#p1 = Chromosome((1,9,8,10,7,2,5,4,3,6))
#p2 = Chromosome((1,8,6,2,3,10,5,9,4,7))

#F10 2f2 2if
#p1 = Chromosome((1,2,6,7,11,4,9,3,8,5,10,12))
#p2 = Chromosome((1,11,8,3,7,12,10,6,9,4,5,2))

#F11 4i (2 fusions) execption
#p1 = Chromosome((1,2,9,7,10,12,3,6,5,4,11,8))
#p2 = Chromosome((1,5,12,11,9,3,10,8,7,6,2,4))

#F12 5i (one fusion)
#p1 = Chromosome((1,3,10,8,11,2,5,7,6,4,9,12))
#p2 = Chromosome((1,4,2,8,12,6,3,9,11,10,5,7))

#F13 1f1, 2if
#p1 = Chromosome((1,4,3,9,6,2,8,7,10,5))
#p2 = Chromosome((9,7,10,6,3,8,5,4,2,1))

# F14 2f2 2if
#p1 = Chromosome((1, 9, 5, 8, 7, 11, 4, 10, 3, 12, 6, 2))
#p2 = Chromosome((1, 10, 7, 11, 6, 9, 3, 12, 5, 8, 4, 2))

# Teste
#p1 = Chromosome((1,2,3,4,5,6))
#p2 = Chromosome((1,2,3,6,4,5))

# Force test
#p1 = Chromosome(100000)
#p2 = Chromosome(100000)

p1 * p2
