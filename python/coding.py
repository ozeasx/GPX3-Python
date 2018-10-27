# Common nodes (intersection)
    # common = self._edges & other._edges
    # Uncommon nodes (union - intersection)
    #uncommon = union - common
    # Align solutions
    # aligned = True
    # if len(common):
        # https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it
    #    for e in common:
    #        break
    #    if any(e == (i, j) for i, j in zip(self._tour[:-1], self._tour[1:])):
    #        aligned = not aligned
    #    if any(e == (i, j) for i, j in zip(other._tour[:-1], other._tour[1:])):
    #        aligned = not aligned
    #    if not aligned:
    #        other.set(reversed(other._tour))

    # Creat ghost nodes

      #c1 = list(self._tour)
      #c2 = list(other._tour)
      # Create ghost nodes
      #for i in ghost:
      #    p1 = c1.index(i)
      #    p2 = c2.index(i)
      #    c1[p1:p1+1] = i, -i
      #    c2[p2:p2+1] = i, -i
      # Create partial solutions
      #c1_edges = self.gen(c1)
      #c2_edges = self.gen(c2)
      # Union
      #union = c1_edges | c2_edges
      # Common edges
      #common = c1_edges & c2_edges
      # Uncommon nodes (union - intersection)
      #uncommon = union - common


        # Create partial tours
        tour_1 = list(self._tour)
        tour_2 = list(other._tour)
        tour_3 = list(reversed(other._tour))
        # Create ghost nodes
        for i in ghost:
            p1 = tour_1.index(i)
            p2 = tour_2.index(i)
            p3 = tour_3.index(i)
            tour_1[p1:p1+1] = i, -i
            tour_2[p2:p2+1] = i, -i
            tour_3[p3:p3+1] = i, -i
        # Generate new edges sets
        edges_1 = self._gen_edges(tour_1)
        edges_2 = self._gen_edges(tour_2)
        edges_3 = self._gen_edges(tour_3)

        def _gen_edges(self, tour):
            edges = set()
            for i, j in zip(tour[:-1], tour[1:]):
                edges.add(tuple(sorted([i, j])))
            return set(edges)

        # Create ghost nodes for degree 4 nodes
        for vertice in xrange(1, self._size + 1):
            if sum(vertice in edge for edge in union) == 4:
                p1 = tour_1.index(vertice)
                p2 = tour_2.index(vertice)
                tour_1.insert(p1 + 1, -vertice)
                tour_2.insert(p2 + 1, -vertice)
                tour_3.insert(p2, -vertice)


class Graph:
    def __init__(self):
        self._nodes = set()
        self._edges = set()

    def get_nodes(self):
        return set(self._nodes)

    def get_edges(self):
        return set(self._edges)

    def generate(self, tour):
        self._edges = set()
        for i, j in zip(tour[:-1], tour[1:]):
            self._nodes.add(i)
            self._edges.add(tuple([i,j]))

    def add_node(self, node):
        self._nodes.add(node)

    def add_nodes(self, nodes):
        for i in nodes:
            self._nodes.add(i)

    def add_edge(self, i, j):
        self._edges.add(tuple([i, i]))

    def add_edges(self, edges):
        for i, j in edges:
            self._edges.add(tuple([i, j]))

    def remove_node(self, node):
        if node in self._nodes:
            edges = set(self._edges)
            for edge in edges:
                if node in edge:
                    self._edges.remove(edge)
            self._nodes.remove(node)
        else:
            print "%s not in nodes" % node

    def remove_edge(self, i, j):
        if tuple([i, j]) in self._edges:
            self._edges.remove(tuple[i, j])
        else:
            print "(%s, %s) not in edges" % (i, j)

grafo = Graph()

grafo.add_nodes([1,2,3,4,5])
grafo.add_edges([(1,2),(2,3),(3,4),(4,5),(5,1)])

print grafo.get_nodes()
print grafo.get_edges()

grafo.remove_node(6)
grafo.remove_edge(1,1)

    def _dfs(self, graph, start, visited=None):
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
        if visited is None:
            visited = set()
            tour = list()
        visited.add(start)
        tour.append(start)
        for next in graph[start] - visited:
            if next not in visited:
                self._dfs(graph, next, visited, tour)
            return visited, tour

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    #def _dfs(self, graph, start, visited = None, tour = None):
    #    if visited is None:
    #        visited = set()
    #        tour = list()
    #    visited.add(start)
    #    tour.append(start)
    #    for next in graph[start] - visited:
    #        self._dfs(graph, next, visited, tour)
    #    return tour[:-1]

    # Create the simple graph for given partition
    #def _gen_simple_graph(self, graph, partitions, start, visited = None, simple_graph = None):
    #    if visited is None:
    #        visited = set()
    #        simple_graph = set()
    #    visited.add(start)
    #    for next in graph[start] - visited:
    #        if start in partition.values() and next not in partition.values():
    #            simple_graph.add(start)
    #        if start not in partition.values() and next in partition.values():
    #            simple_graph.add(next)
    #        self._dfs_simple_graph(graph, partitions, start, visited, simple_graph)
    #    return simple_graph

        def _gen_simple_graph(self, tour, partitions):
            simple_graph = defaultdict(set)
            entrance = exit = None

            for key in partitions:
                for i, j in zip(tour[:-1], tour[1:]):
                    # Entrance
                    if j in partitions[key] and i not in partitions[key]:
                        entrance = j
                    # Exit
                    if i in partitions[key] and j not in partitions[key]:
                        exit = i
                    if entrance and exit:
                        simple_graph[key].add(tuple(sorted([entrance, exit])))
                        entrance = exit = None

import numpy as np
from collections import defaultdict
from collections import deque
from graph import Graph

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
            self._tour = list(np.random.choice(V, len(V), replace=False))
            self._tour.insert(0, 1)
            self._tour.append(1)
        # User defined tour
        else:
            self._tour = tour

        self._graph_undirected = Graph.gen_graph_undirected(self._tour)
        self._edges_directed = Graph.gen_edges_directed(self._tour)

    # Get tour
    def get_tour(self):
        return self._tour

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start, visited = None, tour = None):
        if visited is None:
            visited = set()
            tour = list()
        visited.add(start)
        tour.append(start)
        for next in graph[start] - visited:
            if next not in visited:
                self._dfs(graph, next, visited, tour)
            return visited, tour

    # Function to generate solutions using new edges as possible
    def _gen_diversity(self, size):
        V = set(xrange(1, size + 1))
        graph = dict()
        for vertice in V:
            graph[vertice] = set(V)
            graph[vertice].remove(vertice)

        while graph:
            visited, tour = self._dfs(graph, graph.iterkeys().next())
            print tour

            for i, j in zip(tour[:-1], tour[1:]):
                if i in graph:
                    graph[i].discard(j)
                if j in graph:
                    graph[j].discard(i)

            for vertice in graph.keys():
                if not graph[vertice]:
                    del graph[vertice]

    # Find partitions using dfs
    def _partition(self, graph):
        partitions = dict()
        tours = dict()
        vertices = set(graph)
        index = 1
        while vertices:
            partitions[index], tours[index] = self._dfs(graph, vertices.pop())
            vertices -= partitions[index]
            index += 1
        return partitions, tours

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, tour, partitions):
        simple_graph = defaultdict(list)
        simple_set = defaultdict(set)

        # Identify entrance and exit vertices
        for i, j in zip(tour[:-1], tour[1:]):
            for key in partitions:
                # Entrance
                if j in partitions[key] and i not in partitions[key]:
                    simple_graph[key].append(j)
                    simple_set[key].add(tuple([j, 'in']))
                # Exit
                if i in partitions[key] and j not in partitions[key]:
                    simple_graph[key].append(i)
                    simple_set[key].add(tuple([i, 'out']))

        # Invert simple_graph lists by min value
        for part in simple_graph.values():
            if part[-1] < part[0]:
                part.reverse()

        return dict(simple_graph), dict(simple_set)

    def _gen_simple_graph(self, tour, partitions):
        simple_graph = defaultdict(set)

        # Identify entrance and exit vertices
        for i, j in zip(tour[:-1], tour[1:]):
            for key in partitions:
                # Entrance
                if j in partitions[key] and i not in partitions[key]:
                    simple_graph[key].add(tuple([i, j]))
                # Exit
                if i in partitions[key] and j not in partitions[key]:
                    simple_graph[key].add(tuple([i, j]))

        return dict(simple_graph)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, simple_graph_1, simple_graph_2, simple_set_1,
                         simple_set_2, tours):

        feasible = dict()
        infeasible = dict()

        for key in simple_graph_1:
            # First test
            if simple_graph_1[key] == simple_graph_2[key]:
                feasible[key] = tours[key]
            # Second test
            elif simple_set_1[key] != simple_set_2[key]:
                feasible[key] = tours[key]
            else:
                infeasible[key] = tours[key]

        return feasible, infeasible

    def _return_feasible(self, simple_graph_1, simple_graph_2, tours):

        feasible = dict()
        infeasible = dict()

        for key in simple_graph_1:
            if len(simple_graph_1[key]) == 2:
                if simple_graph_1[key] == simple_graph_2[key]:
                    feasible[key] = tours[key]
                else:
                    infeasible[key] = tours[key]
            elif simple_graph_1[key] != simple_graph_2[key]:
                feasible[key] = tours[key]
            else:
                infeasible[key] = tours[key]

        return feasible, infeasible

    def _fusion(self, common, infeasible):
        union = Graph(common)

        for tour in infeasible.values():
            tour.append(tour[0])
            union |= Graph.gen_graph_undirected(tour)

        fusion = dict(union)
        end = False
        while not end:
            end = True
            for k in union:
                if k in fusion and len(fusion[k]) == 1:
                    for v in fusion[k]:
                        fusion[v].remove(k)
                        if not len(fusion[v]):
                            del fusion[v]
                        del fusion[k]
                    end = False

        print fusion

    # Partitions fusion
    def _fusion(self, common, infeasible):
        union = Graph(common)

        for tour in infeasible.values():
            tour.append(tour[0])
            union |= Graph.gen_graph_undirected(tour)

        fusion = dict(union)
        end = False
        while not end:
            end = True
            for k in union:
                if k in fusion and len(fusion[k]) == 1:
                    for v in fusion[k]:
                        fusion[v].remove(k)
                        if not len(fusion[v]):
                            del fusion[v]
                        del fusion[k]
                    end = False

        print fusion


    # Partition Crossover
    def __mul__(self, other):
        # Tours
        tour_1 = list(self._tour)
        tour_2 = list(other._tour)

        # Align tours
        foo = Graph.gen_edges_directed(list(reversed(tour_2)))
        if (len(self._edges_directed & other._edges_directed) <
            len(self._edges_directed & foo)):
            tour_2.reverse()
        tour_3 = list(tour_2)

        # Union (G*)
        undirected_union = self._graph_undirected | other._graph_undirected

        # Create ghost nodes for degree 4 nodes
        for vertice in undirected_union:
            if len(undirected_union[vertice]) == 4:
                tour_1.insert(tour_1.index(vertice) + 1, -vertice)
                tour_2.insert(tour_2.index(vertice) + 1, -vertice)
                tour_3.insert(tour_3.index(vertice), -vertice)
    # Partitions fusion
    def _fusion(self, common, infeasible):
        union = Graph(common)

        for tour in infeasible.values():
            tour.append(tour[0])
            union |= Graph.gen_graph_undirected(tour)

        fusion = dict(union)
        end = False
        while not end:
            end = True
            for k in union:
                if k in fusion and len(fusion[k]) == 1:
                    for v in fusion[k]:
                        fusion[v].remove(k)
                        if not len(fusion[v]):
                            del fusion[v]
                        del fusion[k]
                    end = False

        print fusion
        # Recreate graphs
        graph_undirected_1 = Graph.gen_graph_undirected(tour_1)
        graph_undirected_2 = Graph.gen_graph_undirected(tour_2)
        graph_undirected_3 = Graph.gen_graph_undirected(tour_3)

        # Partitioning
        partitions_1_2, tours_1_2 = (self._partition(graph_undirected_1 ^
                                     graph_undirected_2))
        partitions_1_3, tours_1_3 = (self._partition(graph_undirected_1 ^
                                     graph_undirected_3))

        # Choose better partitioning
        if len(partitions_1_2) > len(partitions_1_3):
            partitions = partitions_1_2
            tours = tours_1_2
            graph_common = graph_undirected_1 & graph_undirected_2
        else:
            partitions = partitions_1_3
            tours = tours_1_3
            graph_common = graph_undirected_1 & graph_undirected_3
            tour_2 = tour_3

        #print partitions

        # Create simple graphs
        simple_graph_1, simple_set_1 = (self._gen_simple_graph(tour_1,
                                          partitions))
        simple_graph_2, simple_set_2 = (self._gen_simple_graph(tour_2,
                                          partitions))

        print simple_graph_1
        print simple_graph_2

        print simple_set_1
        print simple_set_2

        # Identify feasible and infeasible partitions
        feasible, infeasible = (self._return_feasible(simple_graph_1,
                                simple_graph_2, simple_set_1, simple_set_2,
                                tours))

        print feasible
        print infeasible

        self._fusion(graph_common, infeasible)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,1])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2,1])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,1])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3,1])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2,1])

# Tinos2014-F5
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                24,25,26,27,28,29,30,31,32,1])
#p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
#                7,8,5,6,4,3,22,21,24,23,2,1])
p2 = Chromosome([1,2,23,24,21,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,
                20,25,26,27,28,12,11,31,32,1])

# Tinos2014-F5b
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32])
#p2 = Chromosome([1,2,23,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,20,21,
#                24,25,26,27,28,12,11,31,32])

p1 * p2




#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
from collections import defaultdict
from collections import deque
import numpy as np
from graph import Graph

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
            self._tour = list(np.random.choice(V, len(V), replace=False))
            self._tour.insert(0, 1)
            self._tour.append(1)
        # User defined tour
        else:
            self._tour = tour

        self._graph_undirected = Graph.gen_graph_undirected(self._tour)
        self._edges_directed = Graph.gen_edges_directed(self._tour)

    # Get tour
    def get_tour(self):
        return self._tour

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start, visited = None, tour = None):
        if visited is None:
            visited = set()
            tour = list()
        visited.add(start)
        tour.append(start)
        for next in graph[start] - visited:
            self._dfs(graph, next, visited, tour)
            return visited, tour

    # Find partitions using dfs
    def _partition(self, graph):
        partitions = dict()
        tours = dict()
        vertices = set(graph)
        index = 1
        while vertices:
            partitions[index], tours[index] = self._dfs(graph, vertices.pop())
            vertices -= partitions[index]
            index += 1
        return partitions, tours

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, tour, partitions):
        simple_graph = defaultdict(deque)
        simple_set = defaultdict(set)

        # Identify entrance and exit vertices
        for i, j in zip(tour[:-1], tour[1:]):
            for key in partitions:
                # Entrance
                if j in partitions[key] and i not in partitions[key]:
                    simple_graph[key].append(j)
                    simple_set[key].add(tuple([j, 'in']))
                # Exit
                if i in partitions[key] and j not in partitions[key]:
                    simple_graph[key].append(i)
                    simple_set[key].add(tuple([i, 'out']))

        # Normalize simple graphs
        for part in simple_graph.values():
            if part[-1] < part[0]:
                part.reverse()
            m = list(part).index(min(part))
            part.rotate(-m)

        return dict(simple_graph), dict(simple_set)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, simple_graphs, simple_sets, tours):

        feasible_1 = dict()
        feasible_2 = dict()
        infeasible = dict()

        for key in simple_graphs[0]:
            # First test
            if simple_graphs[0][key] == simple_graphs[1][key]:
                feasible_1[key] = tours[key]
            # Second test
            elif simple_sets[0][key] != simple_sets[1][key]:
                feasible_2[key] = tours[key]
            else:
                infeasible[key] = tours[key]

        return feasible_1, feasible_2, infeasible

    # Find better partitioning scheme
    def _find_better(self, tours, schemes):
        # Create simple graphs for each partitioning scheme
        simple_1 = self._gen_simple_graph(tours[0], schemes[0][0])
        simple_2 = self._gen_simple_graph(tours[1], schemes[0][0])

        simple_3 = self._gen_simple_graph(tours[0], schemes[1][0])
        simple_4 = self._gen_simple_graph(tours[2], schemes[1][0])





    def _fusion(self, common, infeasible):
        union = Graph(common)

        for tour in infeasible.values():
            tour.append(tour[0])
            union |= Graph.gen_graph_undirected(tour)

        fusion = dict(union)
        end = False
        while not end:
            end = True
            for k in union:
                if k in fusion and len(fusion[k]) == 1:
                    for v in fusion[k]:
                        fusion[v].remove(k)
                        if not len(fusion[v]):
                            del fusion[v]
                        del fusion[k]
                    end = False

        print fusion

    # Partition Crossover
    def __mul__(self, other):
        # Tours
        tour_1 = list(self._tour)
        tour_2 = list(other._tour)
        tour_3 = list(other._tour)

        # Union (G*)
        undirected_union = self._graph_undirected | other._graph_undirected

        # Create ghost nodes for degree 4 nodes
        for vertice in undirected_union:
            if len(undirected_union[vertice]) == 4:
                tour_1.insert(tour_1.index(vertice) + 1, -vertice)
                tour_2.insert(tour_2.index(vertice) + 1, -vertice)
                tour_3.insert(tour_3.index(vertice), -vertice)

        # Recreate graphs
        g_undirected_1 = Graph.gen_graph_undirected(tour_1)
        g_undirected_2 = Graph.gen_graph_undirected(tour_2)
        g_undirected_3 = Graph.gen_graph_undirected(tour_3)

        # Partitioning schemes
        scheme_a, tours_a = self._partition(g_undirected_1 ^ g_undirected_2)
        scheme_b, tours_b = self._partition(g_undirected_1 ^ g_undirected_3)

        # Find better partitioning scheme
        tours = (tour_1, tour_2, tour_3)
        schemes = (scheme_a, scheme_b)
        partitions = self._find_better(tours, schemes)


        # Identify feasible and infeasible partitions
        #partitions = (self._return_feasible(simple_graph_1, simple_graph_2,
        #              tours))

        #print partitions[0]
        #print partitions[1]
        #print partitions[2]

        #return tour_1, tour_2, feasible_1, feasible_2, infeasible

        #self._fusion(graph_common, infeasible)

def test(size, limit):
    for x in xrange(limit):
        p1 = Chromosome(size)
        p2 = Chromosome(size)
        tour_1, tour_2, feasible_1, feasible_2, infeasible = p1 * p2
        print '\r', x,
        if any(x < 0 for x in tour_1):
            continue
        if (len(feasible_1) or len(feasible_2)) and len(infeasible):
            print "Count: ", x
            print "Tour 1: ", tour_1
            print "Tour 2: ", tour_2
            print "Feasible partitions type 1: ", feasible_1
            print "Feasible partitions type 2: ", feasible_2
            print "infeasible partitions: ", infeasible
            break

#test(11, 100000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,1])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2,1])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,1])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3,1])

# Tinos2014-F2
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1])
p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2,1])

# Tinos2014-F5
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32,1])
#p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
#                7,8,5,6,4,3,22,21,24,23,2,1])
#p2 = Chromosome([1,2,23,24,21,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,
#                20,25,26,27,28,12,11,31,32,1])

# Tinos2014-F5b
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32])
#p2 = Chromosome([1,2,23,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,20,21,
#                24,25,26,27,28,12,11,31,32])

# F1
#p1 = Chromosome([1, 7, 5, 6, 2, 10, 9, 3, 8, 4, 1])
#p2 = Chromosome([1, 6, 5, 4, 2, 10, 7, 9, 8, 3, 1])

#F2
#p1 = Chromosome([1,-1,10,4, 6, -6, 2, -2, 8, -8, 5, -5, 9, -9, 3, -3, 7, -7, 1])
#p2 = Chromosome([1,-1,3,-3, 2, -2, 9, -9, 7, -7, 5, -5, 6, -6, 10, 4, 8, -8, 1])

#F6
#p1 = Chromosome([1, 3, 6, 7, 10, 4, 8, 11, 2, 5, 9, 1])
#p2 = Chromosome([1, 4, 8, 2, 11, 5, 9, 6, 7, 10, 3, 1])

p1 * p2

#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
from collections import defaultdict
from collections import deque
import numpy as np
from graph import Graph

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
            tour.append(1)
        # User defined # Tuples are fast
        self._tour = tuple(tour)

        self._graph_undirected = Graph.gen_graph_undirected(self._tour)
        self._edges_directed = Graph.gen_edges_directed(self._tour)

    # Get tour
    def get_tour(self):
        return self._tour

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start, visited = None, tour = None):
        if visited is None:
            visited = set()
            tour = list()
        visited.add(start)
        tour.append(start)
        for next in graph[start] - visited:
            self._dfs(graph, next, visited, tour)
            return visited, tuple(tour)

    # Find partitions using dfs
    def _partition(self, graph):
        vertices = dict()
        tours = dict()
        loop = set(graph)
        index = 1
        while loop:
            vertices[index], tours[index] = self._dfs(graph, loop.pop())
            loop -= vertices[index]
            index += 1
        return vertices, tours

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, partitions, tour):
        simple_tour = defaultdict(deque)
        simple_graph = defaultdict(dict)

        # Identify entrance and exit vertices
        for i, j in zip(tour[:-1], tour[1:]):
            for key in partitions:
                # Entrance
                if j in partitions[key] and i not in partitions[key]:
                    simple_tour[key].append(j)
                # Exit
                if i in partitions[key] and j not in partitions[key]:
                    simple_tour[key].append(i)

        # Normalize simple tour
        for key in simple_tour:
            # Invert by min
            if simple_tour[key][-1] < simple_tour[key][0]:
                simple_tour[key].reverse()
            # rotate
            m = list(simple_tour[key]).index(min(simple_tour[key]))
            simple_tour[key].rotate(-m)
            # Tuples are fast
            simple_tour[key] = tuple(simple_tour[key])
            # Covert to graph
            for i, j in zip(simple_tour[key][0::2], simple_tour[key][1::2]):
                simple_graph[key][i] = j

        return dict(simple_graph)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, simple_g_1, simple_g_2, tours):

        feasible_1 = dict()
        feasible_2 = dict()
        infeasible = dict()

        for key in simple_g_1:
            # First test (simple graph)
            if simple_g_1[key] == simple_g_2[key]:
                feasible_1[key] = tours[key]
            # Second test
            elif (set(simple_g_1[key]) == set(simple_g_2[key]) and
            set(simple_g_1[key].values()) == set(simple_g_2[key].values())):
                if len(simple_g_1[key]) % 2:
                    feasible_2[key] = tours[key]
                else:
                    infeasible[key] = tours[key]
            else:
                feasible_2[key] = tours[key]

        return feasible_1, feasible_2, infeasible

    # Partition Crossover
    def __mul__(self, other):
        # Tours
        tour_1 = list(self._tour)
        tour_2 = list(other._tour)
        tour_3 = list(reversed(other._tour))

        # Union (G*)
        undirected_union = self._graph_undirected | other._graph_undirected

        # Create ghost nodes for degree 4 nodes
        for vertice in undirected_union:
            if len(undirected_union[vertice]) == 4:
                tour_1.insert(tour_1.index(vertice) + 1, -vertice)
                tour_2.insert(tour_2.index(vertice) + 1, -vertice)
                tour_3.insert(tour_3.index(vertice) + 1, -vertice)
            if len(undirected_union[vertice]) == 2:


        # Tuples are fast
        tour_1 = tuple(tour_1)
        tour_2 = tuple(tour_2)
        tour_3 = tuple(tour_3)

        # Recreate graphs
        g_undirected_1 = Graph.gen_graph_undirected(tour_1)
        g_undirected_2 = Graph.gen_graph_undirected(tour_2)
        g_undirected_3 = Graph.gen_graph_undirected(tour_3)

        # Partitioning schemes a and b
        vertices_a, tours_a = self._partition(g_undirected_1 ^ g_undirected_2)
        vertices_b, tours_b = self._partition(g_undirected_1 ^ g_undirected_3)

        # Generate simple graphs for each partitioning scheme for each tour
        simple_graph_a_1 = self._gen_simple_graph(vertices_a, tour_1)
        simple_graph_a_2 = self._gen_simple_graph(vertices_a, tour_2)

        simple_graph_b_1 = self._gen_simple_graph(vertices_b, tour_1)
        simple_graph_b_3 = self._gen_simple_graph(vertices_b, tour_3)

        #print simple_graph_a_1
        #print simple_graph_a_2
        #print
        #print simple_graph_b_1
        #print simple_graph_b_3
        #print

        feasible_1_a, feasible_2_a, infeasible_a = \
            self._return_feasible(simple_graph_a_1, simple_graph_a_2,
                                  tours_a)

        feasible_1_b, feasible_2_b, infeasible_b = \
            self._return_feasible(simple_graph_b_1, simple_graph_b_3,
                                  tours_b)

        score_a = (len(feasible_1_a) + len(feasible_2_a) +
                   len(infeasible_a)/float(2))

        score_b = (len(feasible_1_b) + len(feasible_2_b) +
                   len(infeasible_b)/float(2))

        # Choose better partitioning scheme
        if score_a >= score_b:
            feasible_1 = feasible_1_a
            feasible_2 = feasible_2_a
            infeasible = infeasible_a
            vertices = vertices_a
            tours = tours_a
            print simple_graph_a_1
            print simple_graph_a_2
        else:
            feasible_1 = feasible_1_b
            feasible_2 = feasible_2_b
            infeasible = infeasible_b
            vertices = vertices_b
            tours = tours_b
            tour_2 = tour_3
            print simple_graph_b_1
            print simple_graph_b_3

        #print "Tour 1: ", tour_1
        #print "Tour 2: ", tour_2
        #print
        #print "Feasible 1: ", feasible_1
        #print "Feasible 2: ", feasible_2
        #print "Infeasible: ", infeasible

        return feasible_1, feasible_2, infeasible, tours, tour_1, tour_2

def test(size, limit):
    for x in xrange(limit):
        p1 = Chromosome(size)
        p2 = Chromosome(size)
        feasible_1, feasible_2, infeasible, tours, tour_1, tour_2 = p1 * p2
        print '\r', x,
        #if any(x < 0 for x in tour_1):
        #    continue
        if len(infeasible) == 4:
            print
            print "Count: ", x
            print "Tour 1: ", tour_1
            print "Tour 2: ", tour_2
            print
            print "Feasible 1: ", feasible_1
            print "Feasible 2: ", feasible_2
            print "Infeasible: ", infeasible
            break

#test(12, 1000000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,1])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2,1])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,1])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3,1])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2,1])

# Tinos2014-F5
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32,1])
#p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
#                7,8,5,6,4,3,22,21,24,23,2,1])
#p2 = Chromosome([1,2,23,24,21,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,
#                20,25,26,27,28,12,11,31,32,1])

# Tinos2014-F5b
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32])
#p2 = Chromosome([1,2,23,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,20,21,
#                24,25,26,27,28,12,11,31,32])

# F1
#p1 = Chromosome([1, 7, 5, 6, 2, 10, 9, 3, 8, 4, 1])
#p2 = Chromosome([1, 6, 5, 4, 2, 10, 7, 9, 8, 3, 1])

#F2
#p1 = Chromosome([1,-1,10,4, 6, -6, 2, -2, 8, -8, 5, -5, 9, -9, 3, -3, 7, -7, 1])
#p2 = Chromosome([1,-1,3,-3, 2, -2, 9, -9, 7, -7, 5, -5, 6, -6, 10, 4, 8, -8, 1])

#F6
#p1 = Chromosome([1, 3, 6, 7, 10, 4, 8, 11, 2, 5, 9, 1])
#p2 = Chromosome([1, 4, 8, 2, 11, 5, 9, 6, 7, 10, 3, 1])

#F7
#(1, 7, 4, 8, 6, 5, 2, 3, 1)
#(1, 5, 2, 8, 6, 7, 4, 3, 1)

#F8
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,1])
#p2 = Chromosome([1,8,6,4,2,9,7,5,3,1])

#F9
#p1 = Chromosome((1, -1, 9, -9, 8, -8, 10, -10, 7, -7, 2, -2, 5, -5, 4, -4, 3, -3, 6, -6, 1))
#p2 = Chromosome((1, -1, 8, -8, 6, -6, 2, -2, 3, -3, 10, -10, 5, -5, 9, -9, 4, -4, 7, -7, 1))

#F10
#p1 = Chromosome((1, 2, 6, -6, 7, -7, 11, -11, 4, 9, 3, 8, 5, -5, 10, 12, 1))
#p2 = Chromosome((1, 11, -11, 8, 3, 7, -7, 12, 10, 6, -6, 9, 4, 5, -5, 2, 1))

#F11
p1 = Chromosome((1, -1, 2, -2, 9, -9, 7, -7, 10, -10, 12, -12, 3, -3, 6, -6, 5,
                -5, 4, -4, 11, -11, 8, -8, 1))
p2 = Chromosome((1, -1, 5, -5, 12, -12, 11, -11, 9, -9, 3, -3, 10, -10, 8, -8,
                 7, -7, 6, -6, 2, -2, 4, -4, 1))

p1 * p2


    def _return_feasible(self, simple_g_1, simple_g_2, tours):

        feasible_1 = dict()
        feasible_2 = dict()
        infeasible = dict()

        for key in simple_g_1:
            # First test (simple graph)
            if simple_g_1[key] == simple_g_2[key]:
                feasible_1[key] = tours[key]
            # Second test
            elif (set(simple_g_1[key]) == set(simple_g_2[key]) and
            set(simple_g_1[key].values()) == set(simple_g_2[key].values())):
                if len(simple_g_1[key]) % 2:
                    feasible_2[key] = tours[key]
                else:
                    infeasible[key] = tours[key]
            else:
                feasible_2[key] = tours[key]

        return feasible_1, feasible_2, infeasible


def _gen_simple_graph(self, partitions, tour):
    simple_tour = defaultdict(deque)
    simple_graph = defaultdict(dict)

    # TODO: Optimize this
    aux_tour = list(tour)
    aux_tour.append(tour[0])
    aux_tour = tuple(aux_tour)

    # Identify entrance and exit vertices
    for i, j in zip(aux_tour[:-1], aux_tour[1:]):
        for key in partitions:
            # Entrance
            if i not in partitions[key] and j in partitions[key]:
                #simple_tour[key].append(j)
                simple_tour[key].extend([i, j])
            # Exit
            if i in partitions[key] and j not in partitions[key]:
                #simple_tour[key].append(i)
                simple_tour[key].extend([i, j, 'c']) #add cut

    for key in simple_tour:
        # Invert by min
        if simple_tour[key][-2] < simple_tour[key][1]:
            simple_tour[key].reverse()
        # rotate by 'c'
        p = list(reversed(simple_tour[key])).index('c')
        simple_tour[key].rotate(p)
        # Tuples are fast
        simple_tour[key] = tuple(simple_tour[key])
        print simple_tour[key]
        # Converts permutation to graph
        for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
            if not (i == 'c' or j == 'c'):
                simple_graph[key][i] = j

    return dict(simple_graph)

    def _return_feasible(self, simple_g_1, simple_g_2, tours):

        feasible_1 = dict()
        feasible_2 = dict()
        infeasible = dict()

        for key in simple_g_1:
            # First test
            if simple_g_1[key] == simple_g_2[key]:
                feasible_1[key] = tours[key]
            elif (set(simple_g_1[key]) == set(simple_g_2[key].values()) and
                  set(simple_g_1[key].values()) == set(simple_g_2[key])):
                  feasible_1[key] = tours[key]
            # Second test
            elif (set(simple_g_1[key]) == set(simple_g_2[key]) and
            set(simple_g_1[key].values()) == set(simple_g_2[key].values())):
                if len(simple_g_1[key]) % 2:
                    feasible_2[key] = tours[key]
                else:
                    infeasible[key] = tours[key]
            else:
                feasible_2[key] = tours[key]

        return feasible_1, feasible_2, infeasible


#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
from collections import defaultdict
from collections import deque
import numpy as np
from graph import Graph

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
        # User defined # Tuples are fast
        self._tour = tuple(tour)

        self._graph_undirected = Graph.gen_graph_undirected(self._tour)
        self._edges_directed = Graph.gen_edges_directed(self._tour)

    # Get tour
    def get_tour(self):
        return self._tour

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start, visited = None, tour = None):
        if visited is None:
            visited = set()
            tour = list()
        visited.add(start)
        tour.append(start)
        for next in graph[start] - visited:
            self._dfs(graph, next, visited, tour)
            return visited, tuple(tour)

    # Find partitions using dfs
    def _partition(self, graph):
        vertices = dict()
        tours = dict()
        loop = set(graph)
        index = 1
        while loop:
            vertices[index], tours[index] = self._dfs(graph, loop.pop())
            loop -= vertices[index]
            index += 1
        return vertices, tours

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, partitions, tour):
        simple_tour = defaultdict(deque)
        simple_graph = defaultdict(dict)

        # TODO: Optimize this
        aux_tour = list(tour)
        aux_tour.append(tour[0])
        aux_tour = tuple(aux_tour)

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
                    simple_tour[key].extend([i, j, 'c']) #add cut

        for key in simple_tour:
            # rotate by 'c'
            p = list(reversed(simple_tour[key])).index('c')
            simple_tour[key].rotate(p)
            # Tuples are fast
            simple_tour[key] = tuple(simple_tour[key])
            #print simple_tour[key]
            # Converts permutation to graph
            for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
                if not (i == 'c' or j == 'c'):
                    simple_graph[key][i] = j

        return dict(simple_graph)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, simple_g_1, simple_g_2, tours):

        feasible_1 = dict()
        feasible_2 = dict()
        infeasible = dict()

        for key in simple_g_1:
            # Define tests values
            first = bool(simple_g_1[key] == simple_g_2[key] or
                    set(simple_g_1[key]) == set(simple_g_2[key].values()) and
                    set(simple_g_1[key].values()) == set(simple_g_2[key]))

            second = bool(set(simple_g_1[key]) == set(simple_g_2[key]) and
            set(simple_g_1[key].values()) == set(simple_g_2[key].values()))

            lenght = bool(len(simple_g_1[key]) % 2)

            # Test conditions
            if first:
                feasible_1[key] = tours[key]
            elif second:
                infeasible[key] = tours[key]
            else:
                feasible_2[key] = tours[key]

        return feasible_1, feasible_2, infeasible


    def _fusion(self, simple_graph_1, simple_graph_2, tours, infeasible):
        # Try fusion only if more than 2 partitions exist
        feasible_1 = dict()
        feasible_2 = dict()
        rg1 = dict()
        rg2 = dict()
        tours_aux = dict()

        print tours

        # Create all combinations
        pairs = combinations(infeasible.keys(), 2)
        for i, j in pairs:
            if (i,j) not in feasible_1 and (i,j) not in feasible_2:
                rg1[(i,j)] = Graph(simple_graph_1[i]) | Graph(simple_graph_1[j])
                rg2[(i,j)] = Graph(simple_graph_2[i]) | Graph(simple_graph_2[j])
                print rg1, rg2
                tours_aux[(i, j)] = tours[i] | tours[j]
                feasible_1[(i,j)], feasible_2[(i,j)], _ = \
                self._return_feasible(rg1, rg2, tours)

        print feasible_1, feasible_2



    # Partition Crossover
    def __mul__(self, other):
        # Tours
        tour_1 = list(self._tour)
        tour_2 = list(other._tour)

        foo = Graph.gen_edges_directed(list(reversed(tour_2)))
        if (len(self._edges_directed & other._edges_directed) <
            len(self._edges_directed & foo)):
            tour_2.reverse()

        tour_3 = list(tour_2)

        removed_tour_1 = dict()
        removed_tour_2 = dict()
        removed_tour_3 = dict()

        # Union (G*)
        undirected_union = self._graph_undirected | other._graph_undirected

        for vertice in undirected_union:
            # Create ghost nodes for degree 4 nodes
            if len(undirected_union[vertice]) == 4:
                tour_1.insert(tour_1.index(vertice) + 1, -vertice)
                tour_2.insert(tour_2.index(vertice) + 1, -vertice)
                tour_3.insert(tour_3.index(vertice), -vertice)
            # Remove degree 2 nodes
            if len(undirected_union[vertice]) == 2:
                removed_tour_1[vertice] = tour_1.index(vertice)
                removed_tour_2[vertice] = tour_2.index(vertice)
                removed_tour_3[vertice] = tour_3.index(vertice)
                tour_1.remove(vertice)
                tour_2.remove(vertice)
                tour_3.remove(vertice)

        # Tuples are fast
        tour_1 = tuple(tour_1)
        tour_2 = tuple(tour_2)
        tour_3 = tuple(tour_3)

        #print tour_1
        #print tour_2
        #print tour_3

        # Recreate graphs
        g_undirected_1 = Graph.gen_graph_undirected(tour_1)
        g_undirected_2 = Graph.gen_graph_undirected(tour_2)
        g_undirected_3 = Graph.gen_graph_undirected(tour_3)

        # Partitioning schemes a and b
        vertices_a, tours_a = self._partition(g_undirected_1 ^ g_undirected_2)
        vertices_b, tours_b = self._partition(g_undirected_1 ^ g_undirected_3)

        # Generate simple graphs for each partitioning scheme for each tour
        simple_graph_a_1 = self._gen_simple_graph(vertices_a, tour_1)
        simple_graph_a_2 = self._gen_simple_graph(vertices_a, tour_2)

        simple_graph_b_1 = self._gen_simple_graph(vertices_b, tour_1)
        simple_graph_b_3 = self._gen_simple_graph(vertices_b, tour_3)

        feasible_1_a, feasible_2_a, infeasible_a = \
            self._return_feasible(simple_graph_a_1, simple_graph_a_2,
                                  tours_a)

        feasible_1_b, feasible_2_b, infeasible_b = \
            self._return_feasible(simple_graph_b_1, simple_graph_b_3,
                                  tours_b)

        score_a = (len(feasible_1_a) + len(feasible_2_a) +
                   len(infeasible_a)/2.1)

        score_b = (len(feasible_1_b) + len(feasible_2_b) +
                   len(infeasible_b)/2.1)

        # Choose better partitioning scheme
        if score_a >= score_b:
            feasible_1 = feasible_1_a
            feasible_2 = feasible_2_a
            infeasible = infeasible_a
            vertices = vertices_a
            tours = tours_a
            simple_graph_1 = simple_graph_a_1
            simple_graph_2 = simple_graph_a_2
        else:
            feasible_1 = feasible_1_b
            feasible_2 = feasible_2_b
            infeasible = infeasible_b
            vertices = vertices_b
            tours = tours_b
            tour_2 = tour_3
            simple_graph_1 = simple_graph_b_1
            simple_graph_2 = simple_graph_b_3

        print "Tour 1: ", tour_1
        print "Tour 2: ", tour_2
        print
        print simple_graph_1
        print simple_graph_2
        print
        print "Feasible 1: ", feasible_1
        print "Feasible 2: ", feasible_2
        print "Infeasible: ", infeasible

        self._fusion(simple_graph_1, simple_graph_2, tours, infeasible)

        return (feasible_1, feasible_2, infeasible, simple_graph_1,
                simple_graph_2, tours, tour_1, tour_2)

def test(size, limit):
    for x in xrange(limit):
        p1 = Chromosome(size)
        p2 = Chromosome(size)
        f1, f2, infeasible, rg1, rg2, tours, tour_1, tour_2 = p1 * p2
        print '\r', x,
        #if any(x < 0 for x in tour_1):
        #    continue
        if len(f2) and len(rg1) == 5:
            print
            print "Count: ", x
            print "Tour 1: ", tour_1
            print "Tour 2: ", tour_2
            print
            print "Feasible 1: ", f1
            print "Feasible 2: ", f2
            print "Infeasible: ", infeasible
            break

#test(12, 1000000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                24,25,26,27,28,29,30,31,32])
p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
                7,8,5,6,4,3,22,21,24,23,2])
#p2 = Chromosome([1,2,23,24,21,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,
#                20,25,26,27,28,12,11,31,32])

# Tinos2018b-F5b
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32])
#p2 = Chromosome([1,2,23,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,20,21,
#                24,25,26,27,28,12,11,31,32])

# F1 2i
#p1 = Chromosome((1, -1, 7, -7, 5, 6, 2, 10, 9, -9, 3, 8, 4, -4))
#p2 = Chromosome((3, 8, 9, -9, 7, -7, 10, 2, 4, -4, 5, 6, 1, -1))

#F2 3i
#p1 = Chromosome([1,-1,10,4, 6, -6, 2, -2, 8, -8, 5, -5, 9, -9, 3, -3, 7, -7])
#p2 = Chromosome([1,-1,3,-3, 2, -2, 9, -9, 7, -7, 5, -5, 6, -6, 10, 4, 8, -8])

#F6 2f1
#p1 = Chromosome([1, 3, 6, 7, 10, 4, 8, 11, 2, 5, 9])
#p2 = Chromosome([1, 4, 8, 2, 11, 5, 9, 6, 7, 10, 3])

#F7 2i
#p1 = Chromosome([1, 7, 4, 8, 6, 5, 2, 3])
#p2 = Chromosome([1, 5, 2, 8, 6, 7, 4, 3])

#F8 3i
#p1 = Chromosome([1,2,3,4,5,6,7,8,9])
#p2 = Chromosome([1,8,6,4,2,9,7,5,3])

#F9 1f, 3i (1f, 1f2, 2i)
#p1 = Chromosome((1, -1, 9, -9, 8, -8, 10, -10, 7, -7, 2, -2, 5, -5, 4, -4, 3, -3, 6, -6))
#p2 = Chromosome((1, -1, 8, -8, 6, -6, 2, -2, 3, -3, 10, -10, 5, -5, 9, -9, 4, -4, 7, -7))

#F10 4f2
#p1 = Chromosome((1, 2, 6, -6, 7, -7, 11, -11, 4, 9, 3, 8, 5, -5, 10, 12))
#p2 = Chromosome((1, 11, -11, 8, 3, 7, -7, 12, 10, 6, -6, 9, 4, 5, -5, 2))

#F11 4i
#p1 = Chromosome((1, -1, 2, -2, 9, -9, 7, -7, 10, -10, 12, -12, 3, -3, 6, -6, 5,
#                -5, 4, -4, 11, -11, 8, -8))
#p2 = Chromosome((1, -1, 5, -5, 12, -12, 11, -11, 9, -9, 3, -3, 10, -10, 8, -8,
#                 7, -7, 6, -6, 2, -2, 4, -4))

#F12 5i
#p1 = Chromosome((1,-1,3,-3,10,-10,8,-8,11,-11,2,-2,5,7,6,-6,4,-4,9,-9,12,-12))
#p2 = Chromosome((1,-1,4,-4,2,-2,8,-8,12,-12,6,-6,3,-3,9,-9,11,-11,10,-10,5,7))

p1 * p2


    # Simetric diferente (union - intersection)
    def __xor__(self, other):
        result = Graph()
        for key in self:
            if key in other:
                r = self[key] ^ other[key]
                if r:
                    result[key] = r
            else:
                result[key] = self[key]
        for key in other:
            if key in self:
                r = other[key] ^ self[key]
                if r:
                    result[key]
            else:
                result[key] = other[key]
        return result


# Identify feasible and infeasible partitions by simple graph comparison
def _return_feasible(self, simple_g_1, simple_g_2):

    feasible_1 = set()
    feasible_2 = set()
    infeasible = set()

    for key in simple_g_1:
        # NOTE: It is possible that inverted graphs of invalid partitions
        #       match fist test even with aligned tours
        # Fisrt test (simple graph and inverted simple graph)
        first = bool(simple_g_1[key] == simple_g_2[key] or
                set(simple_g_1[key]) == set(simple_g_2[key].values()) and
                set(simple_g_1[key].values()) == set(simple_g_2[key]))

        # Second test. Any partition that fails first test is invalid only
        # if both simple graphs have identical bags of in/out vertices...
        second = bool(set(simple_g_1[key]) == set(simple_g_2[key]) and
        set(simple_g_1[key].values()) == set(simple_g_2[key].values()))

        # and the number os in/out is even

        # Test conditions
        if first:
            feasible_1.add(key)
        elif second:
            infeasible.add(key)
        else:
            feasible_2.add(key)

    return feasible_1, feasible_2, infeasible




def _fusion(self, simple_g_1, simple_g_2, infeasible):
    # Try fusion only if more than 2 partitions exist
    feasible_1 = set()
    feasible_2 = set()
    rg1 = dict()
    rg2 = dict()

    pairs = combinations(infeasible, 2)

    while pairs:
        if

        if not condition:
            rg1[(i,j)] = Graph(simple_g_1[i]) & Graph(simple_g_1[j])
            rg2[(i,j)] = Graph(simple_g_2[i]) & Graph(simple_g_2[j])

            #tour = self._dfs_rg(rg1[(i,j)], rg1[(i,j)].iteritems().next())

            #print tour

            f1, f2, _ = self._return_feasible(rg1, rg2)
            if f1 or f2:
                infeasible.remove(i)
                infeasible.remove(j)
            feasible_1.update(f1)
            feasible_2.update(f2)

    return feasible_1, feasible_2


#############################################################
#############################################################

#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
from collections import defaultdict
from collections import deque
import numpy as np
from graph import Graph

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
        self._tour = tuple(tour)

        self._undirected_graph = Graph.gen_undirected_graph(self._tour)
        self._directed_edges = Graph.gen_directed_edges(self._tour)

    # Get tour
    def get_tour(self):
        return self._tour

    def get_undirected_graph(self):
        return self._undirected_graph

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start, visited = None, tour = None):
        if visited is None:
            visited = set()
            tour = list()
        visited.add(start)
        tour.append(start)
        for next in graph[start] - visited:
            self._dfs(graph, next, visited, tour)
            return visited, tuple(tour)

    # Depth first search to create simple graph
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs_rg(self, graph, start, rg = None):
        if rg is None:
            rg = dict()
        tour.append(start)
        if start in graph:
            self._dfs_rg(graph, graph[start], tour)
            return tuple(tour)

    # Find partitions using dfs
    def _partition(self, graph):
        vertices = dict()
        tours = dict()
        loop = set(graph)
        index = 1
        while loop:
            vertices[index], tours[index] = self._dfs(graph, loop.pop())
            loop -= vertices[index]
            index += 1
        return vertices, tours

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, partitions, tour):
        simple_tour = defaultdict(deque)
        simple_graph = defaultdict(dict)

        # TODO: Optimize this
        aux_tour = list(tour)
        aux_tour.append(tour[0])
        aux_tour = tuple(aux_tour)

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
                    simple_tour[key].extend([i, j, 'c']) #add cut

        for key in simple_tour:
            # rotate by 'c'
            p = list(reversed(simple_tour[key])).index('c')
            simple_tour[key].rotate(p)
            # Tuples are fast
            simple_tour[key] = tuple(simple_tour[key])
            #print simple_tour[key]
            # Converts permutation to graph
            for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
                if not (i == 'c' or j == 'c'):
                    simple_graph[key][i] = j

        return dict(simple_graph)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, simple_g_1, simple_g_2):

        feasible_1 = set()
        feasible_2 = set()
        infeasible = set()

        for key in simple_g_1:
            # NOTE: It is possible that inverted graphs of invalid partitions
            #       match fist test even with aligned tours
            # Fisrt test (simple graph and inverted simple graph)
            first = bool(simple_g_1[key] == simple_g_2[key] or
                    set(simple_g_1[key]) == set(simple_g_2[key].values()) and
                    set(simple_g_1[key].values()) == set(simple_g_2[key]))

            # Second test. Any partition that fails first test is invalid only
            # if both simple graphs have identical bags of in/out vertices...
            second = bool(set(simple_g_1[key]) == set(simple_g_2[key]) and
            set(simple_g_1[key].values()) == set(simple_g_2[key].values()))

            # and the number os in/out is even
            #l = bool(len(simple_g_1[key]) % 2)

            # Test conditions
            if first:
                feasible_1.add(key)
            elif second:
                infeasible.add(key)
            else:
                feasible_2.add(key)

        return feasible_1, feasible_2, infeasible

    def _fusion(self, simple_g_1, simple_g_2, infeasible):
        # Try fusion only if more than 2 partitions exist
        feasible_1 = set()
        feasible_2 = set()
        graph_1 = dict()
        graph_2 = dict()
        candidates = list()

        # Create all combinations
        for i, j in combinations(infeasible, 2):
            common = len(Graph(simple_g_1[i]) & Graph(simple_g_1[j]))
            candidates.append(tuple([i, j, common]))

        # Sort by common edges
        candidates.sort(key = lambda t: t[2], reverse = True)

        for i, j, d in candidates:
            graph_1[(i,j)] = Graph(simple_g_1[i]) | Graph(simple_g_1[j])
            graph_2[(i,j)] = Graph(simple_g_2[i]) | Graph(simple_g_2[j])

        return feasible_1, feasible_2


    # Partition Crossover
    def __mul__(self, other):
        # Tours
        tour_1 = list(self._tour)
        tour_2 = list(other._tour)

        # Align tours
        foo = Graph.gen_directed_edges(list(reversed(tour_2)))
        if (len(self._directed_edges & other._directed_edges) <
            len(self._directed_edges & foo)):
            tour_2.reverse()

        tour_3 = list(tour_2)

        #print "Tour 1: ", tour_1
        #print "Tour 2: ", tour_2
        #print "Tour 3: ", tour_3

        removed_tour_1 = dict()
        removed_tour_2 = dict()
        removed_tour_3 = dict()

        # Union (G*)
        undirected_union = self._undirected_graph | other._undirected_graph

        for vertice in undirected_union:
            # Create ghost nodes for degree 4 nodes
            if len(undirected_union[vertice]) == 4:
                tour_1.insert(tour_1.index(vertice) + 1, -vertice)
                tour_2.insert(tour_2.index(vertice) + 1, -vertice)
                tour_3.insert(tour_3.index(vertice), -vertice)
            # Remove degree 2 nodes
            if len(undirected_union[vertice]) == 2:
                removed_tour_1[vertice] = tour_1.index(vertice)
                removed_tour_2[vertice] = tour_2.index(vertice)
                removed_tour_3[vertice] = tour_3.index(vertice)
                tour_1.remove(vertice)
                tour_2.remove(vertice)
                tour_3.remove(vertice)

        # Tuples are fast
        tour_1 = tuple(tour_1)
        tour_2 = tuple(tour_2)
        tour_3 = tuple(tour_3)

        #print "Tour 1: ", tour_1
        #print "Tour 2: ", tour_2
        #print "Tour 3: ", tour_3

        # Recreate graphs
        g_undirected_1 = Graph.gen_undirected_graph(tour_1)
        g_undirected_2 = Graph.gen_undirected_graph(tour_2)
        g_undirected_3 = Graph.gen_undirected_graph(tour_3)

        # Partitioning schemes a and b
        vertices_a, tours_a = self._partition(g_undirected_1 ^ g_undirected_2)
        vertices_b, tours_b = self._partition(g_undirected_1 ^ g_undirected_3)

        # Generate simple graphs for each partitioning scheme for each tour
        simple_graph_a_1 = self._gen_simple_graph(vertices_a, tour_1)
        simple_graph_a_2 = self._gen_simple_graph(vertices_a, tour_2)

        simple_graph_b_1 = self._gen_simple_graph(vertices_b, tour_1)
        simple_graph_b_3 = self._gen_simple_graph(vertices_b, tour_3)

        feasible_1_a, feasible_2_a, infeasible_a = \
            self._return_feasible(simple_graph_a_1, simple_graph_a_2)

        feasible_1_b, feasible_2_b, infeasible_b = \
            self._return_feasible(simple_graph_b_1, simple_graph_b_3)

        score_a = (len(feasible_1_a) + len(feasible_2_a) +
                   len(infeasible_a)/2.1)

        score_b = (len(feasible_1_b) + len(feasible_2_b) +
                   len(infeasible_b)/2.1)

        # Choose better partitioning scheme
        if score_a >= score_b:
            feasible_1 = feasible_1_a
            feasible_2 = feasible_2_a
            infeasible = infeasible_a
            vertices = vertices_a
            tours = tours_a
            simple_graph_1 = simple_graph_a_1
            simple_graph_2 = simple_graph_a_2
        else:
            feasible_1 = feasible_1_b
            feasible_2 = feasible_2_b
            infeasible = infeasible_b
            vertices = vertices_b
            tours = tours_b
            tour_2 = tour_3
            simple_graph_1 = simple_graph_b_1
            simple_graph_2 = simple_graph_b_3

        # Fusion
        f1, f2 = self._fusion(simple_graph_1, simple_graph_2, infeasible)
        feasible_1.update(f1)
        feasible_2.update(f2)

        print "Tour 1: ", tour_1
        print "Tour 2: ", tour_2
        print
        print simple_graph_1
        print simple_graph_2
        print
        print tours
        print
        print "Feasible 1: ", feasible_1
        print "Feasible 2: ", feasible_2
        print "Infeasible: ", infeasible

        return (feasible_1, feasible_2, infeasible, simple_graph_1,
                simple_graph_2, tours, tour_1, tour_2)

def test(size, limit):
    for x in xrange(limit):
        p1 = Chromosome(size)
        p2 = Chromosome(size)
        while (p1.get_undirected_graph() == p2.get_undirected_graph()):
            p2 = Chromosome(size)
        f1, f2, infeasible, rg1, rg2, tours, tour_1, tour_2 = p1 * p2
        print '\r', x,
        #if any(x < 0 for x in tour_1):
        #    continue
        if (len(f1) or len(f2)) and len(infeasible) == 1:
            print
            print "Count: ", x
            print "Tour 1: ", tour_1
            print "Tour 2: ", tour_2
            print
            print "Partitions: ", tours
            print
            print "Feasible 1: ", f1
            print "Feasible 2: ", f2
            print "Infeasible: ", infeasible
            break

#test(10, 1000000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

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

# Tinos2018b-F5b
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#                24,25,26,27,28,29,30,31,32])
#p2 = Chromosome([1,2,23,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,20,21,
#                24,25,26,27,28,12,11,31,32])

# F1 2i
#p1 = Chromosome((1, -1, 7, -7, 5, 6, 2, 10, 9, -9, 3, 8, 4, -4))
#p2 = Chromosome((3, 8, 9, -9, 7, -7, 10, 2, 4, -4, 5, 6, 1,-1))

#F2 3i
#p1 = Chromosome([1,-1,10,4,6,-6,2,-2,8,-8,5,-5,9,-9,3,-3,7,-7])
#p2 = Chromosome([1,-1,3,-3,2,-2,9,-9,7,-7,5,-5,6,-6,10,4,8,-8])

#F6 2f1
#p1 = Chromosome([1,3,6,7,10,4,8,11,2,5,9])
#p2 = Chromosome([1,4,8,2,11,5,9,6,7,10,3])

#F7 2i
#p1 = Chromosome([1,7,4,8,6,5,2,3])
#p2 = Chromosome([1,5,2,8,6,7,4,3])

#F8 3i
#p1 = Chromosome([1,2,3,4,5,6,7,8,9])
#p2 = Chromosome([1,8,6,4,2,9,7,5,3])

#F9 1f, 3i (1f, 1f2, 2i)
#p1 = Chromosome((1,-1,9,-9,8,-8,10,-10,7,-7,2,-2,5,-5,4,-4,3,-3,6,-6))
#p2 = Chromosome((1,-1,8,-8,6,-6,2,-2,3,-3,10,-10,5,-5,9,-9,4,-4,7,-7))

#F10 4f2
#p1 = Chromosome((1,2,6,-6,7,-7,11,-11,4,9,3,8,5,-5,10,12))
#p2 = Chromosome((1,11,-11,8,3,7,-7,12,10,6,-6,9,4,5,-5,2))

#F11 4i (2 fusions)
p1 = Chromosome((1,-1,2,-2,9,-9,7,-7,10,-10,12,-12,3,-3,6,-6,5,
                -5,4,-4,11,-11,8,-8))
p2 = Chromosome((1,-1,5,-5,12,-12,11,-11,9,-9,3,-3,10,-10,8,-8,
                 7,-7,6,-6,2,-2,4,-4))

#F12 5i (one fusion)
#p1 = Chromosome((1,-1,3,-3,10,-10,8,-8,11,-11,2,-2,5,7,6,-6,4,-4,9,-9,12,-12))
#p2 = Chromosome((1,-1,4,-4,2,-2,8,-8,12,-12,6,-6,3,-3,9,-9,11,-11,10,-10,5,7))

#F13
#p1 = Chromosome((1,-1,4,-4,3,-3,9,-9,6,-6,2,-2,8,-8,7,10,5,-5))
#p2 = Chromosome((9,-9,7,10,6,-6,3,-3,8,-8,5,-5,4,-4,2,-2,1,-1))

p1 * p2
##################################################################3
###################################################################
#!/usr/bin/python
# ozeasx@gmail.com

from collections import defaultdict

# Class to provide graph and edge generation and dict operators overload
class Graph(dict):
    # Generates directed edge set
    @staticmethod
    def gen_directed_edges(tour):
        edges = set()
        for i, j in zip(tour[:-1], tour[1:]):
            edges.add(tuple([i, j]))
        # Close circle
        edges.add(tuple([tour[-1], tour[0]]))

        return edges

    # Generates undirected edge set
    @staticmethod
    def gen_undirected_edges(tour):
        edges = set()
        for i, j in zip(tour[:-1], tour[1:]):
            edges.add(tuple(sorted([i, j])))
        # Close circle
        edges.add(tuple(sorted([tour[-1], tour[0]])))

        return edges

    # Generates directed graph
    @staticmethod
    def gen_directed_graph(tour):
        graph = defaultdict(set)
        for i, j in zip(tour[:-1], tour[1:]):
            graph[i].add(j)
        # Close circle
        graph[tour[-1]].add(tour[0])
        return Graph(graph)

    # Generates undirected graph
    @staticmethod
    def gen_undirected_graph(tour):
        graph = defaultdict(set)
        for i, j in zip(tour[:-1], tour[1:]):
            graph[i].add(j)
            graph[j].add(i)

        # Close circle both ways
        graph[tour[-1]].add(tour[0])
        graph[tour[0]].add(tour[-1])

        return Graph(graph)

    # Return vertice degree
    def get_vertice_degree(self, vertice):
        return len(self[vertice])

    # remove vertice from graph
    def pop_vertice(self, vertice):
        # Store subgraph to be poped
        subgraph = dict()
        subgraph[vertice] = self[vertice].copy()

        # Remove vertice
        del self[vertice]

        # Replace path
        for key in subgraph[vertice]:
            self[key].remove(vertice)
            self[key].update(subgraph[vertice])
            self[key].remove(key)

        # Return subgraph
        return subgraph

    def insert_vertice(self, vertice, vertice_a, vertice_b):
        self[vertice_a].update(vertice)
        self[vertice_a].remove(vertice_b)

        self[vertice_b].update(vertice)
        self[vertice_b].remove(vertice_a)



    # Invert graph mapping
    def __neg__(self):
        result = Graph()
        for key, value in self.iteritems():
            result[value] = key
        return result

    # Union
    def __or__(self, other):
        result = Graph()
        for key in self:
            if key in other:
                result[key] = self[key] | other[key]
            else:
                result[key] = self[key]
        for key in other:
            if key in self:
                result[key] = other[key] | self[key]
            else:
                result[key] = other[key]
        return result

    # Intersection
    def __and__(self, other):
        result = Graph()
        if len(self) <= len(other):
            loop = self
        else:
            loop = other
            other = self
        for key in loop:
            if key in other:
                result[key] = self[key] & other[key]
        return result

    # Diference
    def __sub__(self, other):
        result = Graph()
        for key in self:
            if key in other:
                result[key] = self[key] - other[key]
            else:
                result[key] = self[key]
        return result

    # Simetric diferente (union - intersection)
    def __xor__(self, other):
        result = Graph()
        for key in self:
            if key in other:
                r = self[key] ^ other[key]
                if r:
                    result[key] = r
            else:
                result[key] = self[key]
        for key in other:
            if key in self:
                r = other[key] ^ self[key]
                if r:
                    result[key]
            else:
                result[key] = other[key]
        return result

#u1 = Graph.gen_graph([1,2,3,4,5,1])
#u2 = Graph.gen_graph([1,4,3,5,2,1])

#print u1
#print -u1 | u1
#print u1





# Partition Crossover
def __mul__(self, other):

    # Solutions graphs to be manipulated
    undirected_g_1 = copy.deepcopy(self._undirected_graph)
    undirected_g_2 = copy.deepcopy(other._undirected_graph)
    undirected_g_3 = copy.deepcopy(other._undirected_graph)

    # Graphs to store removed vertices
    removed_g_1 = dict()
    removed_g_2 = dict()

    # Union (G*)
    undirected_union = undirected_g_1 | undirected_g_2

    for vertice in undirected_union:
        # Remove degree 2 nodes
        if len(undirected_union[vertice]) == 2:
            removed_g_1.update(undirected_g_1.pop_vertice(vertice))
            removed_g_2.update(undirected_g_2.pop_vertice(vertice))
            undirected_g_3.pop_vertice(vertice)

        # Insert ghost nodes
        if len(undirected_union[vertice]) == 4:
            neighbor, _ = undirected_g_1[vertice]
            undirected_g_1.insert_vertice(vertice, neighbor)

            neighbor_a, neighbor_b = undirected_g_2[vertice]
            undirected_g_2.insert_vertice(vertice, neighbor_a)

            neighbor_a, neighbor_b = undirected_g_3[vertice]
            undirected_g_3.insert_vertice(vertice, neighbor_b)

    # Partitioning schemes a and b
    vertices_a, tours_a = self._partition(undirected_g_1 ^ undirected_g_2)
    vertices_b, tours_b = self._partition(undirected_g_1 ^ undirected_g_3)

    # Generate simple graphs for each partitioning scheme for each tour
    simple_graph_a_1 = self._gen_simple_graph(vertices_a, self._tour)
    simple_graph_a_2 = self._gen_simple_graph(vertices_a, other._tour)

    simple_graph_b_1 = self._gen_simple_graph(vertices_b, self._tour)
    simple_graph_b_3 = self._gen_simple_graph(vertices_b, other._tour)

    feasible_1_a, feasible_2_a, infeasible_a = \
        self._return_feasible(simple_graph_a_1, simple_graph_a_2)

    feasible_1_b, feasible_2_b, infeasible_b = \
        self._return_feasible(simple_graph_b_1, simple_graph_b_3)

    score_a = (len(feasible_1_a) + len(feasible_2_a) +
               len(infeasible_a)/2.1)

    score_b = (len(feasible_1_b) + len(feasible_2_b) +
               len(infeasible_b)/2.1)

    # Choose better partitioning scheme
    if score_a >= score_b:
        feasible_1 = feasible_1_a
        feasible_2 = feasible_2_a
        infeasible = infeasible_a
        vertices = vertices_a
        tours = tours_a
        simple_graph_1 = simple_graph_a_1
        simple_graph_2 = simple_graph_a_2
    else:
        feasible_1 = feasible_1_b
        feasible_2 = feasible_2_b
        infeasible = infeasible_b
        vertices = vertices_b
        tours = tours_b
        tour_2 = other._tour
        simple_graph_1 = simple_graph_b_1
        simple_graph_2 = simple_graph_b_3

    # Fusion
    f1, f2 = self._fusion(simple_graph_1, simple_graph_2, infeasible)
    feasible_1.update(f1)
    feasible_2.update(f2)

    print "Tour 1: ", self._tour
    print "Tour 2: ", other._tour
    print
    print simple_graph_1
    print simple_graph_2
    print
    print tours
    print
    print "Feasible 1: ", feasible_1
    print "Feasible 2: ", feasible_2
    print "Infeasible: ", infeasible

    return (feasible_1, feasible_2, infeasible, simple_graph_1,
            simple_graph_2, tours, self._tour, other._tour)


##########################################################3
############################################################33
# Identify feasible and infeasible partitions by simple graph comparison
def _return_feasible(self, simple_g_1, simple_g_2):

    feasible_1 = set()
    feasible_2 = set()
    infeasible = set()

    for key in simple_g_1:
        # NOTE: It is possible that inverted graphs of invalid partitions
        #       match fist test even with aligned tours
        # Fisrt test (simple graph and inverted simple graph)
        first = bool(simple_g_1[key] == simple_g_2[key] or
                set(simple_g_1[key]) == set(simple_g_2[key].values()) and
                set(simple_g_1[key].values()) == set(simple_g_2[key]))

        # Second test. Any partition that fails first test is invalid only
        # if both simple graphs have identical bags of in/out vertices...
        second = bool(set(simple_g_1[key]) == set(simple_g_2[key]) and
        set(simple_g_1[key].values()) == set(simple_g_2[key].values()))

        # and the number os in/out is even
        l = bool(len(simple_g_1[key]) % 2)

        # and the number of (in/out - common) is even
        l = bool((len(simple_g_1[key]) -
                len(Graph(simple_g_1[key]) & Graph(simple_g_2[key]))) % 2)

        # and the number of (in/out - common) is even
        #third = bool(len(Graph(simple_g_1[key]) ^
        #                 Graph(simple_g_2[key])) % 2)

        # Test conditions
        if first:
            feasible_1.add(key)
        elif second:
            if l:
                feasible_2.add(key)
            else:
                infeasible.add(key)
        else:
            feasible_2.add(key)

    return feasible_1, feasible_2, infeasible



    def _dfs(self, graph, start, visited = None, tour = None):
        if visited is None:
            visited = set()
            tour = list()
        visited.add(start)
        tour.append(start)
        for next in graph[start] - visited:
            self._dfs(graph, next, visited, tour)
            return visited, tuple(tour)

def _dfs_rg(self, graph, start, tour = None):
    if tour is None:
        tour = list()
    tour.append(start)
    if start in graph:
        self._dfs_rg(graph, graph.pop(start), tour)
        graph.update(dict({tour[0]: tour[-1]}))

########################################################################
#########################################################################
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

# Weight to compare feasible and infeasible relative value
WEIGHT = 2.1

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
        self._tour = tuple(tour)

        self._undirected_graph = Graph.gen_undirected_graph(self._tour)
        self._directed_edges = Graph.gen_directed_edges(self._tour)

    # Get tour
    def get_tour(self):
        return self._tour

    # Get undirected graph
    def get_undirected_graph(self):
        return self._undirected_graph

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start):
        visited, stack, tour = set(), [start], list()
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                tour.append(vertex)
                visited.add(vertex)
                stack.extend(graph[vertex] - visited)
        return visited, tuple(tour)

    # Depth first search to create simple graph
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs_rg(self, graph, start, tour = None):
        if tour is None:
            tour = list()
        tour.append(start)
        if start in graph:
            self._dfs_rg(graph, graph.pop(start), tour)
            graph.update(dict({tour[0]: tour[-1]}))

    # Find partitions using dfs
    def _partition(self, graph):
        vertices = dict()
        tours = dict()
        loop = set(graph)
        index = 1
        while loop:
            vertices[index], tours[index] = self._dfs(graph, loop.pop())
            loop -= vertices[index]
            index += 1
        return vertices, tours

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, partitions, tour):
        simple_tour = defaultdict(deque)
        simple_graph = defaultdict(dict)

        # TODO: Optimize this
        aux_tour = list(tour)
        aux_tour.append(tour[0])
        aux_tour = tuple(aux_tour)

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
                    simple_tour[key].extend([i, j, 'c']) #add cut

        # Covert tour to simple graph
        for key in simple_tour:
            # rotate by 'c'
            p = list(reversed(simple_tour[key])).index('c')
            simple_tour[key].rotate(p)
            # Tuples are fast
            simple_tour[key] = tuple(simple_tour[key])
            #print simple_tour[key]
            # Converts permutation to graph
            for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
                if not (i == 'c' or j == 'c'):
                    simple_graph[key][i] = j

        return dict(simple_graph)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, simple_g_1, simple_g_2):

        feasible_1 = set()
        feasible_2 = set()
        infeasible = set()

        for key in simple_g_1:
            # QUESTION: Is it possible that inverted graphs of invalid
            # partitions match fist test even with aligned tours?

            # Fisrt test (simple graph and inverted simple graph equivalence)
            first = bool(simple_g_1[key] == simple_g_2[key] or
                    set(simple_g_1[key]) == set(simple_g_2[key].values()) and
                    set(simple_g_1[key].values()) == set(simple_g_2[key]))

            # Second test. Any partition that fails first test is invalid only
            # if both simple graphs have identical bags of in/out vertices...
            second = bool(set(simple_g_1[key]) == set(simple_g_2[key]) and
            set(simple_g_1[key].values()) == set(simple_g_2[key].values()))

            # Test conditions
            if first:
                feasible_1.add(key)
            elif second:
                infeasible.add(key)
            else:
                feasible_2.add(key)

        return feasible_1, feasible_2, infeasible

    def _fusion(self, simple_g_1, simple_g_2, infeasible):
        feasible_1 = set()
        feasible_2 = set()
        rg_1 = dict()
        rg_2 = dict()
        candidates = list()

        # Create all combinations
        for i, j in combinations(infeasible, 2):
            common = len(Graph(simple_g_1[i]) & Graph(simple_g_1[j]))
            candidates.append(tuple([i, j, common]))

        # Sort by common edges value
        candidates.sort(key = lambda t: t[2], reverse = True)

        for i, j, d in candidates:
            # Test to determine if partition is fused already
            condition = (any(i in t for t in feasible_1) or
                         any(j in t for t in feasible_1) or
                         any(i in t for t in feasible_2) or
                         any(j in t for t in feasible_2))

            if not condition:
                # Infeasible partitions graph union
                rg_1[(i,j)] = Graph(simple_g_1[i]) | Graph(simple_g_1[j])
                rg_2[(i,j)] = Graph(simple_g_2[i]) | Graph(simple_g_2[j])

                # Transform graph 1 to simple graph 1
                vertices = set(rg_1[(i,j)].keys())
                while vertices:
                    self._dfs_rg(rg_1[(i,j)], vertices.pop())
                #print rg_1[(i,j)]

                # Transform graph 2 to simple graph 2
                vertices = set(rg_2[(i,j)].keys())
                while vertices:
                    self._dfs_rg(rg_2[(i,j)], vertices.pop())
                #print rg_2[(i,j)]

                # Check if fusion is feasible
                f1, f2, _ = self._return_feasible(rg_1, rg_2)

                if (i,j) in f1 or (i,j) in f2:
                    infeasible.remove(i)
                    infeasible.remove(j)
                    feasible_1.update(f1)
                    feasible_2.update(f2)

        #for key in rg_1:
        #    print key, rg_1[key], rg_2[key]

        return feasible_1, feasible_2

    def _build(self, feasible_1, feasible_2, infeasible):
        pass

    # Partition Crossover
    def __mul__(self, other):
        # Tours
        tour_1 = list(self._tour)
        tour_2 = list(other._tour)

        # Align tours 2 and 3
        foo = Graph.gen_directed_edges(list(reversed(tour_2)))
        if (len(self._directed_edges & other._directed_edges) <
            len(self._directed_edges & foo)):
            tour_2.reverse()
        tour_3 = list(tour_2)

#        print "Tour 1: ", tour_1
#        print "Tour 2: ", tour_2
#        print "Tour 3: ", tour_3

#        removed_tour_1 = dict()
#        removed_tour_2 = dict()
#        removed_tour_3 = dict()

        # Union (G*)
        undirected_union = self._undirected_graph | other._undirected_graph

        for vertice in undirected_union:
            # Create ghost nodes for degree 4 nodes
            if len(undirected_union[vertice]) == 4:
                tour_1.insert(tour_1.index(vertice) + 1, -vertice)
                tour_2.insert(tour_2.index(vertice) + 1, -vertice)
                tour_3.insert(tour_3.index(vertice), -vertice)
            # Remove degree 2 nodes
            if len(undirected_union[vertice]) == 2:
#                removed_tour_1[vertice] = tour_1.index(vertice)
#                removed_tour_2[vertice] = tour_2.index(vertice)
#                removed_tour_3[vertice] = tour_3.index(vertice)
                tour_1.remove(vertice)
                tour_2.remove(vertice)
                tour_3.remove(vertice)

        # Tuples are fast
        tour_1 = tuple(tour_1)
        tour_2 = tuple(tour_2)
        tour_3 = tuple(tour_3)

        #print "Tour 1: ", tour_1
        #print "Tour 2: ", tour_2
        #print "Tour 3: ", tour_3

        # Recreate graphs
        undirected_g_1 = Graph.gen_undirected_graph(tour_1)
        undirected_g_2 = Graph.gen_undirected_graph(tour_2)
        undirected_g_3 = Graph.gen_undirected_graph(tour_3)

        # Partitioning schemes a and b
        vertices_a, tours_a = self._partition(undirected_g_1 ^ undirected_g_2)
        vertices_b, tours_b = self._partition(undirected_g_1 ^ undirected_g_3)

        # Generate simple graphs for each partitioning scheme for each tour
        simple_graph_a_1 = self._gen_simple_graph(vertices_a, tour_1)
        simple_graph_a_2 = self._gen_simple_graph(vertices_a, tour_2)

        simple_graph_b_1 = self._gen_simple_graph(vertices_b, tour_1)
        simple_graph_b_3 = self._gen_simple_graph(vertices_b, tour_3)

        # Test simple graphs to identify feasible partitions
        feasible_1_a, feasible_2_a, infeasible_a = \
            self._return_feasible(simple_graph_a_1, simple_graph_a_2)

        feasible_1_b, feasible_2_b, infeasible_b = \
            self._return_feasible(simple_graph_b_1, simple_graph_b_3)

        # Score partitions scheme
        score_a = (len(feasible_1_a) + len(feasible_2_a) +
                   len(infeasible_a)/WEIGHT)

        score_b = (len(feasible_1_b) + len(feasible_2_b) +
                   len(infeasible_b)/WEIGHT)

        # Choose better partitioning scheme
        if score_a >= score_b:
            feasible_1 = feasible_1_a
            feasible_2 = feasible_2_a
            infeasible = infeasible_a
            vertices = vertices_a
            tours = tours_a
            simple_graph_1 = simple_graph_a_1
            simple_graph_2 = simple_graph_a_2
        else:
            feasible_1 = feasible_1_b
            feasible_2 = feasible_2_b
            infeasible = infeasible_b
            vertices = vertices_b
            tours = tours_b
            tour_2 = tour_3
            simple_graph_1 = simple_graph_b_1
            simple_graph_2 = simple_graph_b_3

        # Fusion
        f1, f2 = self._fusion(simple_graph_1, simple_graph_2, infeasible)
        feasible_1.update(f1)
        feasible_2.update(f2)

        if False:
            print "Tour 1: ", tour_1
            print "Tour 2: ", tour_2
            print
            print simple_graph_1
            print simple_graph_2
            print
            print tours
            print
            print "Feasible 1: ", feasible_1
            print "Feasible 2: ", feasible_2
            print "Infeasible: ", infeasible

        return (feasible_1, feasible_2, infeasible, simple_graph_1,
                simple_graph_2, tours, tour_1, tour_2)

def test(size, limit):
    for x in xrange(limit):
        p1 = Chromosome(size)
        p2 = Chromosome(size)
        while (p1.get_undirected_graph() == p2.get_undirected_graph()):
            p2 = Chromosome(size)
        f1, f2, infeasible, rg1, rg2, tours, tour_1, tour_2 = p1 * p2
        print '\r', x,
        #if any(x < 0 for x in tour_1):
        #    continue
        if (len(f2) >= 4):
            print
            print "Count: ", x
            print "Tour 1: ", tour_1
            print "Tour 2: ", tour_2
            print
            print "Partitions: ", tours
            print
            print "Feasible 1: ", f1
            print "Feasible 2: ", f2
            print "Infeasible: ", infeasible
            break

test(10, 100000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

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
#p2 = Chromosome((3, 8, 9, 7, 10, 2, 4, 5, 6, 1))

#F2 3i ??????
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

#F10 4f2
#p1 = Chromosome((1,2,6,7,11,4,9,3,8,5,10,12))
#p2 = Chromosome((1,11,8,3,7,12,10,6,9,4,5,2))

#F11 4i (2 fusions) execption
#p1 = Chromosome((1,2,9,7,10,12,3,6,5,4,11,8))
#p2 = Chromosome((1,5,12,11,9,3,10,8,7,6,2,4))

#F12 5i (one fusion)
#p1 = Chromosome((1,3,10,8,11,2,5,7,6,4,9,12))
#p2 = Chromosome((1,4,2,8,12,6,3,9,11,10,5,7))

#F13
#p1 = Chromosome((1,4,3,9,6,2,8,7,10,5))
#p2 = Chromosome((9,7,10,6,3,8,5,4,2,1))

# Teste
#p1 = Chromosome((1,2,3,4,5,6))
#p2 = Chromosome((1,2,3,6,4,5))

# Force test
#p1 = Chromosome(100000)
#p2 = Chromosome(100000)

#p1 * p2





        feasible_1 = set()
        feasible_2 = set()
        infeasible = set()

        for key in simple_g_1:
            # QUESTION: Is it possible that inverted graphs of invalid
            # partitions match fist test even with aligned tours?

            # Fisrt test (simple graph and inverted simple graph equivalence)
            first = bool(simple_g_1[key] == simple_g_2[key] or
                    set(simple_g_1[key]) == set(simple_g_2[key].values()) and
                    set(simple_g_1[key].values()) == set(simple_g_2[key]))

            # Second test. Any partition that fails first test is invalid only
            # if both simple graphs have identical bags of in/out vertices...
            second = bool(set(simple_g_1[key]) == set(simple_g_2[key]) and
            set(simple_g_1[key].values()) == set(simple_g_2[key].values()))

            # Test conditions
            if first:
                feasible_1.add(key)
            elif second:
                infeasible.add(key)
            else:
                feasible_2.add(key)

        return feasible_1, feasible_2, infeasible





                aux_1 = dict()
                aux_2 = dict()

                for k in simple_g_1[key]:
                    aux_1[k] = set([simple_g_1[key][k]])

                for k in simple_g_2[key]:
                    aux_2[k] = set([simple_g_2[key][k]])




    def _fusion(self, simple_g_1, simple_g_2, infeasible):
        feasible_1 = set()
        feasible_2 = set()
        sg_1 = dict()
        sg_2 = dict()
        candidates = list()

        # Create all combinations
        for i, j in combinations(infeasible, 2):
            common = len(Graph(simple_g_1[i]) & Graph(simple_g_1[j]))
            candidates.append(tuple([i, j, common]))

        # Sort by common edges value
        candidates.sort(key = lambda t: t[2], reverse = True)

        for i, j, d in candidates:
            # Test to determine if partition is fused already
            condition = (any(i in t for t in feasible_1) or
                         any(j in t for t in feasible_1) or
                         any(i in t for t in feasible_2) or
                         any(j in t for t in feasible_2))

            if not condition:
                # Infeasible partitions graph union
                sg_1[(i,j)] = Graph(simple_g_1[i]) | Graph(simple_g_1[j])
                sg_2[(i,j)] = Graph(simple_g_2[i]) | Graph(simple_g_2[j])

                print i, j, sg_1[(i,j)]
                print
                print i, j, sg_2[(i,j)]
                print

                # Transform graph 1 to simple graph 1
                vertices = set(sg_1[(i,j)].keys())
                while vertices:
                    self._dfs_sg(sg_1[(i,j)], vertices.pop())
                print sg_1[(i,j)]

                # Transform graph 2 to simple graph 2
                vertices = set(sg_2[(i,j)].keys())
                while vertices:
                    self._dfs_sg(sg_2[(i,j)], vertices.pop())
                print sg_2[(i,j)]

                # Check if fusion is feasible
                f1, f2, _ = self._return_feasible(sg_1, sg_2)

                if (i,j) in f1 or (i,j) in f2:
                    infeasible.remove(i)
                    infeasible.remove(j)
                    feasible_1.update(f1)
                    feasible_2.update(f2)

        #for key in sg_1:
        #    print key, sg_1[key], sg_2[key]

        return feasible_1, feasible_2

    # Depth first search to create simple graph
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs_sg(self, graph, start, tour = None):
        if tour is None:
            tour = list()
        tour.append(start)
        if start in graph:
            self._dfs_sg(graph, graph.pop(start).pop(), tour)
            graph.update(dict({tour[0]: set([tour[-1]])}))




# Create the simple graph for all partions for given tour
def _gen_simple_graph(self, partitions, tour):
    simple_tour = defaultdict(deque)
    simple_graph = defaultdict(dict)

    # TODO: Optimize this
    aux_tour = list(tour)
    aux_tour.append(tour[0])
    aux_tour = tuple(aux_tour)

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
                simple_tour[key].append('c') #add cut

    # Covert tour to simple graph
    for key in simple_tour:
        # rotate by 'c'
        p = list(reversed(simple_tour[key])).index('c')
        simple_tour[key].rotate(p)
        # Tuples are fast
        simple_tour[key] = tuple(simple_tour[key])
        #print simple_tour[key]
        # Converts permutation to graph
        for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
            if not (i == 'c' or j == 'c'):
                t = sorted([i, j])
                if t[0] not in simple_graph[key]:
                    simple_graph[key][t[0]] = set()
                simple_graph[key][t[0]].add(t[1])

    return dict(simple_graph)



# Identify feasible and infeasible partitions by simple graph comparison
def _return_feasible(self, simple_g_1, simple_g_2):

    feasible_1 = set()
    feasible_2 = set()
    infeasible = set()

    for key in simple_g_1:

        if simple_g_1[key] == simple_g_2[key]:
            feasible_1.add(key)
        else:
            sm = Graph(simple_g_1[key]) ^ Graph(simple_g_2[key])

            s = set()
            for k in sm:
                if len(sm[k]) == 1:
                    s |= sm[k]

            if len(s) == 1:
                feasible_2.add(key)
            else:
                infeasible.add(key)

    return feasible_1, feasible_2, infeasible




# Create the simple graph for all partions for given tour
def _gen_simple_graph(self, partitions, tour):
    simple_tour = defaultdict(deque)
    simple_graph = defaultdict(dict)

    # TODO: Optimize this
    aux_tour = list(tour)
    aux_tour.append(tour[0])
    aux_tour = tuple(aux_tour)

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
        # Tuples are fast
        simple_tour[key] = tuple(simple_tour[key])
        #print simple_tour[key]
        # Converts permutation to graph
        simple_graph[key] = defaultdict(set)
        for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
            if not (i == 'c' or j == 'c'):
                simple_graph[key][i].add(j)
                simple_graph[key][j].add(i)
        simple_graph[key] = dict(simple_graph[key])

    return dict(simple_graph)


def _fusion(self, vertices, sga, tour_a, sgb, tour_b, infeasible):

    feasible_1 = set()
    feasible_2 = set()
    fused = set()

    n = 2
    while n < len(infeasible):
        # Create all combinations
        candidates = list()
        for fusion in combinations(infeasible, n):
            # Count common edges
            common = Graph(sga[fusion[0]])
            for i in fusion[1:]:
                common &= Graph(sga[i])

            fusion = list(fusion)
            fusion.append(len(common))
            candidates.append(fusion)

        print candidates

        # Sort by common edges value
        candidates.sort(key = lambda fusion: fusion[n], reverse = True)
        for fusion in candidates:
            fusion.pop(-1)
        candidates = [tuple(fusion) for fusion in candidates]

        print candidates

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

                # Update
                if fusion in f1 or fusion in f2:
                    for i in fusion:
                        infeasible.remove(i)
                        fused.add(i)
                        feasible_1.update(f1)
                        feasible_2.update(f2)

        n += 1


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



===============================================================================

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

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    def _dfs(self, graph, start):
        visited, stack = defaultdict(set), [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited[vertex].update(graph[vertex])
                stack.extend(graph[vertex] - visited.viewkeys())
        return visited

    # Find partitions using dfs
    def _partition(self, graph_a, graph_b):
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), dict()
        # Simetric diference
        graph = graph_a ^ graph_b
        print graph
        # Loop
        loop, index = set(graph), 1

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
            candidates = map(tuple, candidates)

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

    #def _build(self, ab_cycles, feasible_1, feasible_2, infeasible):
    #    feasible_1.update(feasible_2)
    #    for partition in ab_cycles:



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

        #self._test_feasibility(ab_cycles_m, ab_cycles_n)

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
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                24,25,26,27,28,29,30,31,32])
p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
                7,8,5,6,4,3,22,21,24,23,2])
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


================================================================================

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
INFEASIBLE_WEIGHT = 1

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
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), dict()
        # Simetric diference
        graph = graph_a ^ graph_b
        #print graph
        # Loop
        loop, index = set(graph), 1

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

    def _dfs_2(self, graph, start):
        visited, stack = dict(), [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited[vertex] = graph[vertex]
                stack.extend(graph[vertex] - visited.viewkeys())
        return visited

    def _partition_2(self, graph_a, graph_b):
        # AB cycles
        ab_cycles = dict()
        # Simetric diference
        graph = graph_a ^ graph_b
        #print graph
        # Loop
        loop, index = set(graph), 1
        while loop:
            ab_cycles[index] = self._dfs_2(graph, loop.pop())
            loop -= ab_cycles[index].viewkeys()
            index += 1
        return ab_cycles

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, partitions, tour):
        inner_tour = defaultdict(deque)
        inner_graph = defaultdict(dict)
        outer_tour = defaultdict(deque)
        outer_graph = defaultdict(dict)

        # TODO: Optimize this
        aux_tour = list(tour)
        aux_tour.append(tour[0])

        # Identify entrance and exit vertices
        for i, j in zip(aux_tour[:-1], aux_tour[1:]):
            for key in partitions:
                # Entrance
                if i not in partitions[key] and j in partitions[key]:
                    #simple_tour[key].append(j)
                    inner_tour[key].extend([i, j])
                    outer_tour[key].extend([i, j, 'c'])
                # Exit
                if i in partitions[key] and j not in partitions[key]:
                    #simple_tour[key].extend([i, 'c'])
                    inner_tour[key].extend([i, j, 'c'])
                    outer_tour[key].extend([i, j])

        # Covert tour to simple graph
        for key in inner_tour:
            # rotate by 'c'
            p_inner = list(reversed(inner_tour[key])).index('c')
            p_outer = list(reversed(outer_tour[key])).index('c')
            inner_tour[key].rotate(p_inner)
            outer_tour[key].rotate(p_outer)
            #print simple_tour[key]
            inner_tour[key] = list(inner_tour[key])
            outer_tour[key] = list(outer_tour[key])
            inner_graph[key] = defaultdict(set)
            outer_graph[key] = defaultdict(set)
            # Converts permutation to graph
            for i, j in zip(inner_tour[key][:-1], inner_tour[key][1:]):
                if not (i == 'c' or j == 'c'):
                    inner_graph[key][i].add(j)
                    inner_graph[key][j].add(i)
                    outer_graph[key][i].add(j)
                    outer_graph[key][j].add(i)

        return dict(inner_graph), dict(outer_graph)

    # Identify feasible and infeasible partitions by simple graph comparison
    def _return_feasible(self, inner_a, inner_b, outer_a = None, outer_b = None):

        feasible_1 = set()
        feasible_2 = set()
        infeasible = set()

        for key in inner_a:
            if inner_a[key] == inner_b[key]:
                feasible_1.add(key)
            else:
                test_1 = Graph(inner_a[key]) | Graph(outer_b[key])
                test_2 = Graph(inner_b[key]) | Graph(outer_a[key])
                ham_1 = self._dfs_2(test_1, next(iter(test_1.keys())))
                ham_2 = self._dfs_2(test_2, next(iter(test_2.keys())))
                print ham_1, key
                print ham_2, key
                if len(ham_1) == 1 and len(ham_2) == 1:
                    feasible_2.add(key)
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
            candidates = map(tuple, candidates)

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

    #def _build(self, ab_cycles, feasible_1, feasible_2, infeasible):
    #    feasible_1.update(feasible_2)
    #    for partition in ab_cycles:

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
        ab_cycles_m = self._partition_2(undirected_a, undirected_b)
        ab_cycles_n = self._partition_2(undirected_a, undirected_c)

        # Generate simple graphs for each partitioning scheme for each tour
        inner_graph_a_m, outer_graph_a_m = \
            self._gen_simple_graph(ab_cycles_m, tour_a)
        inner_graph_b_m, outer_graph_b_m = \
            self._gen_simple_graph(ab_cycles_n, tour_b)

        inner_graph_a_n, outer_graph_a_n = \
            self._gen_simple_graph(ab_cycles_n, tour_a)
        inner_graph_c_n, outer_graph_c_n = \
            self._gen_simple_graph(ab_cycles_n, tour_c)

        # Test simple graphs to identify feasible partitions
        feasible_1_m, feasible_2_m, infeasible_m = \
            self._return_feasible(inner_graph_a_m, inner_graph_b_m,
                                  outer_graph_a_m, outer_graph_b_m)

        feasible_1_n, feasible_2_n, infeasible_n = \
            self._return_feasible(inner_graph_a_n, inner_graph_c_n,
                                  outer_graph_a_n, outer_graph_c_n)

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
            ab_cycles = ab_cycles_m
            inner_graph_a = inner_graph_a_m
            outer_graph_a = outer_graph_a_m
            inner_graph_b = inner_graph_b_m
            oter_graph_b = outer_graph_b_m
        else:
            feasible_1 = feasible_1_n
            feasible_2 = feasible_2_n
            infeasible = infeasible_n
            ab_cycles = ab_cycles_n
            tour_b = tour_c
            inner_graph_a = inner_graph_a_n
            outer_graph_a = outer_graph_a_n
            inner_graph_b = inner_graph_c_n
            outer_graph_b = outer_graph_c_n

        # Fusion
        #f1, f2 = self._fusion(vertices, simple_graph_a, tour_a, simple_graph_b,
        #                      tour_b, infeasible)
        #feasible_1.update(f1)
        #feasible_2.update(f2)

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

        return (feasible_1, feasible_2, infeasible, inner_graph_a,
                inner_graph_b, ab_cycles, tour_a, tour_b)

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
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                24,25,26,27,28,29,30,31,32])
p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
                7,8,5,6,4,3,22,21,24,23,2])
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

================================================================================
Versão Boa:

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
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), dict()
        # Simetric diference
        graph = graph_a ^ graph_b
        print graph
        # Loop
        loop, index = set(graph), 1
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
            candidates = map(tuple, candidates)

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

    #def _build(self, ab_cycles, feasible_1, feasible_2, infeasible):
    #    feasible_1.update(feasible_2)
    #    for partition in ab_cycles:



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

        #self._test_feasibility(ab_cycles_m, ab_cycles_n)

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
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                24,25,26,27,28,29,30,31,32])
p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
                7,8,5,6,4,3,22,21,24,23,2])
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


===============================================================================
Parte final

#        if True:
#            print "Tour 1: ", tour_a
#            print "Tour 2: ", tour_b
#            print
#            print simple_graph_1
#            print simple_graph_2
#            print
#            print ab_cycles
#            print
#            print "Feasible 1: ", feasible_1
#            print "Feasible 2: ", feasible_2
#            print "Infeasible: ", infeasible

#        return (feasible_1, feasible_2, infeasible, simple_graph_a,
#                simple_graph_b, ab_cycles, tour_a, tour_b)

#def test(size, limit):
#    for x in xrange(limit):
#        p1 = Chromosome(size)
#        p2 = Chromosome(size)
#        while (p1.get_undirected_graph() == p2.get_undirected_graph()):
#            p2 = Chromosome(size)
#        f1, f2, infeasible, sga, sgb, ab_cycles, tour_a, tour_b = p1 * p2
#        print '\r', x,
        #if any(x < 0 for x in tour_a):
        #    continue
#        if (len(f2) >= 4):
#            print
#            print "Count: ", x
#            print "Tour 1: ", tour_a
#            print "Tour 2: ", tour_b
#            print
#            print "Partitions: ", ab_cycles
#            print
#            print "Feasible 1: ", f1
#            print "Feasible 2: ", f2
#            print "Infeasible: ", infeasible
#            break

#test(1000, 1000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

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

#F2 3i (2f1)
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

#F11 4i (2 fusions f2) execption
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

#p1 * p2
====================================
    def _build(self, ab_cycles, feasible_1, feasible_2, infeasible):

        # Distances of each partition in each solution
        dists_a = dict()
        dists_b = dict()

        # Get distance of all partitions tours
        for key in feasible_1 | feasible_2 | infeasible:
            dists_a[key] = tsp.get_ab_cycle_length(ab_cycles[key])
            ab_cycles[key].rotate(-1)
            dists_b[key] = tsp.get_ab_cycle_length(ab_cycles[key])
            ab_cycles[key].rotate(1)

        sum_a = sum(dists_a.values())
        sum_b = sum(dists_b.values())

        common = tsp.get_route_length(self.get_tour()) - sum_a
        solution = deque()

        for key in feasible_1 | feasible_2:
            if dists_a[key] <= dists_b[key]:
                solution.extend(ab_cycles[key])
            else:
                ab_cycles[key].rotate(-1)
                solution.extend(ab_cycles[key])
                ab_cycles[key].rotate(1)

        print solution



#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
from collections import defaultdict
from collections import deque
import numpy as np
from graph import Graph
from tsp import TSPLIB
from shell import Shell

# Infeasible value compared to feasible partitions
INFEASIBLE_WEIGHT = 0.4

cmd = Shell()
tsp = TSPLIB("eil76", cmd)

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

        # undirected graph representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)

        # Calc solution length
        self._length = tsp.get_route_length(self._tour)

    # == method overload
    def __eq__(self, other):
        return self._undirected_graph == other._undirected_graph

    # Get tour
    def get_tour(self):
        return self._tour

    # Get undirected graph
    def get_undirected_graph(self):
        return self._undirected_graph

    def get_length(self):
        return self._length

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
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), dict()
        # Simetric diference
        graph = graph_a ^ graph_b
        # Loop
        loop, index = set(graph), 1
        while loop:
            vertices[index], ab_cycles[index] = self._dfs(graph, loop.pop())
            # Normalize AB cycles to begin with solution A
            if ab_cycles[index][0] in graph_b:
                if ab_cycles[index][1] in graph_b[ab_cycles[index][0]]:
                    ab_cycles[index].rotate(-1)
            #ab_cycles[index] = list(ab_cycles[index])
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
                    simple_tour[key].extend([i, j, 'c'])

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

    def _fusion(self, vertices, ab_cycles, sg_a, sg_b, tour_a, tour_b,
                infeasible):

        # Feasible 1 partitions fused
        feasible_1 = set()
        # Feasible 2 partitions fused
        feasible_2 = set()
        # Fused partitions
        fused = set()

        # Start fusion with 2 partitions
        n = 2
        while n < len(infeasible):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(infeasible, n):
                # Count common edges
                count = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    count += len(Graph(sg_a[i]) & Graph(sg_a[j]))

                # Create element with (fusion, count)
                fusion = list(fusion)
                fusion.append(count)
                candidates.append(fusion)

            # Sort by common edges count
            candidates.sort(key = lambda fusion: fusion[n], reverse = True)
            # Discard common edges count
            for fusion in candidates:
                fusion.pop(-1)
            # Convert elements to tuples
            candidates = map(tuple, candidates)

            # Try fusions
            for fusion in candidates:
                union = defaultdict(set)
                # Test to determine if partition is fused already
                if not any(i in fused for i in fusion):
                    for i in fusion:
                        union[fusion] |= vertices[i]

                    # Create simple graphs for fusion
                    sga = self._gen_simple_graph(union, tour_a)
                    sgb = self._gen_simple_graph(union, tour_b)

                    # Check if fusion is feasible
                    f1, f2, _ = self._return_feasible(sga, sgb)

                    # Update information
                    if fusion in f1 or fusion in f2:
                        #ab_cycles[fusion] = list()
                        for i in fusion:
                            #ab_cycles[fusion].append(ab_cycles[i])
                            #del ab_cycles[i]
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

    def _build(self, ab_cycles, feasible_1, feasible_2, infeasible):

        # Distances of each partition in each solution
        dists_a = defaultdict(float)
        dists_b = defaultdict(float)

        print ab_cycles

        # Get distance of all partitions tours
        for key in feasible_1 | feasible_2 | infeasible:
            if isinstance(key, tuple):
                for i in key:
                    dists_a[key] += tsp.get_ab_cycle_length(ab_cycles[i])
                    ab_cycles[i].rotate(-1)
                    dists_b[key] += tsp.get_ab_cycle_length(ab_cycles[i])
                    ab_cycles[i].rotate(1)
            else:
                dists_a[key] += tsp.get_ab_cycle_length(ab_cycles[key])
                ab_cycles[key].rotate(-1)
                dists_b[key] += tsp.get_ab_cycle_length(ab_cycles[key])
                ab_cycles[key].rotate(1)

        sum_a = sum(dists_a.values())
        sum_b = sum(dists_b.values())

        common = tsp.get_route_length(self.get_tour()) - sum_a
        solution = deque()

        for key in feasible_1 | feasible_2:
            if isinstance(key, tuple):
                for i in key:
                    if dists_a[key] <= dists_b[key]:
                        solution.extend(ab_cycles[i])
                    else:
                        ab_cycles[i].rotate(-1)
                        solution.extend(ab_cycles[i])
                        ab_cycles[i].rotate(1)
            else:
                if dists_a[key] <= dists_b[key]:
                    solution.extend(ab_cycles[key])
                else:
                    ab_cycles[key].rotate(-1)
                    solution.extend(ab_cycles[key])
                    ab_cycles[key].rotate(1)

        print solution


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
            # Remove degree 2 nodes (surrogate edge)
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
        f1, f2 = self._fusion(vertices, ab_cycles, simple_graph_a,
                              simple_graph_b, tour_a, tour_b, infeasible)
        feasible_1.update(f1)
        feasible_2.update(f2)

        # Buil solutions
        self._build(ab_cycles, feasible_1, feasible_2, infeasible)

        if False:
            print "Tour 1: ", tour_a
            print "Tour 2: ", tour_b
            print
            print simple_graph_a
            print simple_graph_b
            print
            print ab_cycles
            print
            print "Feasible 1: ", feasible_1
            print "Feasible 2: ", feasible_2
            print "Infeasible: ", infeasible

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

# Tinos2014-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
#p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                24,25,26,27,28,29,30,31,32])
p2 = Chromosome([1,32,31,11,12,28,27,26,25,20,19,17,18,15,16,14,13,29,30,10,9,
                7,8,5,6,4,3,22,21,24,23,2])
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

#F2 3i (2f1)
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

#F11 4i (2 fusions f2) execption
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
================================================================================

#!/usr/bin/python
# ozeasx@gmail.com

import numpy as np
from itertools import combinations
from collections import defaultdict
from collections import deque
from graph import Graph
from tsp import TSPLIB
from shell import Shell

# Infeasible value compared to feasible partitions
INFEASIBLE_WEIGHT = 0.4

cmd = Shell()
tsp = TSPLIB("eil76", cmd)

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

        # Number of cities
        self._size = len(tour)
        # Tourlength
        self._len = None

        # undirected graph representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)

    # == method overload
    def __eq__(self, other):
        return self._undirected_graph == other._undirected_graph

    # Get tour
    def get_tour(self):
        return self._tour

    # Get size
    def get_size(self):
        return self._size

    # Get tour length
    def get_len(self):
        if not self._len:
            self._len = tsp.get_tour_len(self._tour)
        return self._len

    # Set tour length
    def set_len(self, value):
        if abs(value - tsp.get_tour_len(self._tour)) > 0.05:
            print value, tsp.get_tour_len(self._tour)
            print "We have a problem!"
            exit()
        self._len = value

    # Find partitions using dfs
    def _partition(self, graph_a, graph_b):
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), defaultdict(dict)
        # Simetric diference
        graph = graph_a ^ graph_b
        # Loop
        loop, index = set(graph), 1
        while loop:
            vertices[index], ab_cycles['A'][index] = Graph.dfs(graph, loop.pop())
            ab_cycles['B'][index] = deque(ab_cycles['A'][index])
            # Fork AB_cycles
            if ab_cycles['A'][index][0] in graph_b:
                if ab_cycles['A'][index][1] in graph_b[ab_cycles['A'][index][0]]:
                    ab_cycles['A'][index].rotate(1)
                else:
                    ab_cycles['B'][index].rotate(1)
            # Reduce loop
            loop -= vertices[index]
            # Increment index
            index += 1
        # Return
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
                    simple_tour[key].extend([i, j, 'c'])

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

    # Try fusion of infeasible partitions
    def _fusion(self, vertices, ab_cycles, sg_a, sg_b, tour_a, tour_b,
                feasible_1, feasible_2, infeasible):

        # Fused partitions
        fused = set()

        # Start fusion try with 2 partitions
        n = 2
        while n < len(infeasible):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(infeasible, n):
                # Count common edges
                count = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    count += len(Graph(sg_a[i]) & Graph(sg_a[j]))

                # Create element with (fusion, count)
                fusion = list(fusion)
                fusion.append(count)
                candidates.append(fusion)

            # Sort by common edges count
            candidates.sort(key = lambda fusion: fusion[n], reverse = True)
            # Discard common edges count
            for fusion in candidates:
                fusion.pop(-1)
            # Convert elements to tuples
            candidates = map(tuple, candidates)

            # Try fusions
            for fusion in candidates:
                union = defaultdict(set)
                # Test to determine if partition is fused already
                if not any(i in fused for i in fusion):
                    for i in fusion:
                        union[fusion] |= vertices[i]

                    # Create simple graphs for fusion
                    sga = self._gen_simple_graph(union, tour_a)
                    sgb = self._gen_simple_graph(union, tour_b)

                    # Check if fusion is feasible
                    f1, f2, _ = self._return_feasible(sga, sgb)

                    # Update information
                    if fusion in f1 or fusion in f2:
                        ab_cycles['A'][fusion] = deque()
                        ab_cycles['B'][fusion] = deque()
                        for i in fusion:
                            ab_cycles['A'][fusion].extend(ab_cycles['A'][i])
                            ab_cycles['B'][fusion].extend(ab_cycles['B'][i])
                            infeasible.remove(i)
                            fused.add(i)
                            feasible_1.update(f1)
                            feasible_2.update(f2)

            # Increment fusion size
            n += 1

        # Fuse all remaining partitions
        if len(infeasible) > 1:
            fusion = tuple(infeasible)
            ab_cycles['A'][fusion] = deque()
            ab_cycles['B'][fusion] = deque()
            for i in fusion:
                ab_cycles['A'][fusion].extend(ab_cycles['A'][i])
                ab_cycles['B'][fusion].extend(ab_cycles['B'][i])
            feasible_1.add(fusion)
            infeasible.clear()
        # Remaining partition is feasilbe 2?
        else:
            feasible_2.update(infeasible)
            infeasible.clear()

    # Build solutions
    def _build_solutions(self, ab_cycles, feasible_1, feasible_2, common,
                         tour_a, tour_b):

        # dists of each partition in each solution
        dists = dict()
        dists['A'] = defaultdict(float)
        dists['B'] = defaultdict(float)

        # Set to store partial solution (cycle, key)s
        solution = set()

        # Partition with minor diff
        minor = {'key': None, 'value': None}

        # Get distance of all partitions tours
        for key in feasible_1 | feasible_2:
            dists['A'][key] += tsp.get_ab_cycle_len(ab_cycles['A'][key])
            dists['B'][key] += tsp.get_ab_cycle_len(ab_cycles['B'][key])

            # Distance diference inside AB_cycle
            diff = abs(dists['A'][key] - dists['B'][key])

            # Store AB_cycle with minor diference
            if not minor['key']:
                minor['key'] = key
                minor['value'] = diff
            elif diff < minor['value']:
                minor['key'] = key
                minor['value'] = diff

            # Chose best edges in each partition
            if dists['A'][key] <= dists['B'][key]:
                solution.add(tuple(['A', key]))
            else:
                solution.add(tuple(['B', key]))

        # Create each solution
        solution_1 = list()
        solution_2 = list()
        sum_1 = 0

        for cycle, key in solution:
            solution_1.extend(ab_cycles[cycle][key])
            sum_1 += dists[cycle][key]
            if key != minor['key']:
                solution_2.extend(ab_cycles[cycle][key])
            elif cycle == 'A':
                solution_2.extend(ab_cycles['B'][minor['key']])
            else:
                solution_2.extend(ab_cycles['A'][minor['key']])

        # Total sum of one partial solution
        sum_A = sum(dists['A'].values())

        # Calc common edges distance
        common_dist = self.get_len() - sum_A

        # Create solutions graphs
        graph_1 = defaultdict(set)
        graph_2 = defaultdict(set)

        # Create solution 1 undirected graph
        for i, j in zip(solution_1[0::2], solution_1[1::2]):
            graph_1[abs(i)].add(abs(j))
            graph_1[abs(j)].add(abs(i))

        # Create solution 1 undirected graph
        for i, j in zip(solution_2[0::2], solution_2[1::2]):
            graph_2[abs(i)].add(abs(j))
            graph_2[abs(j)].add(abs(i))

        # Common graph and solutions graph union
        graph_1 = Graph(graph_1) | Graph(common)
        graph_2 = Graph(graph_2) | Graph(common)

        # Create tours from solutions graph
        vertices_1, tour_1 = Graph.dfs(graph_1, 1)
        vertices_2, tour_2 = Graph.dfs(graph_2, 1)

        # Verify infeasible tours
        if len(vertices_1) < self._size:
            print "tour_1 infeasible", tour_1
            print tour_a
            print tour_b
            print ab_cycles['A']
            print graph_1
            exit()
        else:
            c_1 = Chromosome(tour_1)
            c_1.set_len(common_dist + sum_1)

        if len(vertices_2) < self._size:
            print "tour_2 infeasible", tour_2
            print tour_a
            print tour_b
            print ab_cycles['A']
            print graph_2
            exit()
        else:
            c_2 = Chromosome(tour_2)
            c_2.set_len(common_dist + sum_1 + minor['value'])

        return c_1, c_2

    # Partition Crossover
    def __mul__(self, other):
        # Toursinfeasible.clear()
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
            # Remove degree 2 nodes (surrogate edge)
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

        # If exists one or no partition, return parents
        if len(vertices_m) <= 1 and len(vertices_n) <= 1:
            return self, other

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

        # Try to fuse infeasible partitions
        self._fusion(vertices, ab_cycles, simple_graph_a, simple_graph_b,
                     tour_a, tour_b, feasible_1, feasible_2, infeasible)

        # After fusion, if exists one or no partition, return parents
        if len(feasible_1) + len(feasible_2) + len(infeasible) <= 1:
            return self, other

        # Common edges
        common = self._undirected_graph & other._undirected_graph

        if False:
            print "Tour 1: ", tour_a
            print "Tour 2: ", tour_b
            print
            print simple_graph_a
            print simple_graph_b
            print
            print ab_cycles['A']
            print
            print "Feasible 1: ", feasible_1
            print "Feasible 2: ", feasible_2
            print "Infeasible: ", infeasible

        # Buil solutions
        return self._build_solutions(ab_cycles, feasible_1, feasible_2, common,
                                     tour_a, tour_b)

def test(size, limit):
    for x in xrange(limit):
        p1 = Chromosome(size)
        p2 = Chromosome(size)
        while (p1 == p2):
            p2 = Chromosome(size)
        c_1, c_2 = p1 * p2
        print '\r', x,

test(76, 1000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

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

#F2 3i (2f1)
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

#F11 4i (2 fusions f2) execption
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

# F15 (1f1, 1f2)
#p1 = Chromosome((1, 8, 4, 5, 6, 7, 9, 2, 10, 3))
#p2 = Chromosome((1, 6, 9, 8, 3, 7, 5, 10, 4, 2))

# F16
#p1 = Chromosome((1, 8, 10, 2, 6, 9, 5, 3, 7, 4))
#p2 = Chromosome((1, 4, 5, 2, 3, 9, 6, 10, 8, 7))

# Teste
#p1 = Chromosome((1,2,3,4,5,6))
#p2 = Chromosome((1,2,3,6,4,5))

# Force test
#p1 = Chromosome(100000)
#p2 = Chromosome(100000)

# Minimal exemple
#p1 = Chromosome((1,2,3,4))
#p2 = Chromosome((1,3,2,4))

#p1 * p2


    def two_opt(self):
        improved = True
        best_route = self._tour
        best_distance = self.get_len()
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for k in range(i+1, len(best_route)):
                    # Get joint distances
                    join_a = tsp.ab_cycle_len(best_route[i-1:i+1] +
                                              best_route[k:k+2])
                    join_b = tsp.ab_cycle_len([best_route[i], best_route[k],
                                              best_route[i], best_route[k+1]])
                    if join_b < join_a:
                        new_route = best_route[0:i]
    	                new_route.extend(reversed(best_route[i:k + 1]))
    	                new_route.extend(best_route[k+1:])
                        best_distance = best_distance - join_a + join_b
                        improved = True
                        break #improvement found, return to the top of the while loop
                if improved:
                    break
        assert len(best_route) == len(route)
        print best_route


join_a = tsp.ab_cycle_len([best_route[i-1], best_route[i],
                          best_route[j], best_route[j+1]])
join_b = tsp.ab_cycle_len([best_route[i-1], best_route[j],
                          best_route[i], best_route[j+1]])


def two_opt(self):
    best_route = self._tour
    best_len = self.get_len()
    improved = True
    while improved:
        improved = False
        for i in range(len(best_route)-1):
            for j in range(i+1, len(best_route)):
                if j-i == self._size - 1: continue
                # Measure joint edges
                print i, j
                print [best_route[i-1], best_route[i], best_route[j], best_route[(j+1)%self._size]]
                print [best_route[i-1], best_route[j], best_route[i], best_route[(j+1)%self._size]]
                join_a = tsp.ab_cycle_len([best_route[i-1], best_route[i],
                                          best_route[j],
                                          best_route[(j+1)%self._size]])

                join_b = tsp.ab_cycle_len([best_route[i-1], best_route[j],
                                          best_route[i],
                                          best_route[(j+1)%self._size]])
                # Verify if new joint is shorter
                if join_b < join_a:
                    new_route = best_route[0:i]
                    new_route.extend(reversed(best_route[i:j + 1]))
                    new_route.extend(best_route[j+1:])
                    best_len = best_len - join_a + join_b
                    best_route = new_route
                    improved = True
                    print new_route
                    break

            if improved:
                break

    p = best_route.index(1)
    best_route = deque(best_route)
    best_route.rotate(-p)
    self._tour = list(best_route)
    self.set_len(best_len)

#!/usr/bin/python
# ozeasx@gmail.com

import time
import sys
import numpy as np
from itertools import combinations
from collections import defaultdict
from collections import deque
from graph import Graph
from tsp import TSPLIB
from shell import Shell

# Infeasible value compared to feasible partitions
INFEASIBLE_WEIGHT = 0.4

cmd = Shell()
tsp = TSPLIB("d493", cmd)

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
        # Number of cities
        self._size = len(tour)
        # Tourlength
        self._len = None
        # undirected graph representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)

    # == method overload
    def __eq__(self, other):
        return self._undirected_graph == other._undirected_graph

    # Get tour
    def get_tour(self):
        return self._tour

    # Get size
    def get_size(self):
        return self._size

    # Get tour length
    def get_len(self):
        if not self._len:
            self._len = tsp.route_len_2(self._tour + [1])
        return self._len

    # Set tour length
    def set_len(self, value):
        #print value, tsp.route_len(self._tour + [1])
        assert abs(value - tsp.route_len_2(self._tour + [1])) < 0.05
        self._len = value

    # Find partitions using dfs
    def _partition(self, graph_a, graph_b):
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), defaultdict(dict)
        # Simetric diference
        graph = graph_a ^ graph_b
        # Loop
        loop, index = set(graph), 1
        while loop:
            vertices[index], ab_cycles['A'][index] = Graph.dfs(graph, loop.pop())
            ab_cycles['B'][index] = deque(ab_cycles['A'][index])
            # Fork AB_cycles
            if ab_cycles['A'][index][0] in graph_b:
                if ab_cycles['A'][index][1] in graph_b[ab_cycles['A'][index][0]]:
                    ab_cycles['A'][index].rotate(1)
                else:
                    ab_cycles['B'][index].rotate(1)
            # Reduce loop
            loop -= vertices[index]
            # Increment index
            index += 1
        # Return
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
                    simple_tour[key].extend([i, j, 'c'])

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

    # Try fusion of infeasible partitions
    def _fusion(self, vertices, ab_cycles, sg_a, sg_b, tour_a, tour_b,
                feasible_1, feasible_2, infeasible):

        # Fused partitions
        fused = set()

        # Start fusion try with 2 partitions
        n = 2
        while n < len(infeasible):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(infeasible, n):
                # Count common edges
                count = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    count += len(Graph(sg_a[i]) & Graph(sg_a[j]))

                # Create element with (fusion, count)
                fusion = list(fusion)
                fusion.append(count)
                candidates.append(fusion)

            # Sort by common edges count
            candidates.sort(key = lambda fusion: fusion[n], reverse = True)
            # Discard common edges count
            for fusion in candidates:
                fusion.pop(-1)
            # Convert elements to tuples
            candidates = map(tuple, candidates)

            # Try fusions
            for fusion in candidates:
                union = defaultdict(set)
                # Test to determine if partition is fused already
                if not any(i in fused for i in fusion):
                    for i in fusion:
                        union[fusion] |= vertices[i]

                    # Create simple graphs for fusion
                    sga = self._gen_simple_graph(union, tour_a)
                    sgb = self._gen_simple_graph(union, tour_b)

                    # Check if fusion is feasible
                    f1, f2, _ = self._return_feasible(sga, sgb)

                    # Update information
                    if fusion in f1 or fusion in f2:
                        ab_cycles['A'][fusion] = deque()
                        ab_cycles['B'][fusion] = deque()
                        for i in fusion:
                            ab_cycles['A'][fusion].extend(ab_cycles['A'][i])
                            ab_cycles['B'][fusion].extend(ab_cycles['B'][i])
                            infeasible.remove(i)
                            fused.add(i)
                            feasible_1.update(f1)
                            feasible_2.update(f2)

            # Increment fusion size
            n += 1

        # Fuse all remaining partitions
        if len(infeasible) > 1:
            fusion = tuple(infeasible)
            ab_cycles['A'][fusion] = deque()
            ab_cycles['B'][fusion] = deque()
            for i in fusion:
                ab_cycles['A'][fusion].extend(ab_cycles['A'][i])
                ab_cycles['B'][fusion].extend(ab_cycles['B'][i])
            feasible_1.add(fusion)
            infeasible.clear()
        # Remaining partition is feasilbe 2?
        else:
            feasible_2.update(infeasible)
            infeasible.clear()

    # Build solutions
    def _build_solutions(self, ab_cycles, feasible_1, feasible_2, common,
                         tour_a, tour_b):

        # dists of each partition in each solution
        dists = dict()
        dists['A'] = defaultdict(float)
        dists['B'] = defaultdict(float)

        # Set to store partial solution (cycle, key)s
        solution = set()

        # Partition with minor diff
        minor = {'key': None, 'value': None}

        # Get distance of all partitions tours
        for key in feasible_1 | feasible_2:
            dists['A'][key] += tsp.ab_cycle_len(ab_cycles['A'][key])
            dists['B'][key] += tsp.ab_cycle_len(ab_cycles['B'][key])

            # Distance diference inside AB_cycle
            diff = abs(dists['A'][key] - dists['B'][key])

            # Store AB_cycle with minor diference
            if not minor['key']:
                minor['key'] = key
                minor['value'] = diff
            elif diff < minor['value']:
                minor['key'] = key
                minor['value'] = diff

            # Chose best edges in each partition
            if dists['A'][key] <= dists['B'][key]:
                solution.add(tuple(['A', key]))
            else:
                solution.add(tuple(['B', key]))

        # Create each solution
        solution_1 = list()
        solution_2 = list()
        sum_1 = 0

        for cycle, key in solution:
            solution_1.extend(ab_cycles[cycle][key])
            sum_1 += dists[cycle][key]
            if key != minor['key']:
                solution_2.extend(ab_cycles[cycle][key])
            elif cycle == 'A':
                solution_2.extend(ab_cycles['B'][minor['key']])
            else:
                solution_2.extend(ab_cycles['A'][minor['key']])

        # Total sum of one partial solution
        sum_A = sum(dists['A'].values())

        # Calc common edges distance
        common_dist = self.get_len() - sum_A

        # Create solutions graphs
        graph_1 = defaultdict(set)
        graph_2 = defaultdict(set)

        # Create solution 1 undirected graph
        for i, j in zip(solution_1[0::2], solution_1[1::2]):
            graph_1[abs(i)].add(abs(j))
            graph_1[abs(j)].add(abs(i))

        # Create solution 1 undirected graph
        for i, j in zip(solution_2[0::2], solution_2[1::2]):
            graph_2[abs(i)].add(abs(j))
            graph_2[abs(j)].add(abs(i))

        # Common graph and solutions graph union
        graph_1 = Graph(graph_1) | Graph(common)
        graph_2 = Graph(graph_2) | Graph(common)

        # Create tours from solutions graph
        vertices_1, tour_1 = Graph.dfs(graph_1, 1)
        vertices_2, tour_2 = Graph.dfs(graph_2, 1)

        # Verify infeasible tours
        assert len(vertices_1) != self._size
        c_1 = Chromosome(tour_1)
        c_1.set_len(common_dist + sum_1)

        assert len(vertices_2) != self._size
        c_2 = Chromosome(tour_2)
        c_2.set_len(common_dist + sum_1 + minor['value'])

        return c_1, c_2

    # Partition Crossover
    def __mul__(self, other):
        # Toursinfeasible.clear()
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
            # Remove degree 2 nodes (surrogate edge)
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

        # If exists one or no partition, return parents
        if len(vertices_m) <= 1 and len(vertices_n) <= 1:
            return self, other

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

        # Try to fuse infeasible partitions
        self._fusion(vertices, ab_cycles, simple_graph_a, simple_graph_b,
                     tour_a, tour_b, feasible_1, feasible_2, infeasible)

        # After fusion, if exists one or no partition, return parents
        if len(feasible_1) + len(feasible_2) + len(infeasible) <= 1:
            return self, other

        if False:
            print "Tour 1: ", tour_a
            print "Tour 2: ", tour_b
            print
            print simple_graph_a
            print simple_graph_b
            print
            print ab_cycles['A']
            print
            print "Feasible 1: ", feasible_1
            print "Feasible 2: ", feasible_2
            print "Infeasible: ", infeasible

        # Common edges
        common = self._undirected_graph & other._undirected_graph

        # Buil solutions
        return self._build_solutions(ab_cycles, feasible_1, feasible_2, common)

    # 2-opt adapted from
    # https://en.wikipedia.org/wiki/2-opt
    # https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
    # http://pedrohfsd.com/2017/08/09/2opt-part1.html
    # https://rawgit.com/pedrohfsd/TSP/develop/2opt.js
    def two_opt(self):
        best_route = self._tour
        best_len = self.get_len()
        improved = True
        tested = set()
        # Stop when no improvement is made
        while improved:
            improved = False
            for i in xrange(self._size - 1):
                for j in xrange(i + 1, self._size):
                    if j-i == self._size - 1: continue
                    # Create edges swap in advance
                    join_a = sorted([sorted([best_route[i-1], best_route[i]]),
                                   sorted([best_route[j],
                                   best_route[(j+1)%self._size]])])

                    join_b = sorted([sorted([best_route[i-1], best_route[j]]),
                                   sorted([best_route[i],
                                   best_route[(j+1)%self._size]])])

                    # List of lists to tuple
                    join_a = tuple(v for sub in join_a for v in sub)
                    join_b = tuple(v for sub in join_b for v in sub)

                    # Avoid duplicated tests
                    if (frozenset([join_a, join_b]) in tested or
                        join_a == join_b):
                        continue

                    # Store cases to not be tested again
                    tested.add(frozenset([join_a, join_b]))

                    # Calc distances
                    join_a_len = tsp.ab_cycle_len_2(join_a)
                    join_b_len = tsp.ab_cycle_len_2(join_b)

                    # Verify if swap is shorter
                    if join_b_len < join_a_len:
                        # 2opt swap
                        new_route = best_route[0:i]
                        new_route.extend(reversed(best_route[i:j + 1]))
                        new_route.extend(best_route[j+1:])
                        best_route = new_route
                        best_len = best_len - join_a_len + join_b_len
                        improved = True

        # Rotate solution to begin with 1
        assert len(set(best_route)) == self._size
        p = best_route.index(1)
        best_route = deque(best_route)
        best_route.rotate(-p)
        self._tour = list(best_route)
        self.set_len(best_len)

#def test(size, limit):
#    for x in xrange(limit):
#        p1 = Chromosome(size)
#        p2 = Chromosome(size)
#        while (p1 == p2):
#            p2 = Chromosome(size)
#        c_1, c_2 = p1 * p2
#        print '\r', x,

#test(76, 1000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

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

#F2 3i (2f1)
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

#F11 4i (2 fusions f2) execption
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

# F15 (1f1, 1f2)
#p1 = Chromosome((1, 8, 4, 5, 6, 7, 9, 2, 10, 3))
#p2 = Chromosome((1, 6, 9, 8, 3, 7, 5, 10, 4, 2))

# F16
#p1 = Chromosome((1, 8, 10, 2, 6, 9, 5, 3, 7, 4))
#p2 = Chromosome((1, 4, 5, 2, 3, 9, 6, 10, 8, 7))

# Teste
#p1 = Chromosome((1,2,3,4,5,6))
#p2 = Chromosome((1,2,3,6,4,5))

# Force test
#p1 = Chromosome(100000)
#p2 = Chromosome(100000)

# Minimal exemple
#p1 = Chromosome((1,2,3,4))
#p2 = Chromosome((1,3,2,4))

#p1 * p2

p1 = Chromosome(493)

print p1.get_tour()
p1.two_opt()
print p1.get_tour()


#!/usr/bin/python
# ozeasx@gmail.com

import time
import numpy as np
from itertools import combinations
from collections import defaultdict
from collections import deque
from graph import Graph

# Infeasible value compared to feasible partitions
INFEASIBLE_WEIGHT = 0.4

class Chromosome(object):
    # Constructor
    def __init__(self, data, tour = None, dist = None):
        self.set(data, tour, dist)

    # Chromosome setup
    def set(self, data, tour = None, dist = None):
        # Creates tour
        # Create random based on instance dimension
        if not tour:
            nodes = range(2, data.get_dimension() + 1)
            self._tour = list(np.random.choice(nodes, len(nodes),
                              replace=False))
            self._tour.insert(0, 1)
        # Create random tour based on given dimension
        elif isinstance(tour, int):
            nodes = range(2, tour + 1)
            self._tour = list(np.random.choice(nodes, len(nodes),
                              replace=False))
            self._tour.insert(0, 1)
        # User defined tour
        elif (isinstance(tour, list) or isinstance(tour, tuple) or
              isinstance(tour, deque)):
            self._tour = list(tour)

        # Tour distance
        if dist:
            self._dist = dist
        else:
            self._dist = data.route_dist(self._tour + [1])

        # TSPLIB Instance
        self._data = data

        # Number of cities
        self._dimension = len(self._tour)

        # Fitness
        self._fitness = None

        # undirected graph representaition
        self._undirected_graph = Graph.gen_undirected_graph(self._tour)

    # == method overload
    def __eq__(self, other):
        return self._undirected_graph == other._undirected_graph

    # Get fitness
    def get_fitness(self):
        return self._fitness

    # Get tour
    def get_tour(self):
        return self._tour

    # Get tour dimension
    def get_dimension(self):
        return self._dimension

    # Get tour distance
    def get_dist(self):
        return self._dist

    # Set fitness
    def set_fitness(self, value):
        self._fitness = value

    # Set tour distance
    def set_dist(self, value):
        #print value, tsp.route_dist(self._tour + [1])
        #assert abs(value - tsp.route_dist(self._tour + [1])) < 0.05
        self._dist = value

    # Find partitions using dfs
    def _partition(self, graph_a, graph_b):
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), defaultdict(dict)
        # Simetric diference
        graph = graph_a ^ graph_b
        # Loop
        loop = set(graph)
        index = 1
        while loop:
            vertices[index], ab_cycles['A'][index] = Graph.dfs(graph, loop.pop())
            ab_cycles['B'][index] = deque(ab_cycles['A'][index])
            # Fork AB_cycles
            if (ab_cycles['A'][index][0] in graph_b and
                ab_cycles['A'][index][1] in graph_b[ab_cycles['A'][index][0]]):
                ab_cycles['A'][index].rotate(1)
            else:
                ab_cycles['B'][index].rotate(1)
            # Reduce loop
            loop -= vertices[index]
            # Increment index
            index += 1
        # Return
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
                    simple_tour[key].extend([i, j, 'c'])

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

    # Try fusion of infeasible partitions
    def _fusion(self, vertices, ab_cycles, sg_a, sg_b, tour_a, tour_b,
                feasible_1, feasible_2, infeasible):

        # Fused partitions
        fused = set()

        # Start fusion try with 2 partitions
        n = 2
        while n < len(infeasible):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(infeasible, n):
                # Count common edges
                count = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    count += len(Graph(sg_a[i]) & Graph(sg_a[j]))

                # Create element with (fusion, count)
                fusion = list(fusion)
                fusion.append(count)
                candidates.append(fusion)

            # Sort by common edges count
            candidates.sort(key = lambda fusion: fusion[n], reverse = True)
            # Increment fusion size
            n += 1
            # Discard common edges count
            for fusion in candidates:
                fusion.pop(-1)
            # Convert elements to tuples
            candidates = map(tuple, candidates)

            # Try fusions
            for fusion in candidates:
                union = defaultdict(set)
                # Test to determine if partition is fused already
                if not any(i in fused for i in fusion):
                    for i in fusion:
                        union[fusion] |= vertices[i]

                    # Create simple graphs for fusion
                    sga = self._gen_simple_graph(union, tour_a)
                    sgb = self._gen_simple_graph(union, tour_b)

                    # Check if fusion is feasible
                    f1, f2, _ = self._return_feasible(sga, sgb)

                    # Update information
                    if fusion in f1 or fusion in f2:
                        ab_cycles['A'][fusion] = deque()
                        ab_cycles['B'][fusion] = deque()
                        for i in fusion:
                            ab_cycles['A'][fusion].extend(ab_cycles['A'][i])
                            ab_cycles['B'][fusion].extend(ab_cycles['B'][i])
                            infeasible.remove(i)
                            fused.add(i)
                            feasible_1.update(f1)
                            feasible_2.update(f2)

        # Fuse all remaining partitions
        if len(infeasible) > 1:
            fusion = tuple(infeasible)
            ab_cycles['A'][fusion] = deque()
            ab_cycles['B'][fusion] = deque()
            for i in fusion:
                ab_cycles['A'][fusion].extend(ab_cycles['A'][i])
                ab_cycles['B'][fusion].extend(ab_cycles['B'][i])
            feasible_1.add(fusion)
        # Remaining partition is feasilbe 2?
        elif len(infeasible) == 1:
            feasible_2.update(infeasible)

        # Clear infeasible
        infeasible.clear()

    # Build solutions
    def _build_solutions(self, ab_cycles, feasible_1, feasible_2, common):

        # dists of each partition in each solution
        dists = dict()
        dists['A'] = defaultdict(float)
        dists['B'] = defaultdict(float)

        # Set to store partial solution (cycle, key)s
        solution = set()

        # Partition with minor diff
        minor = {'key': None, 'value': None}

        # Get distance of all partitions tours
        for key in feasible_1 | feasible_2:
            dists['A'][key] += self._data.ab_cycle_dist(ab_cycles['A'][key])
            dists['B'][key] += self._data.ab_cycle_dist(ab_cycles['B'][key])

            # Distance diference inside AB_cycle
            diff = abs(dists['A'][key] - dists['B'][key])

            # Store AB_cycle with minor diference
            if not minor['key']:
                minor['key'] = key
                minor['value'] = diff
            elif diff < minor['value']:
                minor['key'] = key
                minor['value'] = diff

            # Chose best edges in each partition
            if dists['A'][key] <= dists['B'][key]:
                solution.add(tuple(['A', key]))
            else:
                solution.add(tuple(['B', key]))

        # Create each solution
        solution_1 = list()
        solution_2 = list()
        sum_1 = 0

        for cycle, key in solution:
            solution_1.extend(ab_cycles[cycle][key])
            sum_1 += dists[cycle][key]
            if key != minor['key']:
                solution_2.extend(ab_cycles[cycle][key])
            elif cycle == 'A':
                solution_2.extend(ab_cycles['B'][minor['key']])
            else:
                solution_2.extend(ab_cycles['A'][minor['key']])

        # Total sum of one partial solution
        sum_A = sum(dists['A'].values())

        # Calc common edges distance
        common_dist = self.get_dist() - sum_A

        # Create solutions graphs
        graph_1 = defaultdict(set)
        graph_2 = defaultdict(set)

        # Create solution 1 undirected graph
        for i, j in zip(solution_1[0::2], solution_1[1::2]):
            graph_1[abs(i)].add(abs(j))
            graph_1[abs(j)].add(abs(i))

        # Create solution 1 undirected graph
        for i, j in zip(solution_2[0::2], solution_2[1::2]):
            graph_2[abs(i)].add(abs(j))
            graph_2[abs(j)].add(abs(i))

        # Common graph and solutions graph union
        graph_1 = Graph(graph_1) | Graph(common)
        graph_2 = Graph(graph_2) | Graph(common)

        # Create tours from solutions graph
        vertices_1, tour_1 = Graph.dfs(graph_1, 1)
        vertices_2, tour_2 = Graph.dfs(graph_2, 1)

        # Verify infeasible tours
        assert len(vertices_1) == self._dimension
        assert len(vertices_2) == self._dimension

        # Return solutions
        return (Chromosome(self._data, tour_1, common_dist + sum_1),
                Chromosome(self._data, tour_2, common_dist + sum_1 +
                           minor['value']))

    # Partition Crossover
    def __mul__(self, other):
        # Duplicated solutions
        if self == other:
            return (Chromosome(self._data, self._tour, self._dist),
                    Chromosome(other._data, other._tour, other._dist))

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
            # Remove degree 2 nodes (surrogate edge)
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

        # If exists one or no partition, return parents
        if len(vertices_m) <= 1 and len(vertices_n) <= 1:
            return (Chromosome(self._data, self._tour, self._dist),
                    Chromosome(other._data, other._tour, other._dist))

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

        # Try to fuse infeasible partitions
        self._fusion(vertices, ab_cycles, simple_graph_a, simple_graph_b,
                     tour_a, tour_b, feasible_1, feasible_2, infeasible)

        # After fusion, if exists one or no partition, return parents
        if len(feasible_1) + len(feasible_2) + len(infeasible) <= 1:
            return (Chromosome(self._data, self._tour, self._dist),
                    Chromosome(other._data, other._tour, other._dist))

        if False:
            print "Tour 1: ", tour_a
            print "Tour 2: ", tour_b
            print
            print simple_graph_a
            print simple_graph_b
            print
            print ab_cycles['A']
            print
            print "Feasible 1: ", feasible_1
            print "Feasible 2: ", feasible_2
            print "Infeasible: ", infeasible

        # Common edges
        common = self._undirected_graph & other._undirected_graph

        # Buil and return solutions
        return self._build_solutions(ab_cycles, feasible_1, feasible_2, common)

    # 2-opt adapted from
    # https://en.wikipedia.org/wiki/2-opt
    # https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
    # http://pedrohfsd.com/2017/08/09/2opt-part1.html
    # https://rawgit.com/pedrohfsd/TSP/develop/2opt.js
    def two_opt(self):
        best_route = list(self._tour)
        best_dist = self.get_dist()
        improved = True
        tested = set()
        # Stop when no improvement is made
        while improved:
            improved = False
            for i in xrange(self._dimension - 1):
                for j in xrange(i + 1, self._dimension):
                    if j-i == self._dimension - 1:
                        continue
                    # Create edges swap in advance
                    join_a = sorted([sorted([best_route[i-1], best_route[i]]),
                                   sorted([best_route[j],
                                   best_route[(j+1) % self._dimension]])])

                    join_b = sorted([sorted([best_route[i-1], best_route[j]]),
                                   sorted([best_route[i],
                                   best_route[(j+1) % self._dimension]])])

                    # List of lists to tuple
                    join_a = tuple(v for sub in join_a for v in sub)
                    join_b = tuple(v for sub in join_b for v in sub)

                    # Avoid duplicated tests
                    if (frozenset([join_a, join_b]) in tested or
                        join_a == join_b):
                        continue

                    # Store cases to not be tested again
                    tested.add(frozenset([join_a, join_b]))

                    # Calc distances
                    join_a_dist = self._data.ab_cycle_dist(join_a)
                    join_b_dist = self._data.ab_cycle_dist(join_b)

                    # Verify if swap is shorter
                    if join_b_dist < join_a_dist:
                        # 2opt swap
                        new_route = best_route[0:i]
                        new_route.extend(reversed(best_route[i:j + 1]))
                        new_route.extend(best_route[j+1:])
                        best_route = new_route
                        best_dist = best_dist - join_a_dist + join_b_dist
                        improved = True

        # Rotate solution to begin with 1
        assert len(set(best_route)) == self._dimension
        p = best_route.index(1)
        best_route = deque(best_route)
        best_route.rotate(-p)
        self._tour = list(best_route)
        self._dist = best_dist
        #return Chromosome(self._data, best_route, best_dist)

# Snipet code to test a lot of random cases
#from shell import Shell
#from tsp import TSPLIB

#cmd = Shell()
#tsp = TSPLIB("../tsplib/ulysses16.tsp", cmd)

#def test(data, limit, dimension = None):
#    for x in xrange(limit):
#        p1 = Chromosome(data)
#        p2 = Chromosome(data)
#        while (p1 == p2):
#            p2 = Chromosome(data)
#        c1, c2 = p1 * p2
#        print (c1.get_dist() + c2.get_dist()) - (p1.get_dist() + p2.get_dist())
#        print c1.get_tour()
#        print c2.get_tour()
#        print p1.get_tour()
#        print p1.get_dist()
#        p1.two_opt()
#        print p1.get_tour()
#        print p1.get_dist()
#        print '\r', x,

#test(tsp, 10000)

# Whitley2010-F1
#p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

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

#F2 3i (2f1)
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

#F11 4i (2 fusions f2) execption
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

# F15 (1f1, 1f2)
#p1 = Chromosome((1, 8, 4, 5, 6, 7, 9, 2, 10, 3))
#p2 = Chromosome((1, 6, 9, 8, 3, 7, 5, 10, 4, 2))

# F16
#p1 = Chromosome((1, 8, 10, 2, 6, 9, 5, 3, 7, 4))
#p2 = Chromosome((1, 4, 5, 2, 3, 9, 6, 10, 8, 7))

# Teste
#p1 = Chromosome((1,2,3,4,5,6))
#p2 = Chromosome((1,2,3,6,4,5))

# Force test
#p1 = Chromosome(100000)
#p2 = Chromosome(100000)

# Minimal exemple
#p1 = Chromosome((1,2,3,4))
#p2 = Chromosome((1,3,2,4))

# FUck me
#p1 = Chromosome(tsp, [1, 3, 2, 4, 8, 16, 15, 14, 13, 12, 10, 9, 11, 5, 6, 7])
#p2 = Chromosome(tsp, [1, 15, 3, 11, 5, 9, 7, 16, 12, 13, 14, 6, 10, 4, 2, 8])

#c1, c2 = p1 * p2

#print c1.get_tour()
#print c2.get_tour()

#p1 = Chromosome(76)

#print p1.get_tour()
#p1.two_opt()
#print p1.get_tour()


#!/usr/bin/python
# ozeasx@gmail.com

import time
import random
import copy
from operator import itemgetter, attrgetter
from chromosome import Chromosome
from gpx import GPX
import mut

class GA(object):
    # GA initialization
    def __init__(self, data, p_cross=0.8, p_mut=0.05, n_elite = None):
        # Parametrization
        self._data = data
        self._gpx = GPX(data)
        self._p_cross = p_cross
        self._p_mut = p_mut
        self._n_elite = n_elite

        # Statitics
        # Population size
        self._pop_size = 0
        # Generations
        self._generation = 0
        # Average fitness of the current generation
        self._avg_fitness = 0
        # Numbers of crossover
        self._cross = 0
        self._last_cross = 0
        # Numbers os mutation
        self._mut = 0
        self._last_mut = 0

        # List to store current population
        self._population = list()
        # Best and worst solutions found
        self._best_solution = None
        self._worst_solution = None

    # Get current generation number
    @property
    def generation(self):
        return self._generation

    # return best individuals
    @property
    def best_solution(self):
        return self._best_solution

    # Generate inicial population
    # N = Number of individuals
    def gen_pop(self, size):
        assert not (size % 2), "Invalid population size. \
                                Must be even and greater than 0"
        print "Generating initial population..."
        i = 0
        while i < size:
            c = Chromosome(self._data.dimension)
            c.dist = self._data.tour_dist(c.tour)
            c = mut.two_opt(c, self._data)
            # Prevent duplicated individuals
            if not c in self._population:
                self._population.append(c)
                i += 1
        print "Done..."
        # Start time count
        self._start_time = time.time()

    # Evaluate the entire population
    def evaluate(self):
        # Set population size and increment generation
        self._pop_size = len(self._population)
        self._generation += 1

        # Update fitness sum and average
        total_fitness = 0
        for c in self._population:
            c.fitness = self._evaluate(c)
            total_fitness += c.fitness

        # Calc average fitness
        self._avg_fitness = total_fitness/float(self._pop_size)

        # Sort population
        self._population.sort(key = attrgetter('fitness'), reverse=True)

        # Store best and worst solutions found
        if not self._best_solution:
            self._best_solution = copy.deepcopy(self._population[0])

        if self._population[0].fitness > self._best_solution.fitness:
            self._best_solution = copy.deepcopy(self._population[0])

        if not self._worst_solution:
            self._worst_solution = copy.deepcopy(self._population[-1])

        if self._population[-1].fitness < self._worst_solution.fitness:
            self._worst_solution = copy.deepcopy(self._population[-1])

    # Tournament selection
    def select_tournament(self, k):
        # Tournament winners
        selected = list()

        for i in xrange(self._pop_size):
            # Retrieve k-sized sample
            tournament = random.sample(self._population, k)
            # Sort by fitness
            tournament.sort(key = attrgetter('fitness'))
            # Get best solution
            selected.append(tournament.pop())

        # Update population
        self._population = selected

    # Recombine parents according to p_cross probability
    def recombine(self):

        # Shuffle population
        random.shuffle(self._population)

        for p1, p2 in zip(self._population[0::2], self._population[1::2]):
            if random.random() < self._p_cross:
                c1, c2 = self._gpx.recombine(p1, p2)
                # Replace p1 and p2 only if c1 or c2 are different from parents
                if c1 not in [p1, p2] or c2 not in [p1, p2]:
                    p1, p2 = c1, c2
                    self._cross += 1

    # Mutate individuals according to p_mut probability
    def mutate(self):
        for c in self._population:
            if random.random() < self._p_mut:
                while c in self._population:
                    c = Chromosome(self._data.dimension)
                    c.dist = self._data.tour_dist(c.tour)
                    c = mut.two_opt(c, self._data)
                self._mut += 1

    # Reset population
    def restart_pop(self, percent):
        if not (self._cross - self._last_cross):
            self._population.sort(key = attrgetter('fitness'))
            for c in self._population[:int(self._pop_size * percent)]:
                while c in self._population:
                    c = Chromosome(self._data.dimension)
                    c.dist = self._data.tour_dist(c.tour)
                    c = mut.two_opt(c, self._data)

    # Generation info
    def print_info(self):
        cross = self._cross - self._last_cross
        mut = self._mut - self._last_mut
        print "T: %i\tCross: %i\tMut: %i\tAverage: %f\tWorst: %f\tBest: %f" % \
        (self._generation, cross, mut, self._avg_fitness, \
        self._worst_solution.fitness, self._best_solution.fitness)
        self._last_cross = self._cross
        self._last_mut = self._mut

    # Final report
    def report(self):
        print "---------------------------------------------------------------"
        print "Total Crossover: ", self._cross
        print "Total mutations: ", self._mut
        print "Execution time: ", time.time() - self._start_time
        print "---------------------------------------------------------------"
        if self._data.best_tour:
            print "Best known solution:"
            print "Tour: ", self._data.best_tour
            print "Distance: ", self._data.tour_dist(self._data.best_tour)
            print "-----------------------------------------------------------"
        print "Best individual found:"
        print "Tour: ", self._best_solution.tour
        print "Distance: ", self._best_solution.dist
        print "---------------------------------------------------------------"

    # Calculate the individual fitness
    def _evaluate(self, c):
        return -c.dist


#!/usr/bin/python
# ozeasx@gmail.com

import time
import threading
import random
import copy
from operator import itemgetter, attrgetter
from graph import Graph
from chromosome import Chromosome
from gpx import GPX
import mut

class GA(object):
    # GA initialization
    def __init__(self, data, p_cross=0.8, p_mut=0.05, n_elite = None):
        # Parametrization
        self._data = data
        self._gpx = GPX(data)
        self._p_cross = p_cross
        self._p_mut = p_mut
        self._n_elite = n_elite

        # Statitics
        # Population size
        self._pop_size = 0
        # Generations
        self._generation = 0
        # Average fitness of the current generation
        self._avg_fitness = 0
        # Numbers of crossover
        self._cross = 0
        self._last_cross = 0
        # Numbers os mutation
        self._mut = 0
        self._last_mut = 0

        # List to store current population
        self._population = list()
        # Best and worst solutions found
        self._best_solution = None

    # Get current generation number
    @property
    def generation(self):
        return self._generation

    # return best individuals
    @property
    def best_solution(self):
        return self._best_solution

    # Generate inicial population
    # N = Number of individuals
    def gen_pop(self, size):
        assert not (size % 2), "Invalid population size. \
                                Must be even and greater than 0"
        print "Generating initial population..."
        i = 0
        while i < size:
            c = Chromosome(self._data.dimension)
            if c not in self._population and self._max_dist(c) < 1:
                c.dist = self._data.tour_dist(c.tour)
                self._population.append(c)
                i += 1
            print '\r', i,
        print "Done..."
        # Start time count
        self._start_time = time.time()

    # Evaluate the entire population
    def evaluate(self):
        # Set population size and increment generation
        self._pop_size = len(self._population)
        self._generation += 1

        # Update fitness sum and average
        total_fitness = 0
        for c in self._population:
            c.fitness = self._evaluate(c)
            total_fitness += c.fitness

        # Calc average fitness
        self._avg_fitness = total_fitness/float(self._pop_size)

        # Sort population
        self._population.sort(key = attrgetter('fitness'), reverse=True)

        # Store best and worst solutions found
        if not self._best_solution:
            self._best_solution = copy.deepcopy(self._population[0])

        if self._population[0].fitness > self._best_solution.fitness:
            self._best_solution = copy.deepcopy(self._population[0])

    # Tournament selection
    def select_tournament(self, k):
        # Tournament winners
        selected = list()

        for i in xrange(self._pop_size):
            # Retrieve k-sized sample
            tournament = random.sample(self._population, k)
            # Sort by fitness
            tournament.sort(key = attrgetter('fitness'))
            # Get best solution
            selected.append(tournament.pop())

        # Update population
        self._population = selected

    # Recombine parents according to p_cross probability
    def recombine(self):

        # Shuffle population loop
        loop = random.sample(xrange(self._pop_size), self._pop_size)

        for i, j in zip(loop[0::2], loop[1::2]):
            if random.random() < self._p_cross:
                c1, c2 = self._gpx.recombine(self._population[i],
                                             self._population[j])
                # Replace p1 and p2 only if c1 or c2 are different from parents
                if c1 not in [self._population[i], self._population[j]] or \
                   c2 not in [self._population[i], self._population[j]]:
                   self._population[i], self._population[j] = c1, c2
                   self._cross += 1

    # Mutate individuals according to p_mut probability
    def mutate(self):
        for i in xrange(self._pop_size):
            if random.random() < self._p_mut:
                self._population[i] = mut.two_opt(self._population[i],
                                                  self._data)
                self._mut += 1

    # Reset population
    def restart_pop(self, percent):
        if not (self._cross - self._last_cross):
            self._population.sort(key = attrgetter('fitness'))
            for i in xrange(int(self._pop_size * percent)):
                c = Chromosome(self._data.dimension)
                while c in self._population or self._max_dist(c) == 1:
                    c = Chromosome(self._data.dimension)
                c.dist = self._data.tour_dist(c.tour)
                self._population[i] = c

    # Generation info
    def print_info(self):
        cross = self._cross - self._last_cross
        mut = self._mut - self._last_mut
        print "T: %i\tCross: %i\tMut: %i\tAverage: %f\tBest: %f" % \
        (self._generation, cross, mut, self._avg_fitness, \
        self._best_solution.fitness)
        self._last_cross = self._cross
        self._last_mut = self._mut
        #for c in self._population:
        #    print c
        #raw_input("Press enter to continue")

    # Final report
    def report(self):
        print "---------------------------------------------------------------"
        print "Total Crossover: ", self._cross
        print "Total mutations: ", self._mut
        print "Execution time: ", time.time() - self._start_time
        print "---------------------------------------------------------------"
        if self._data.best_tour:
            print "Best known solution:"
            print "Tour: ", self._data.best_tour
            print "Distance: ", self._data.tour_dist(self._data.best_tour)
            print "-----------------------------------------------------------"
        print "Best individual found:"
        print "Tour: ", self._best_solution.tour
        print "Distance: ", self._best_solution.dist
        print "---------------------------------------------------------------"

    # Calculate the individual fitness
    def _evaluate(self, c):
        return -c.dist

    # Calculate avg distance from individual to population
    def _max_dist(self, c):
        if self._population:
            aux = 0
            for p in self._population[:10]:
                aux = max(aux, c - p)
            return aux


#!/usr/bin/python
# ozeasx@gmail.com

import time
from multiprocessing import Pool
from collections import defaultdict
import random
import copy
from operator import itemgetter, attrgetter
from graph import Graph
from chromosome import Chromosome
from gpx import GPX
import mut

WORKERS = 2

class GA(object):
    # GA initialization
    def __init__(self, data, p_cross, p_mut, elite = 0):
        # Parametrization
        self._data = data
        self._gpx = GPX(data)
        self._p_cross = p_cross
        self._p_mut = p_mut
        self._elite = elite

        # Statitics
        # Population size
        self._pop_size = 0
        # Generations
        self._generation = 0
        # Average fitness of the current generation
        self._avg_fitness = 0
        # Numbers of crossover
        self._cross = 0
        self._last_cross = 0
        # Numbers os mutation
        self._mut = 0
        self._last_mut = 0
        # Timers
        self._execution_time = defaultdict(list)

        # List to store current population
        self._population = list()
        # Best and worst solutions found
        self._best_solution = None

    # Get current generation number
    @property
    def generation(self):
        return self._generation

    # return best individuals
    @property
    def best_solution(self):
        return self._best_solution

    # Generate inicial population
    # N = Number of individuals
    def gen_pop(self, size):
        # Regiter start time
        start_time = time.time()
        # Need even population
        assert not (size % 2), "Invalid population size. \
                                Must be even and greater than 0"
        print "Generating initial population..."
        i = 0
        while i < size:
            c = Chromosome(self._data.dimension)
            if c not in self._population:
                c.dist = self._data.tour_dist(c.tour)
                self._population.append(c)
                i += 1
            print '\r', i,
        print "Done..."
        # Store execution time
        self._execution_time['gen_pop'].append(time.time() - start_time)
        # Global start time
        self._start_time = time.time()

    # Evaluate the entire population
    def evaluate(self):
        # Register star time
        start_time = time.time()
        # Set population size and increment generation
        self._pop_size = len(self._population)
        self._generation += 1

        # Update fitness sum and average
        total_fitness = 0
        for c in self._population:
            c.fitness = self._evaluate(c)
            total_fitness += c.fitness

        # Calc average fitness
        self._avg_fitness = total_fitness/float(self._pop_size)

        # Sort population
        self._population.sort(key = attrgetter('fitness'))

        # Store best and worst solutions found
        if not self._best_solution:
            self._best_solution = copy.deepcopy(self._population[-1])

        if self._population[-1].fitness > self._best_solution.fitness:
            self._best_solution = copy.deepcopy(self._population[-1])

        # Register execution Timers
        self._execution_time['evaluate'].append(time.time() - start_time)

    # Tournament selection
    def select_tournament(self, k):
        # Register start time
        start_time = time.time()
        # Tournament winners
        selected = list()

        for n in xrange(self._elite):
            selected.append(self._population.pop())

        for i in xrange(self._pop_size - self._elite):
            # Retrieve k-sized sample
            tournament = random.sample(self._population, k)
            # Sort by fitness
            tournament.sort(key = attrgetter('fitness'))
            # Get best solution
            selected.append(tournament.pop())

        # Update population
        self._population = selected
        # Regiter execution time
        self._execution_time['select_tournament'].append(time.time() -
                                                         start_time)

    # Recombine parents according to p_cross probability
    def recombine(self):
        # Register start time
        start_time = time.time()
        # Shuffle population loop
        loop = random.sample(xrange(self._pop_size), self._pop_size)

        for i, j in zip(loop[0::2], loop[1::2]):
            if random.random() < self._p_cross:
                c1, c2 = self._gpx.recombine(self._population[i],
                                             self._population[j])
                # Replace p1 and p2 only if c1 or c2 are different from parents
                if c1 not in [self._population[i], self._population[j]] or \
                   c2 not in [self._population[i], self._population[j]]:
                   self._population[i], self._population[j] = c1, c2
                   self._cross += 1
        # Register execution time
        self._execution_time['recombine'].append(time.time() - start_time)

    # Recombine parents according to p_cross probability
    def recombine_t(self):
        # Register start time
        start_time = time.time()
        # Function to be maped
        def work(p):
            if random.random() < self._p_cross:
                gpx = GPX(self._data)
                c1, c2 = gpx.recombine(self._population[p[0]],
                                       self._population[p[1]])

                if c1 not in [self._population[p[0]], self._population[p[1]]] or \
                   c2 not in [self._population[p[0]], self._population[p[1]]]:
                   self._population[p[0]], self._population[p[1]] = c1, c2
                   self._cross += 1

        # Shuffle population loop
        pop = range(self._pop_size)
        couples = zip(pop[0::2], pop[1::2])

        # Multiprocessing
        pool = Pool(WORKERS)
        pool.map(work, couples)
        pool.close()
        pool.join()
        # Register execution time
        self._execution_time['recombine'].append(time.time() - start_time)

    # Mutate individuals according to p_mut probability
    def mutate(self):
        # Register start time
        start_time = time.time()
        for i in xrange(self._pop_size):
            if random.random() < self._p_mut:
                self._population[i] = mut.two_opt(self._population[i],
                                                  self._data)
                self._mut += 1
        # Register execution time
        self._execution_time['mutate'].append(time.time() - start_time)

    # Mutate individuals according to p_mut probability
    def mutate_t(self):
        # Register start time
        start_time = time.time()
        # Function to be maped
        def work(i):
            if random.random() < self._p_mut:
                self._population[i] = mut.two_opt(self._population[i],
                                                  self._data)
                self._mut += 1

        # Multiprocessing
        pool = Pool(WORKERS)
        pool.map(work, xrange(self._pop_size))
        pool.close()
        pool.join()
        # Register execution time
        self._execution_time['mutate'].append(time.time() - start_time)

    # Reset population
    def restart_pop(self, percent):
        # Register start time
        start_time = time.time()

        if not (self._cross - self._last_cross):
            self._population.sort(key = attrgetter('fitness'))
            for i in xrange(int(self._pop_size * percent)):
                c = Chromosome(self._data.dimension)
                while c in self._population:
                    c = Chromosome(self._data.dimension)
                c.dist = self._data.tour_dist(c.tour)
                self._population[i] = c

        # Register execution time
        self._execution_time['restart_pop'].append(time.time() - start_time)

    # Generation info
    def print_info(self):
        cross = self._cross - self._last_cross
        mut = self._mut - self._last_mut
        print "T: %i\tCross: %i\tMut: %i\tAverage: %f\tBest: %f" % \
        (self._generation, cross, mut, self._avg_fitness, \
        self._best_solution.fitness)
        self._last_cross = self._cross
        self._last_mut = self._mut

    # Final report
    def report(self):
        print "------------------------ Statitics -----------------------------"
        print "Total Crossover:", self._cross
        print "Total mutations:", self._mut
        print "---------------------- Time statistics--------------------------"
        print "Execution time:", time.time() - self._start_time
        print "Inicial population:", sum(self._execution_time['gen_pop'])
        print "Evaluation:", sum(self._execution_time['evaluate'])
        print "Selection:", sum(self._execution_time['select_tournament'])
        print "Recombination:", sum(self._execution_time['recombine'])
        print "Mutation:", sum(self._execution_time['mutate'])
        print "Population restart:", sum(self._execution_time['restart_pop'])
        if self._data.best_tour:
            print "----------------- Best known solution ----------------------"
            print "Tour:", self._data.best_tour
            print "Distance:", self._data.tour_dist(self._data.best_tour)
        print "-------------------- Best individual found ---------------------"
        print "Tour:", self._best_solution.tour
        print "Distance:", self._best_solution.dist
        print "----------------------------------------------------------------"

    # Calculate the individual fitness
    def _evaluate(self, c):
        return -c.dist

    # Calculate avg distance from individual to population
    def _max_dist(self, c):
        if self._population:
            aux = 0
            for p in self._population[:10]:
                aux = max(aux, c - p)
            return aux

def recombine(self):
    # Register start time
    start_time = time.time()
    # Shuffle population loop
    loop = random.sample(xrange(self._pop_size), self._pop_size)

    for i, j in zip(loop[0::2], loop[1::2]):
        if random.random() < self._p_cross:
            c1, c2 = self._gpx.recombine(self._population[i],
                                         self._population[j])
            # Replace p1 and p2 only if c1 or c2 are different from parents
            if (c1 not in [self._population[i], self._population[j]] or
                c2 not in [self._population[i], self._population[j]]):
               self._population[i], self._population[j] = c1, c2
               self._cross += 1
    # Register execution time
    self._execution_time['recombine'].append(time.time() - start_time)

#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
import os

class TSPLIB(object):
    def __init__(self, instance_path, shell):
        # Set instance file and shell object
        self._instance_path = instance_path
        self._instance_name = instance_path[:-4]
        self._shell = shell
        self._best_tour = None

        # Set tsp dimension
        self._dimension = int(shell.run("grep DIMENSION " + instance_path +
                                       " | cut -d':' -f2").strip())

        # Set best known solution, if exists
        if os.path.isfile(self._instance_name  + ".opt.tour"):
            with open(self._instance_name  + ".opt.tour") as best:
                do = False
                self._best_tour = list()
                for word in best:
                    # End loop if EOF
                    if word.strip() == "EOF":
                        break
                    # If do, store tour
                    if do:
                        self._best_tour.append(int(word))
                    # If TOUR_SECTION, set do to store solution
                    if word.strip() == "TOUR_SECTION":
                        do = True
            # Try to remove cycle close
            try:
                self._best_tour.remove(-1)
            except ValueError:
                pass
            # Converto to tuple
            self._tour = tuple(self._best_tour)

        # Generate distance matrix file
        dm_file = self._instance_name + ".tsp.dm"
        if not os.path.isfile(self._instance_name + ".tsp.dm"):
            shell.run("../tsplib/create_dm.r " + instance_path)

        # Generate dict combination lookup
        self._hash = dict()
        self._dm = dict()
        line_number = 1
        with open(self._instance_name + ".tsp.dm") as dm:
            for key, dist in zip(combinations(xrange(1, self._dimension + 1), 2), dm):
                self._hash[key] = line_number
                self._dm[key] = float(dist)
                line_number += 1

    # Get instance dimension
    @property
    def dimension(self):
        return self._dimension

    # Get best tour found
    @property
    def best_tour(self):
        return self._best_tour

    # Calc AB_cycle distance using distance matrix (memory)
    def ab_cycle_dist(self, ab_cycle):
        # Convert deque to list
        aux = list(ab_cycle)
        # Distance
        distance = 0
        # Distance lookup
        for i, j in zip(aux[0::2], aux[1::2]):
            distance += self._dm[tuple(sorted([abs(i), abs(j)]))]
        # Return result
        return distance

    # Calc tour distance using distance matrix (memory)
    def tour_dist(self, tour):
        assert len(set(tour)) == len(tour), "Invalid TSP tour"
        # distance
        dist = 0
        # Distance matrix lookup
        for i, j in zip(tour[:-1], tour[1:]):
            dist += self._dm[tuple(sorted([abs(i), abs(j)]))]

        # Close path
        dist += self._dm[tuple(sorted([abs(tour[0]), abs(tour[-1])]))]

        return dist

###############################################################################
# Use this methods if distance matrix is too large to store into memory

    # Calc AB cycle distance using file
    def ab_cycle_dist_2(self, ab_cycle):
        # Convert deque to list
        aux = list(ab_cycle)
        # List to store line numbers to be used by sed
        lines = list()
        # Hash lookup to fill line numbers list
        for i, j in zip(aux[0::2], aux[1::2]):
            lines.append(self._hash[tuple(sorted([abs(i), abs(j)]))])
        lines.sort()
        # sed command construction
        # https://stackoverflow.com/questions/14709384/how-to-get-some-specific-lines-from-huge-text-file-in-unix
        cmd = "sed '"
        for number in lines[:-1]:
            cmd += str(number) + "p;"
        cmd += str(lines[-1]) + "q;d' " + self._instance_name + ".tsp.dm"
        # https://stackoverflow.com/questions/3096259/bash-command-to-sum-a-column-of-numbers
        cmd += " | paste -sd+ | bc"
        # Return result
        return float(self._shell.run(cmd))

    # Calc tour distance using file
    def tour_dist_2(self, tour, closed = False):
        # List to store line numbers to be used by sed
        lines = list()
        # Hash lookup to fill line numbers list
        for i, j in zip(tour[:-1], tour[1:]):
            lines.append(self._hash[tuple(sorted([i, j]))])
        # Close path
        lines.append(self._hash[tuple(sorted([tour[0], tour[-1]]))])
        # Line numbers must be in ascending order
        lines.sort()
        # sed command construction
        # https://stackoverflow.com/questions/14709384/how-to-get-some-specific-lines-from-huge-text-file-in-unix
        cmd = "sed '"
        for number in lines[:-1]:
            cmd += str(number) + "p;"
        cmd += str(lines[-1]) + "q;d' " + self._instance_name + ".tsp.dm"
        # https://stackoverflow.com/questions/3096259/bash-command-to-sum-a-column-of-numbers
        cmd += " | paste -sd+ | bc"
        # Return result
        return float(self._shell.run(cmd))


#!/usr/bin/python
# ozeasx@gmail.com

import time
from collections import defaultdict
from collections import deque
from itertools import combinations
from graph import Graph
from chromosome import Chromosome


class GPX(object):
    # Class initialization
    def __init__(self, data=None):
        # dataset to compute distances
        self._data = data
        # Infeasible value compared to feasible partitions
        self._infeasible_weight = 0.4
        # Limit fusion trys
        self._fusion_limit = False
        # Dict with lists containing execution time of each step
        self._execution_time = defaultdict(list)
        # Partitioning data
        self._partitions = dict()
        # Tours created for partitioning
        self._tour_a = None
        self._tour_b = None

    @property
    def infeasible_weight(self):
        return self._infeasible_weight

    @property
    def fusion_limit(self):
        return self._fusion_limit

    @property
    def execution_time(self):
        return self._execution_time

    @property
    def tour_a(self):
        if self._tour_a:
            return tuple(self._tour_a)

    @property
    def tour_b(self):
        if self._tour_b:
            return tuple(self._tour_b)

    @property
    def partitions(self):
        return self._partitions

    @infeasible_weight.setter
    def infeasible_weight(self, value):
        assert 0 < value < 1, "Infeasible weight must be in ]0,1[] interval"
        self._infeasible_weight = value

    @fusion_limit.setter
    def fusion_limit(self, value):
        assert value in [True, False], "Fusion limit must be True/False value"
        self._fusion_limit = value

    # Find partitions using dfs
    def _partition(self, graph_a, graph_b):
        # Mark start time
        start_time = time.time()
        # Vertice set and AB cycles
        vertices, ab_cycles = dict(), defaultdict(dict)
        # Simetric diference
        graph = graph_a ^ graph_b
        # Loop
        loop = set(graph)
        index = 1
        while loop:
            vertices[index], ab_cycles['A'][index] = Graph.dfs(graph,
                                                               loop.pop())
            ab_cycles['B'][index] = deque(ab_cycles['A'][index])
            # Fork AB_cycles
            if ab_cycles['A'][index][0] in graph_b:
                if ab_cycles['A'][index][1] in graph_b[ab_cycles['A'][index][0]]:
                    ab_cycles['A'][index].rotate(1)
                else:
                    ab_cycles['B'][index].rotate(1)
            # Reduce loop
            loop -= vertices[index]
            # Increment index
            index += 1
        # Store execution time and return
        self._execution_time['partition'].append(time.time() - start_time)
        return vertices, ab_cycles

    # Create the simple graph for all partions for given tour
    def _gen_simple_graph(self, tour, vertices):
        # Mark start time
        start_time = time.time()
        # Variables
        simple_tour = defaultdict(deque)
        simple_graph = defaultdict(dict)

        # TODO: Optimize this
        aux_tour = list(tour)
        aux_tour.append(tour[0])

        # Identify entrance and exit vertices
        for i, j in zip(aux_tour[:-1], aux_tour[1:]):
            for key in vertices:
                # Entrance
                if i not in vertices[key] and j in vertices[key]:
                    # simple_tour[key].append(j)
                    simple_tour[key].extend([i, j])
                # Exit
                if i in vertices[key] and j not in vertices[key]:
                    # simple_tour[key].extend([i, 'c'])
                    simple_tour[key].extend([i, j, 'c'])

        # Covert tour to simple graph
        for key in simple_tour:
            # rotate by 'c'
            p = list(reversed(simple_tour[key])).index('c')
            simple_tour[key].rotate(p)
            # print simple_tour[key]
            simple_tour[key] = list(simple_tour[key])
            simple_graph[key] = defaultdict(set)
            # Converts permutation to graph
            for i, j in zip(simple_tour[key][:-1], simple_tour[key][1:]):
                if not (i == 'c' or j == 'c'):
                    simple_graph[key][i].add(j)
                    simple_graph[key][j].add(i)
            # simple_graph[key] = dict(simple_graph[key])

        # Store execution time and return
        self._execution_time['gen_simple_graph'].append(time.time()-start_time)
        return dict(simple_graph)

    # Classify partitions feasibility by simple graph comparison
    def _classify(self, simple_graph_a, simple_graph_b):
        # Mark start time
        start_time = time.time()

        # Return Variables
        feasible_1 = set()
        feasible_2 = set()
        infeasible = set()

        for key in simple_graph_a:
            if simple_graph_a[key] == simple_graph_b[key]:
                feasible_1.add(key)
            else:
                infeasible.add(key)

        # Store execution time
        self._execution_time['classify'].append(time.time() - start_time)

        return feasible_1, feasible_2, infeasible

    # Try fusion of infeasible partitions
    def _fusion(self, partitions):

        # Mark start time
        start_time = time.time()

        # Fused partitions
        fused = set()

        # Start fusion try with 2 partitions
        n = 2
        while n < len(partitions['infeasible']):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(partitions['infeasible'], n):
                # Count common edges
                count = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    count += len(Graph(partitions['simple_graph_a'][i])
                                 & Graph(partitions['simple_graph_a'][j]))

                # Create element with (fusion, count)
                fusion = list(fusion)
                fusion.append(count)
                candidates.append(fusion)

            # Sort by common edges count
            candidates.sort(key=lambda fusion: fusion[n], reverse=True)
            # Increment fusion size
            n += 1
            # Discard common edges count
            for fusion in candidates:
                fusion.pop(-1)
            # Convert elements to tuples
            candidates = map(tuple, candidates)

            # Try fusions
            for fusion in candidates:
                union = defaultdict(set)
                # Test to determine if partition is fused already
                if not any(i in fused for i in fusion):
                    for i in fusion:
                        union[fusion] |= partitions['vertices'][i]

                    # Create simple graphs for fusion
                    simple_graph_1 = self._gen_simple_graph(self._tour_a,
                                                            union)
                    simple_graph_2 = self._gen_simple_graph(self._tour_b,
                                                            union)

                    # Check if fusion is feasible
                    f1, f2, _ = self._classify(simple_graph_1, simple_graph_2)

                    # Update information
                    if fusion in f1 or fusion in f2:
                        partitions['ab_cycles']['A'][fusion] = deque()
                        partitions['ab_cycles']['B'][fusion] = deque()
                        for i in fusion:
                            partitions['ab_cycles']['A'][fusion].extend(
                                               partitions['ab_cycles']['A'][i])
                            partitions['ab_cycles']['B'][fusion].extend(
                                               partitions['ab_cycles']['B'][i])
                            fused.add(i)
                            partitions['feasible_1'].update(f1)
                            partitions['feasible_2'].update(f2)
                            partitions['infeasible'].remove(i)

        # Fuse all remaining partitions
        if len(partitions['infeasible']) > 1:
            fusion = tuple(partitions['infeasible'])
            partitions['ab_cycles']['A'][fusion] = deque()
            partitions['ab_cycles']['B'][fusion] = deque()
            for i in fusion:
                partitions['ab_cycles']['A'][fusion].extend(
                                               partitions['ab_cycles']['A'][i])
                partitions['ab_cycles']['B'][fusion].extend(
                                               partitions['ab_cycles']['B'][i])
            partitions['feasible_1'].add(fusion)
        # Remaining partition is feasilbe 2?
        elif len(partitions['infeasible']) == 1:
            partitions['feasible_2'].update(partitions['infeasible'])
            # Clear infeasible
            partitions['infeasible'].clear()

        # Store execution time
        self._execution_time['fusion'].append(time.time() - start_time)

    # Build solutions
    def _build(self, partitions, tour_dist, common_graph):
        # Mark start time
        start_time = time.time()

        # Resete time
        self._execution_time = defaultdict(list)

        # dists of each partition in each solution
        dists = dict()
        dists['A'] = defaultdict(float)
        dists['B'] = defaultdict(float)

        # Set to store partial solution (cycle, key)s
        solution = set()

        # Partition with minor diff
        minor = {'key': None, 'value': None}

        # Get distance of all partitions tours
        for key in partitions['feasible_1'] | partitions['feasible_2']:
            dists['A'][key] += self._data.ab_cycle_dist(
                                             partitions['ab_cycles']['A'][key])
            dists['B'][key] += self._data.ab_cycle_dist(
                                             partitions['ab_cycles']['B'][key])

            # Distance diference inside AB_cycle
            diff = abs(dists['A'][key] - dists['B'][key])

            # Store AB_cycle with minor diference
            if not minor['key']:
                minor['key'] = key
                minor['value'] = diff
            elif diff < minor['value']:
                minor['key'] = key
                minor['value'] = diff

            # Chose best edges in each partition
            if dists['A'][key] <= dists['B'][key]:
                solution.add(tuple(['A', key]))
            else:
                solution.add(tuple(['B', key]))

        # Create each solution
        solution_1 = list()
        solution_2 = list()
        sum_1 = 0

        for cycle, key in solution:
            solution_1.extend(partitions['ab_cycles'][cycle][key])
            sum_1 += dists[cycle][key]
            if key != minor['key']:
                solution_2.extend(partitions['ab_cycles'][cycle][key])
            elif cycle == 'A':
                solution_2.extend(partitions['ab_cycles']['B'][minor['key']])
            else:
                solution_2.extend(partitions['ab_cycles']['A'][minor['key']])

        # Total sum of one partial solution
        sum_A = sum(dists['A'].values())

        # Calc common edges distance
        common_dist = tour_dist - sum_A

        # Create solutions graphs
        graph_1 = defaultdict(set)
        graph_2 = defaultdict(set)

        # Create solution 1 undirected graph
        for i, j in zip(solution_1[0::2], solution_1[1::2]):
            graph_1[abs(i)].add(abs(j))
            graph_1[abs(j)].add(abs(i))

        # Create solution 1 undirected graph
        for i, j in zip(solution_2[0::2], solution_2[1::2]):
            graph_2[abs(i)].add(abs(j))
            graph_2[abs(j)].add(abs(i))

        graph_1 = Graph(graph_1) | Graph(common_graph)
        graph_2 = Graph(graph_2) | Graph(common_graph)

        # Create tours from solutions graph
        vertices_1, tour_1 = Graph.dfs(graph_1, 1)
        vertices_2, tour_2 = Graph.dfs(graph_2, 1)

        # Verify infeasible tours
        assert len(vertices_1) == self._data.dimension
        assert len(vertices_2) == self._data.dimension

        # Store execution time
        self._execution_time['buil'].append(time.time() - start_time)

        # Create solutions
        return (tour_1, common_dist + sum_1), \
               (tour_2, common_dist + sum_1 + minor['value'])

    # Partition Crossover
    def recombine(self, parent_1, parent_2):
        # Mark start time
        start_time = time.time()

        # Duplicated solutions
        if parent_1 == parent_2:
            return parent_1, parent_2

        # Tours
        tour_a = list(parent_1.tour)
        tour_b = list(parent_2.tour)
        tour_c = list(parent_2.tour)

        # Undirected union graph (G*)
        undirected_graph = (parent_1.undirected_graph
                            | parent_2.undirected_graph)

        for vertice in undirected_graph:
            # Create ghost nodes for degree 4 nodes
            if len(undirected_graph[vertice]) == 4:
                tour_a.insert(tour_a.index(vertice) + 1, -vertice)
                tour_b.insert(tour_b.index(vertice) + 1, -vertice)
                tour_c.insert(tour_c.index(vertice), -vertice)
            # Remove degree 2 nodes (surrogate edge)
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

        # If exists one or no partition, return parents
        if len(vertices_m) <= 1 and len(vertices_n) <= 1:
            return parent_1, parent_2

        # Generate simple graphs for each partitioning scheme for each tour
        simple_graph_a_m = self._gen_simple_graph(tour_a, vertices_m)
        simple_graph_b_m = self._gen_simple_graph(tour_b, vertices_m)

        simple_graph_a_n = self._gen_simple_graph(tour_a, vertices_n)
        simple_graph_c_n = self._gen_simple_graph(tour_c, vertices_n)

        # Test simple graphs to identify feasible partitions
        feasible_1_m, feasible_2_m, infeasible_m = \
            self._classify(simple_graph_a_m, simple_graph_b_m)

        feasible_1_n, feasible_2_n, infeasible_n = \
            self._classify(simple_graph_a_n, simple_graph_c_n)

        # Score partitions scheme
        score_m = (len(feasible_1_m) + len(feasible_2_m)
                   + len(infeasible_m) * self._infeasible_weight)

        score_n = (len(feasible_1_n) + len(feasible_2_n)
                   + len(infeasible_n) * self._infeasible_weight)

        # Store better partitioning scheme
        partitions = dict()
        if score_m >= score_n:
            partitions['feasible_1'] = feasible_1_m
            partitions['feasible_2'] = feasible_2_m
            partitions['infeasible'] = infeasible_m
            partitions['vertices'] = vertices_m
            partitions['ab_cycles'] = ab_cycles_m
            partitions['simple_graph_a'] = simple_graph_a_m
            partitions['simple_graph_b'] = simple_graph_b_m
        else:
            partitions['feasible_1'] = feasible_1_n
            partitions['feasible_2'] = feasible_2_n
            partitions['infeasible'] = infeasible_n
            partitions['vertices'] = vertices_n
            partitions['ab_cycles'] = ab_cycles_n
            partitions['simple_graph_a'] = simple_graph_a_n
            partitions['simple_graph_b'] = simple_graph_c_n
            tour_b = tour_c

        # Constructed tours
        self._tour_a = tour_a
        self._tour_b = tour_b

        # Try to fuse infeasible partitions
        self._fusion(partitions)

        # After fusion, if exists one or no partition, return parents
        if len(partitions['feasible_1']) + len(partitions['feasible_2']) <= 1:
            return parent_1, parent_2

        # Store partitioning data
        self._partitions = partitions

        # Common graph
        if self._data:
            # Create solutions
            common_graph = (parent_1.undirected_graph
                            & parent_2.undirected_graph)
            inf_1, inf_2 = self._build(partitions, parent_1.dist, common_graph)
            # Measure time
            self._execution_time['recombine'].append(time.time() - start_time)
            # Return created solutions
            return Chromosome(*inf_1), Chromosome(*inf_2)

        # Store total execution time
        self._execution_time['recombine'].append(time.time() - start_time)
