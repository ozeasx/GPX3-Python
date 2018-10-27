#!/usr/bin/python
# ozeasx@gmail.com

from collections import defaultdict
from collections import deque


# Class to provide graph and edge generation and dict operators overload
class Graph(dict):

    # Depth first search to find connected components
    # https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
    @staticmethod
    def dfs(graph, start):
        visited, stack, ab_cycle = set(), [start], deque()
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                ab_cycle.append(vertex)
                visited.add(vertex)
                stack.extend(graph[vertex] - visited)
        return visited, ab_cycle

    # Generates undirected edge set
    @staticmethod
    def gen_undirected_edges(tour):
        edges = set()
        for i, j in zip(tour[:-1], tour[1:]):
            edges.add(frozenset([i, j]))
        # Close circle
        edges.add(frozenset([tour[-1], tour[0]]))

        return frozenset(edges)

    # Generates undirected graph
    @staticmethod
    def gen_undirected_graph(tour):
        graph = defaultdict(set)
        for i, j in zip(tour[:-1], tour[1:]):
            graph[i].add(j)
            graph[j].add(i)
        # Close path
        graph[tour[-1]].add(tour[0])
        graph[tour[0]].add(tour[-1])

        return Graph(graph)

    # Graph union
    def __or__(self, other):
        result = dict.fromkeys(self.viewkeys() | other.viewkeys())

        for key in result:
            result[key] = self.get(key, set()) | other.get(key, set())

        return Graph(result)

    # Graph intersection
    def __and__(self, other):
        result = dict.fromkeys(self.viewkeys() & other.viewkeys())

        for key in result:
            result[key] = self.get(key, set()) & other.get(key, set())

        return Graph(result)

    # Graph simetric diferente (union - intersection)
    def __xor__(self, other):
        result = dict.fromkeys(self.viewkeys() | other.viewkeys())

        for key in result:
            result[key] = self.get(key, set()) ^ other.get(key, set())
            if not result[key]:
                print key

        return Graph(result)
