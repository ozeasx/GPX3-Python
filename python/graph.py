#!/usr/bin/python
# ozeasx@gmail.com

from collections import defaultdict
from collections import deque


# Class to provide graph and edge generation and dict operators overload
class Graph(dict):

    # Insert node between nodes a and b
    def insert(self, node, left=False):
        if left:
            a, _ = sorted(self[-node])
            b = -node
        else:
            _, b = sorted(self[-node])
            a = -node

        self[node] = set([a, b])
        self[a].remove(b)
        self[b].remove(a)
        self[a].add(node)
        self[b].add(node)

    # Remove node
    def remove(self, node):
        a = self[node].pop()
        b = self[node].pop()
        del self[node]
        self[a].remove(node)
        self[b].remove(node)
        self[a].add(b)
        self[b].add(a)

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

    # Generate graph of ab_cycles
    @staticmethod
    def gen_undirected_ab_graph(ab_cycle):
        result = defaultdict(set)
        ab_cycle = list(ab_cycle)
        for i, j in zip(ab_cycle[0::2], ab_cycle[1::2]):
            result[abs(i)].add(abs(j))
            result[abs(j)].add(abs(i))

        return Graph(result)

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

        return Graph(result)
