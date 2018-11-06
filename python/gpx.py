#!/usr/bin/python
# ozeasx@gmail.com

import time
from collections import defaultdict
from collections import deque
from itertools import combinations
from graph import Graph
from chromosome import Chromosome


# Generalized partition crossover operator
class GPX(object):
    # Class initialization
    def __init__(self, data=None):
        # dataset to compute distances
        self._data = data
        # Infeasible value compared to feasible partitions
        self._infeasible_weight = 0.4
        # Limit fusion trys (unused)
        self._fusion_limit = False
        # Dict with lists containing execution time of each step
        self._exec_time = defaultdict(list)
        # Partitioning data
        self._partitions = dict()
        # Tours created for partitioning
        self._tour_a = None
        self._tour_b = None
        # Count failed and suceeded partitioning
        self._failed = 0

    @property
    def infeasible_weight(self):
        return self._infeasible_weight

    @property
    def fusion_limit(self):
        return self._fusion_limit

    @property
    def exec_time(self):
        return self._exec_time

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

    @property
    def failed(self):
        return self._failed

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
                if ab_cycles['A'][index][1] \
                   in graph_b[ab_cycles['A'][index][0]]:
                    ab_cycles['A'][index].rotate(1)
                else:
                    ab_cycles['B'][index].rotate(1)
            # Reduce loop
            loop -= vertices[index]
            # Increment index
            index += 1
        # Store execution time and return
        self._exec_time['partition'].append(time.time() - start_time)
        # Return vertice set and ab_cycles
        return vertices, ab_cycles

    # Create the simple graph for all partitions for given tour
    def _gen_simple_graph(self, tour, vertices):
        # Mark start time
        start_time = time.time()
        # Simplified graph
        simple_g = defaultdict(dict)

        # Create inner, outer and common graphs
        size = len(tour)
        for key in vertices:
            simple_g[key]['in'] = set()  # inner simplified graph
            simple_g[key]['out'] = set()  # outter simplified graph
            simple_g[key]['common'] = set()  # common graph
            last = None
            first = None
            for i, current in enumerate(tour):
                if current not in vertices[key]:
                    continue
                prev = tour[i-1]
                next = tour[(i+1) % size]
                if not first:
                    first = prev, current, next
                    last = current
                    continue
                # Entrance vertice
                if prev not in vertices[key]:
                    simple_g[key]['out'].add(frozenset([last, current]))
                    simple_g[key]['common'].add(frozenset([prev, current]))
                    last = current
                    continue
                # Exit vertice
                if next not in vertices[key]:
                    simple_g[key]['in'].add(frozenset([last, current]))
                    simple_g[key]['common'].add(frozenset([current, next]))
                    last = current
            # Close graphs
            if first[0] not in vertices[key]:  # previous not in
                simple_g[key]['out'].add(frozenset([last, first[1]]))
                simple_g[key]['common'].add(frozenset([first[0], first[1]]))
            if first[2] not in vertices[key]:  # next not in
                simple_g[key]['in'].add(frozenset([last, first[1]]))
                simple_g[key]['common'].add(frozenset([first[2], first[1]]))

        # Store execution time and return constructed graphs
        self._exec_time['simple graph'].append(time.time() - start_time)
        return dict(simple_g)

    # Classify partitions feasibility by inner and outter graph comparison
    def _classify(self, simple_graph_a, simple_graph_b):
        # Mark start time
        start_time = time.time()

        # Return Variables
        feasible = set()
        infeasible = set()

        for key in simple_graph_a:
            # Inner graph
            if simple_graph_a[key]['in'] == simple_graph_b[key]['in']:
                feasible.add(key)
                continue
            # Outter graph
            if simple_graph_a[key]['out'] == simple_graph_b[key]['out']:
                feasible.add(key)
                continue
            infeasible.add(key)

        # Store execution time
        self._exec_time['classify'].append(time.time() - start_time)

        # Return classified partitions
        return feasible, infeasible

    # Try fusion of infeasible partitions
    def _fusion(self, partitions):

        # Function to fuse ab_cycles
        def fuse(fusion, dest='feasible'):
            partitions['ab_cycles']['A'][fusion] = deque()
            partitions['ab_cycles']['B'][fusion] = deque()
            for i in fusion:
                partitions['ab_cycles']['A'][fusion].extend(
                                   partitions['ab_cycles']['A'][i])
                partitions['ab_cycles']['B'][fusion].extend(
                                   partitions['ab_cycles']['B'][i])
                fused.add(i)
                partitions['infeasible'].remove(i)
            partitions[dest].add(fusion)

        # Mark start time
        start_time = time.time()

        # Fused partitions
        fused = set()

        # Start fusion try with 2 partitions
        n = 2
        while n <= len(partitions['infeasible']):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(partitions['infeasible'], n):
                # Count common edges
                count = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    count += len(partitions['simple_graph_a'][i]['common']
                                 & partitions['simple_graph_a'][j]['common'])

                # Create element with (fusion, count)
                if count:
                    candidates.append(list(fusion) + [count])

            # Sort by common edges count
            candidates.sort(key=lambda fusion: fusion[n], reverse=True)
            # Discard common edges count
            for fusion in candidates:
                fusion.pop(-1)
            # Convert elements to tuples to be used as dict keys
            candidates = map(tuple, candidates)
            # Increment fusion size
            n += 1
            # Try fusions
            for fusion in candidates:
                # Partitions vertices union
                union = defaultdict(set)
                # Test to determine if partition is fused already
                if not any(i in fused for i in fusion):
                    for i in fusion:
                        union[fusion] |= partitions['vertices'][i]

                    # Pause time count
                    start_time = time.time() - start_time
                    # Create simple graphs for fusion
                    simple_g_a = self._gen_simple_graph(self._tour_a, union)
                    simple_g_b = self._gen_simple_graph(self._tour_b, union)
                    # Classify fusion
                    feasible, _ = self._classify(simple_g_a, simple_g_b)
                    # Resume time count
                    start_time = time.time() - start_time

                    # Update information if successfull fusion
                    if fusion in feasible:
                        fuse(fusion)

        # Fuse all remaining partitions in one infeasible partition to be
        # handled by build method. The last of the mohicans remains infeasible.
        if len(partitions['infeasible']) > 1:
            fuse(tuple(partitions['infeasible']), 'infeasible')

        # Store execution time
        self._exec_time['fusion'].append(time.time() - start_time)

    # Build solutions
    def _build(self, partitions, common_graph, tour_dist):
        # Mark start time
        start_time = time.time()

        # dists of each partition in each solution
        dists = dict()
        dists['A'] = defaultdict(float)
        dists['B'] = defaultdict(float)

        # Set to store partial solution (cycle, key)s
        solution = set()

        # Partition with minor inside diff
        minor_key = None
        minor_diff = None
        inf_key = None

        # Get distance of all partitions tours
        for key in partitions['feasible'] | partitions['infeasible']:
            dists['A'][key] += self._data.ab_cycle_dist(
                                             partitions['ab_cycles']['A'][key])
            dists['B'][key] += self._data.ab_cycle_dist(
                                             partitions['ab_cycles']['B'][key])

            if key in partitions['infeasible']:
                inf_key = key
                inf_cycle_a = partitions['ab_cycles']['A'][key]
                inf_cycle_b = partitions['ab_cycles']['B'][key]
            else:
                # Distance diference inside AB_cycle
                diff = abs(dists['A'][key] - dists['B'][key])
                # Store AB_cycle with minor diference
                if not minor_key:
                    minor_key = key
                    minor_diff = diff
                elif diff < minor_diff:
                    minor_key = key
                    minor_diff = diff

                # Chose best edges in each partition
                if dists['A'][key] <= dists['B'][key]:
                    solution.add(tuple(['A', key]))
                else:
                    solution.add(tuple(['B', key]))

        # Create base solutions without infeasible part
        base_1 = list()
        base_2 = list()
        base_1_dist = 0

        # Feasible part
        for cycle, key in solution:
            base_1.extend(partitions['ab_cycles'][cycle][key])
            base_1_dist += dists[cycle][key]
            if key != minor_key:
                base_2.extend(partitions['ab_cycles'][cycle][key])
            elif cycle == 'A':
                base_2.extend(partitions['ab_cycles']['B'][minor_key])
            else:
                base_2.extend(partitions['ab_cycles']['A'][minor_key])

        # Calc base2 and common edges distances
        base_2_dist = base_1_dist + minor_diff
        common_dist = tour_dist - sum(dists['A'].values())

        # Create base graphs
        graph_1 = Graph.gen_undirected_ab_graph(base_1) | common_graph
        graph_2 = Graph.gen_undirected_ab_graph(base_2) | common_graph

        # [[ab_cycles, graphs, vertices, tours]]
        if inf_key:
            inf_graph_a = Graph.gen_undirected_ab_graph(inf_cycle_a)
            inf_graph_b = Graph.gen_undirected_ab_graph(inf_cycle_b)

            candidates = list()
            i = 0
            for graph, base_dist in zip([graph_1, graph_2], [base_1_dist,
                                                             base_2_dist]):
                for inf_graph, inf_dist in zip([inf_graph_a, inf_graph_b],
                                               [dists['A'][inf_key],
                                                dists['B'][inf_key]]):
                    vertices, tour = Graph.dfs(graph | inf_graph, 1)
                    if len(vertices) == self._data.dimension:
                        candidates.append(list())
                        candidates[i] = [base_dist + inf_dist, vertices, tour]
                        i += 1
            # Two solutions should be feasible at least
            assert len(candidates) >= 2, len(candidates)
            candidates.sort(key=lambda s: s[0])
            dist_1, vertices_1, tour_1 = candidates[0]
            dist_2, vertices_2, tour_2 = candidates[1]
        else:
            # Create tours from solutions graph
            dist_1 = base_1_dist
            dist_2 = base_2_dist
            vertices_1, tour_1 = Graph.dfs(graph_1, 1)
            vertices_2, tour_2 = Graph.dfs(graph_2, 1)

        # Verify infeasible tours
        assert len(vertices_1) == self._data.dimension, (self._tour_a,
                                                         self._tour_b)
        assert len(vertices_2) == self._data.dimension, (self._tour_a,
                                                         self._tour_b)

        # Store execution time
        self._exec_time['build'].append(time.time() - start_time)

        # Return created solutions
        return (tour_1, dist_1 + common_dist), (tour_2, dist_2 + common_dist)

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
            self._failed += 1
            return parent_1, parent_2

        # Generate simple graphs for each partitioning scheme for each tour
        simple_graph_a_m = self._gen_simple_graph(tour_a, vertices_m)
        simple_graph_b_m = self._gen_simple_graph(tour_b, vertices_m)

        simple_graph_a_n = self._gen_simple_graph(tour_a, vertices_n)
        simple_graph_c_n = self._gen_simple_graph(tour_c, vertices_n)

        # Test simple graphs to identify feasible partitions
        feasible_m, infeasible_m = self._classify(simple_graph_a_m,
                                                  simple_graph_b_m)

        feasible_n, infeasible_n = self._classify(simple_graph_a_n,
                                                  simple_graph_c_n)

        # Score partitions scheme
        score_m = len(feasible_m) + len(infeasible_m) * self._infeasible_weight

        score_n = len(feasible_n) + len(infeasible_n) * self._infeasible_weight

        # Store better partitioning scheme
        partitions = dict()
        if score_m >= score_n:
            partitions['feasible'] = feasible_m
            partitions['infeasible'] = infeasible_m
            partitions['vertices'] = vertices_m
            partitions['ab_cycles'] = ab_cycles_m
            partitions['simple_graph_a'] = simple_graph_a_m
            partitions['simple_graph_b'] = simple_graph_b_m
        else:
            partitions['feasible'] = feasible_n
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
        if len(partitions['infeasible']):
            self._fusion(partitions)

        # After fusion, if exists one or no partition, return parents
        if len(partitions['feasible']) + len(partitions['infeasible']) <= 1:
            self._failed += 1
            return parent_1, parent_2

        # Store partitioning data
        self._partitions = partitions

        # Common graph
        if self._data:
            # Create solutions
            common_graph = (parent_1.undirected_graph
                            & parent_2.undirected_graph)
            inf_1, inf_2 = self._build(partitions, common_graph, parent_1.dist)
            # Measure time
            self._exec_time['recombine'].append(time.time() - start_time)
            # Return created solutions
            return Chromosome(*inf_1), Chromosome(*inf_2)

        # Store total execution time
        self._exec_time['recombine'].append(time.time() - start_time)
