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

                    # Pause time count
                    start_time = time.time() - start_time
                    # Create simple graphs for fusion
                    simple_graph_1 = self._gen_simple_graph(self._tour_a,
                                                            union)
                    simple_graph_2 = self._gen_simple_graph(self._tour_b,
                                                            union)
                    # Resume time count
                    start_time = time.time() - start_time

                    # Pause time count
                    start_time = time.time() - start_time
                    # Check if fusion is feasible
                    f1, f2, _ = self._classify(simple_graph_1, simple_graph_2)
                    # Resume time count
                    start_time = time.time() - start_time

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

        # Reset time
        # self._execution_time = defaultdict(list)

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
        self._execution_time['build'].append(time.time() - start_time)

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
