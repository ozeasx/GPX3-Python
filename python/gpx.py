#!/usr/bin/python
# ozeasx@gmail.com

import time
from collections import defaultdict
from collections import deque
from itertools import combinations
from operator import attrgetter
from graph import Graph
from chromosome import Chromosome


# Generalized partition crossover operator
class GPX(object):
    # Class initialization
    def __init__(self, data=None):
        # dataset to compute distances
        self._data = data
        # Partitions types values
        self._f1_weight = 1
        self._f2_weight = 2
        self._f3_weight = 3
        self._infeasible_weight = 0.4
        # Tests 1, 2 and 3 for partitioning
        self._test_1 = True
        self._test_2 = False
        self._test_3 = False
        # Tests 1, 2 and 3 for fusion
        self._test_1_fusion = True
        self._test_2_fusion = False
        self._test_3_fusion = False
        # Parents tours
        self._parent_1_tour = None
        self._parent_2_tour = None
        # Partitioning information
        self._info = dict()
        # Counters
        self._counters = defaultdict(int)
        # Dict with lists containing execution time of each step
        self._timers = defaultdict(list)

    # Getters -----------------------------------------------------------------

    @property
    def f1_weight(self):
        return self._f1_weight

    @property
    def f2_weight(self):
        return self._f2_weight

    @property
    def f3_weight(self):
        return self._f3_weight

    @property
    def infeasible_weight(self):
        return self.infeasible_weight

    @property
    def test_1(self):
        return self._test_1

    @property
    def test_2(self):
        return self._test_2

    @property
    def test_3(self):
        return self._test_3

    @property
    def test_1_fusion(self):
        return self._test_1_fusion

    @property
    def test_2_fusion(self):
        return self._test_2_fusion

    @property
    def test_3_fusion(self):
        return self._test_3_fusion

    @property
    def info(self):
        return self._info

    @property
    def counters(self):
        return self._counters

    @property
    def timers(self):
        return self._timers

    # Setters -----------------------------------------------------------------

    @f1_weight.setter
    def f1_weight(self, value):
        assert 0 < value <= 1, "f1 weight must be in ]0,1] interval"
        self._f1_weight = value

    @f2_weight.setter
    def f2_weight(self, value):
        assert 0 < value <= 1, "f2 weight must be in ]0,1] interval"
        self._f2_weight = value

    @f3_weight.setter
    def f3_weight(self, value):
        assert 0 < value <= 1, "f3 weight must be in ]0,1] interval"
        self._f3_weight = value

    @infeasible_weight.setter
    def infeasible_weight(self, value):
        assert 0 < value <= 1, "Infeasible weight must be in ]0,1] interval"
        self._infeasible_weight = value

    @test_1.setter
    def test_1(self, value):
        assert value in [True, False]
        self._test_1 = value

    @test_2.setter
    def test_2(self, value):
        assert value in [True, False]
        self._test_2 = value

    @test_3.setter
    def test_3(self, value):
        assert value in [True, False]
        self._test_3 = value

    @test_1_fusion.setter
    def test_1_fusion(self, value):
        assert value in [True, False]
        self._test_1_fusion = value

    @test_2_fusion.setter
    def test_2_fusion(self, value):
        assert value in [True, False]
        self._test_2_fusion = value

    @test_3_fusion.setter
    def test_3_fusion(self, value):
        assert value in [True, False]
        self._test_3_fusion = value

# =============================================================================

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
        self._timers['partitioning'].append(time.time() - start_time)
        # Return vertice set and ab_cycles
        return vertices, ab_cycles

# =============================================================================

    # Create the simple graph for all partitions for given tour
    def _gen_simple_graph(self, tour, vertices):
        # Mark start time
        start_time = time.time()
        # Simplified graph
        simple_g = defaultdict(dict)

        # Create inner, outter and common graphs
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
                previous = tour[i-1]
                next = tour[(i+1) % size]
                if not first:
                    first = previous, current, next
                    last = current
                    continue
                # Entrance vertice
                if previous not in vertices[key]:
                    simple_g[key]['out'].add(tuple(sorted([last, current])))
                    simple_g[key]['common'].add(tuple(sorted([previous,
                                                              current])))
                    last = current
                    continue
                # Exit vertice
                if next not in vertices[key]:
                    simple_g[key]['in'].add(tuple(sorted([last, current])))
                    simple_g[key]['common'].add(tuple(sorted([current, next])))
                    last = current
            # Close outter graph
            if first[0] not in vertices[key]:  # previous not in
                simple_g[key]['out'].add(tuple(sorted([last, first[1]])))
                simple_g[key]['common'].add(tuple(sorted([first[0],
                                                          first[1]])))
            # Close inner graph
            if first[2] not in vertices[key]:  # next not in
                simple_g[key]['in'].add(tuple(sorted([last, first[1]])))
                simple_g[key]['common'].add(tuple(sorted([first[2],
                                                          first[1]])))
        # Store execution time
        self._timers['simple_graph'].append(time.time() - start_time)
        # Return constructed graphs
        return dict(simple_g)

# =============================================================================

    # Classify partitions feasibility by inner and outter graph comparison
    def _classify(self, simple_a, simple_b, fusion=False):
        # Mark start time
        start_time = time.time()

        # Return Variables
        feasible = defaultdict(set)
        infeasible = set()
        # Avoid problems with set.union(*feasible.values())
        feasible[0] = set()
        # Tests conditions
        if not fusion:
            f1 = self._test_1
            f2 = self._test_2
            f3 = self._test_3
        else:
            f1 = self._test_1_fusion
            f2 = self._test_2_fusion
            f3 = self._test_3_fusion

        for key in simple_a:
            # Inner graph test (Test 1)
            if f1 and (simple_a[key]['in'] == simple_b[key]['in']):
                feasible[1].add(key)
            # Outter graph test (Test 2)
            elif f2 and (simple_a[key]['out'] == simple_b[key]['out']):
                feasible[2].add(key)
            # All graphs test (Test 3)
            elif f3 and not (simple_a[key]['in'] & simple_b[key]['out']
                             or simple_a[key]['out'] & simple_b[key]['in']):
                feasible[3].add(key)
            else:
                infeasible.add(key)

        # Store execution time
        self._timers['classification'].append(time.time() - start_time)

        # Return classified partitions
        return feasible, infeasible

# =============================================================================

    # Try fusion of infeasible partitions
    def _fusion(self, info):

        # Mark start time
        start_time = time.time()

        # Fused partitions
        fused = set()

        # Function to fuse ab_cycles
        def fuse(fusion, dest):
            info['ab_cycles']['A'][fusion] = deque()
            info['ab_cycles']['B'][fusion] = deque()
            for i in fusion:
                info['ab_cycles']['A'][fusion].extend(
                                                     info['ab_cycles']['A'][i])
                info['ab_cycles']['B'][fusion].extend(
                                                     info['ab_cycles']['B'][i])
                fused.add(i)
                info['infeasible'].remove(i)
            if dest == 'infeasible':
                info[dest].add(fusion)
            else:
                info['feasible'][dest].add(fusion)

        # Start fusion try with 2 partitions
        n = 2
        while n <= len(info['infeasible']):
            # Create all combinations of n size
            candidates = list()
            for fusion in combinations(info['infeasible'], n):
                # Count common edges
                count = 0
                for i, j in zip(fusion[:-1], fusion[1:]):
                    count += len(info['simple_a'][i]['common']
                                 & info['simple_a'][j]['common'])

                # Create element with (fusion, count)
                if count > 1:
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
            for test in candidates:
                # Partitions vertices union
                union = defaultdict(set)
                # Test to determine if partition is fused already
                if not any(i in fused for i in test):
                    for i in test:
                        union[test] |= info['vertices'][i]

                    # Pause time count
                    start_time = time.time() - start_time
                    # Create simple graphs for fusion
                    simple_a = self._gen_simple_graph(info['tour_a'], union)
                    simple_b = self._gen_simple_graph(info['tour_b'], union)
                    # Classify fusion
                    feasible, _ = self._classify(simple_a, simple_b, True)
                    # Resume time count
                    start_time = time.time() - start_time

                    # Update information if successfull fusion
                    if test in set.union(*feasible.values()):
                        info['simple_a'][test] = simple_a[test]
                        info['simple_b'][test] = simple_b[test]
                        for key in feasible:
                            if test in feasible[key]:
                                fuse(test, key)
                        # Update counters
                        self._counters['fusion_1'] += len(feasible[1])
                        self._counters['fusion_2'] += len(feasible[2])
                        self._counters['fusion_3'] += len(feasible[3])
                        self._counters['fusion'] += (len(feasible[1])
                                                     + len(feasible[2])
                                                     + len(feasible[3]))

        # Fuse all remaining partitions in one infeasible partition to be
        # handled by build method. The last of the mohicans remains infeasible
        # to be handled as well
        if len(info['infeasible']) > 1:
            if self._test_2 or self._test_3:
                self._counters['unsolved'] += len(info['infeasible'])
                fuse(tuple(info['infeasible']), 'infeasible')
            else:
                # All remaining partitions after f1 and fusion are feasible 2?
                fuse(tuple(info['infeasible']), 2)
        # The last of the mohicans
        elif len(info['infeasible']) == 1:
            if self._test_2 or self._test_3:
                self._counters['unsolved'] += 1
            else:
                # All remaining partitions after f1 and fusion are feasible 2?
                info['feasible'][2].add(info['infeasible'].pop())

        # Store execution time
        self._timers['fusion'].append(time.time() - start_time)

# =============================================================================

    # Build solutions
    def _build(self, info, common_graph, tour_1_dist, tour_2_dist):
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
        # Infeasible partition
        inf_key = None

        # Get distance of all partitions tours (feasible and infeasible)
        for key in info['feasible'][0] | info['infeasible']:
            dists['A'][key] += self._data.ab_cycle_dist(
                                                   info['ab_cycles']['A'][key])
            dists['B'][key] += self._data.ab_cycle_dist(
                                                   info['ab_cycles']['B'][key])

            # Feasible partitions
            if key in info['feasible'][0]:
                # Distance diference inside AB_cycle
                diff = abs(dists['A'][key] - dists['B'][key])
                # Save AB_cycle with minor diference
                if not minor_key:
                    minor_key = key
                    minor_diff = diff
                elif diff < minor_diff:
                    minor_key = key
                    minor_diff = diff
                # Chose best path in each feasible partition
                if dists['A'][key] <= dists['B'][key]:
                    solution.add(tuple(['A', key]))
                else:
                    solution.add(tuple(['B', key]))
            # Infeasible partitions
            else:
                inf_key = key
                inf_cycle_a = info['ab_cycles']['A'][key]
                inf_cycle_b = info['ab_cycles']['B'][key]

        # Create base solutions without infeasible partitions
        base_1 = list()
        base_2 = list()
        # Add common distance
        base_1_dist = tour_1_dist - sum(dists['A'].values())

        # Feasible part
        for cycle, key in solution:
            base_1.extend(info['ab_cycles'][cycle][key])
            base_1_dist += dists[cycle][key]
            if key != minor_key:
                base_2.extend(info['ab_cycles'][cycle][key])
            elif cycle == 'A':
                base_2.extend(info['ab_cycles']['B'][minor_key])
            else:
                base_2.extend(info['ab_cycles']['A'][minor_key])

        # Calc base2 and common edges distances
        base_2_dist = base_1_dist + minor_diff

        # Graphs without infeasible partitions
        graphs = list()
        distances = list()

        graphs.append(Graph.gen_undirected_ab_graph(base_1) | common_graph)
        graphs.append(Graph.gen_undirected_ab_graph(base_2) | common_graph)
        distances.extend([base_1_dist, base_2_dist])

        # Graphs with infeasible partitions
        if inf_key:
            for graph, dist in zip(graphs, distances):
                graphs.append(Graph.gen_undirected_ab_graph(inf_cycle_a)
                              | graph)
                graphs.append(Graph.gen_undirected_ab_graph(inf_cycle_b)
                              | graph)
                distances.extend([dist + dists['A'][inf_key],
                                  dist + dists['B'][inf_key]])

        # Candidates solutions
        candidates = list()
        # Builder
        for graph, dist in zip(graphs, distances):
            vertices, tour = Graph.dfs(graph, 1)
            # Feasible tour?
            if len(vertices) == len(self._parent_1_tour):
                if dist <= max(tour_1_dist, tour_2_dist):
                    candidates.append([tour, dist])
                # A bad child was created with only feasible partitions?
                elif not inf_key:
                    self._counters['bad_child'] += 1
            # An infeasible tour was created with only feasible partitions?
            elif not inf_key:
                self._counters['inf_tour'] += 1
                # print tour, self._parent_1_tour, self._parent_2_tour
                # exit()

        # Store execution time
        self._timers['build'].append(time.time() - start_time)
        # Return created tours information
        return candidates

# =============================================================================

    # Partition Crossover
    def recombine(self, parent_1, parent_2):
        # Mark start time
        start_time = time.time()

        # Store parent sum
        self._counters['parents_dist'] += (parent_1.dist + parent_2.dist)

        # Duplicated solutions
        if parent_1 == parent_2:
            self._counters['children_dist'] += (parent_1.dist + parent_2.dist)
            return parent_1, parent_2

        # Save parents tours
        self._parent_1_tour = parent_1.tour
        self._parent_2_tour = parent_2.tour

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
        m = dict()
        n = dict()

        m['vertices'], m['ab_cycles'] = self._partition(undirected_a,
                                                        undirected_b)
        n['vertices'], n['ab_cycles'] = self._partition(undirected_a,
                                                        undirected_c)

        # If exists one or no partition, return parents
        if len(m['vertices']) <= 1 and len(n['vertices']) <= 1:
            self._counters['failed'] += 1
            self._counters['children_dist'] += (parent_1.dist + parent_2.dist)
            return parent_1, parent_2

        # Generate simple graphs for each partitioning scheme for each tour
        m['simple_a'] = self._gen_simple_graph(tour_a, m['vertices'])
        m['simple_b'] = self._gen_simple_graph(tour_b, m['vertices'])

        n['simple_a'] = self._gen_simple_graph(tour_a, n['vertices'])
        n['simple_b'] = self._gen_simple_graph(tour_c, n['vertices'])

        # Test simple graphs to identify feasible partitions
        m['feasible'], m['infeasible'] = self._classify(m['simple_a'],
                                                        m['simple_b'])
        n['feasible'], n['infeasible'] = self._classify(n['simple_a'],
                                                        n['simple_b'])

        # Score partitions scheme
        score_m = (len(m['feasible'][1]) * self._f1_weight
                   + len(m['feasible'][2]) * self._f2_weight
                   + len(m['feasible'][3]) * self._f3_weight
                   + len(m['infeasible']) * self._infeasible_weight)

        score_n = (len(n['feasible'][1]) * self._f1_weight
                   + len(n['feasible'][2]) * self._f2_weight
                   + len(n['feasible'][3]) * self._f3_weight
                   + len(n['infeasible']) * self._infeasible_weight)

        # Choose better partitioning scheme
        if score_m >= score_n:
            info = m
            info['tour_b'] = tour_b
        else:
            info = n
            info['tour_b'] = tour_c

        info['tour_a'] = tour_a
        info['feasible'][0] = set.union(*info['feasible'].values())
        # Counters
        self._counters['feasible'] += len(info['feasible'][0])
        self._counters['feasible_1'] += len(info['feasible'][1])
        self._counters['feasible_2'] += len(info['feasible'][2])
        self._counters['feasible_3'] += len(info['feasible'][3])
        self._counters['infeasible'] += len(info['infeasible'])

        # Try to fuse infeasible partitions
        if len(info['infeasible']) > 1:
            self._fusion(info)
            # Update feasible partitions
            info['feasible'][0] = set.union(*info['feasible'].values())

        # After fusion, if exists one or no partition, return parents
        if len(info['feasible'][0]) + len(info['infeasible']) <= 1:
            self._counters['failed'] += 1
            self._counters['children_dist'] += (parent_1.dist + parent_2.dist)
            return parent_1, parent_2

        # Save partitioning data
        self._info = info

        # Build solutions if there is instance data
        if self._data:
            # Common graph
            common_graph = (parent_1.undirected_graph
                            & parent_2.undirected_graph)
            # Build solutions
            constructed = self._build(info, common_graph, parent_1.dist,
                                      parent_2.dist)
            # Fail if no tour constructed
            if len(constructed) == 0:
                self._counters['failed'] += 1
                self._counters['children_dist'] += (parent_1.dist
                                                    + parent_2.dist)
                return parent_1, parent_2
            # Make sure GPX returns two solutions
            candidates = set([parent_1, parent_2])
            # Add constructed solutions
            for tour, dist in constructed:
                candidates.add(Chromosome(tour, dist))
            # Sort by distance, get two of them
            candidates = tuple(sorted(candidates, key=attrgetter('dist'))[:2])
            # Improvement assertion
            parents_dist = self._counters['parents_dist']
            children_dist = candidates[0].dist + candidates[1].dist
            assert children_dist <= parents_dist, (parent_1.tour,
                                                   parent_2.tour,
                                                   "Improvement assertion")
            if children_dist < parents_dist:
                self._counters['improved'] += 1
            # To calc total improvement
            self._counters['children_dist'] += children_dist
            # Measure execution time
            self._timers['recombination'].append(time.time() - start_time)
            # Return created solutions
            return candidates

        # Store total execution time
        self._timers['recombination'].append(time.time() - start_time)
