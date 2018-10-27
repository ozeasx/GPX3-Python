#!/usr/bin/python
# ozeasx@gmail.com

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
