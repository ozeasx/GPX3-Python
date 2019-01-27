#!/usr/bin/python
# ozeasx@gmail.com

import os
from itertools import combinations
from shell import Shell
from chromosome import Chromosome


class TSPLIB(object):
    def __init__(self, instance_path):
        # Set instance file and
        self._instance_path = instance_path
        self._instance_name = instance_path[:-4]
        self._best_solution = None

        # Set tsp dimension
        self._dimension = int(Shell.run("grep DIMENSION " + instance_path
                                        + " | cut -d':' -f2").strip())
        self._name = str(Shell.run("grep NAME " + instance_path
                                   + " | cut -d':' -f2").strip())

        # Condensed index mapping
        # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist/13079806
        self._cindex = lambda i, j: i*(2*self._dimension - i - 3)/2 + j - 1

        # Generate distance matrix file
        if not os.path.isfile(self._instance_path + ".dm"):
            print "Generating distance matrix..."
            Shell.call("../R/create_dm.r " + instance_path)
            print "Done..."
        else:
            print "Distance matrix file already exists"

        # Hash lockup
        self._hash = dict()
        # Distance matrix
        self._dm = list()
        line_number = 1
        with open(self._instance_path + ".dm") as dm:
            for t, dist in zip(combinations(xrange(self._dimension), 2), dm):
                self._hash[t] = line_number
                self._dm.append(float(dist))
                line_number += 1

        # Set best known solution, if exists
        tour_file = None
        if os.path.isfile(self._instance_name + ".opt.tour.new"):
            tour_file = self._instance_name + ".opt.tour.new"
        elif os.path.isfile(self._instance_name + ".opt.tour"):
            tour_file = self._instance_name + ".opt.tour"
        if tour_file:
            with open(tour_file) as best:
                do = False
                best_tour = list()
                for word in best:
                    # End loop if EOF
                    if word.strip() == "EOF":
                        break
                    # If do, store tour
                    if do:
                        best_tour.append(int(word))
                    # If TOUR_SECTION, set do to store solution
                    if word.strip() == "TOUR_SECTION":
                        do = True
            # Try to remove cycle close
            try:
                best_tour.remove(-1)
            except ValueError:
                pass
            # Store best solution
            self._best_solution = Chromosome(best_tour,
                                             self.tour_dist(best_tour))

    # Get instance dimension
    @property
    def dimension(self):
        return self._dimension

    # Get best known tour
    @property
    def best_solution(self):
        return self._best_solution

    # Set a new best tour and write to file
    @best_solution.setter
    def best_solution(self, solution):

        # Set best solution
        if self._best_solution is None:
            self._best_solution = solution
        elif solution.dist < self._best_solution.dist:
            self._best_solution = solution
        else:
            return

        # Write new solution to file
        with open(self._instance_name + ".opt.tour.new", 'w') as best:
            best.write("NAME : " + self._name + ".opt.tour.new\n")
            best.write("COMMENT : Length " + str(solution.dist)
                                           + ", ozeasx@gmail.com\n")
            best.write("TYPE : TOUR\n")
            best.write("DIMENSION : " + str(self._dimension) + "\n")
            best.write("TOUR_SECTION\n")
            for node in solution.tour:
                best.write(str(node) + "\n")
            best.write("-1\n")
            best.write("EOF\n")

    # Return nearest nodes from i in nodes set
    def get_nearest(self, i, nodes):
        last_dist = float("inf")
        nearest = None
        dist = None
        for j in nodes:
            t = sorted([i-1, j-1])
            dist = self._dm[self._cindex(*t)]
            if dist < last_dist:
                nearest = j
                last_dist = dist
        return nearest, dist

    # Calc AB_cycle distance using distance matrix (memory)
    def ab_cycle_dist(self, ab_cycle):
        # Convert deque to list
        aux = list(ab_cycle)
        # Distance
        dist = 0
        # Distance lookup
        for i, j in zip(aux[0::2], aux[1::2]):
            # Ignore ghost nodes
            t = sorted([abs(i)-1, abs(j)-1])
            dist += self._dm[self._cindex(*t)]
        # Return result
        return dist

    # Calc tour distance using distance matrix (memory)
    def tour_dist(self, tour):
        # assert len(set(tour)) == len(tour), "Invalid TSP tour"
        # distance
        dist = 0
        # Distance matrix lookup
        for i, j in zip(tour[:-1], tour[1:]):
            t = sorted([i-1, j-1])
            dist += self._dm[self._cindex(*t)]

        # Close path
        t = sorted([tour[0]-1, tour[-1]-1])
        dist += self._dm[self._cindex(*t)]

        return dist

    # Calc route distance
    def route_dist(self, route):
        # distance
        dist = 0
        # Distance matrix lookup
        for i, j in zip(route[:-1], route[1:]):
            t = sorted([i-1, j-1])
            dist += self._dm[self._cindex(*t)]

        return dist

###############################################################################
# Use these methods if distance matrix is too large to store into memory

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
        return float(Shell.run(cmd))

    # Calc tour distance using file
    def tour_dist_2(self, tour):
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
        return float(Shell.run(cmd))
