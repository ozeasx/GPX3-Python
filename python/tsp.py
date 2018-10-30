#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
import os
import logging as log


class TSPLIB(object):
    def __init__(self, instance_path, shell):
        # Set instance file and shell object
        self._instance_path = instance_path
        self._instance_name = instance_path[:-4]
        self._shell = shell
        self._best_tour = None

        # Set tsp dimension
        self._dimension = int(shell.run("grep DIMENSION " + instance_path
                                        + " | cut -d':' -f2").strip())

        # Condensed index mapping
        # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist/13079806
        self._cindex = lambda i, j: i*(2*self._dimension - i - 3)/2 + j - 1

        # Set best known solution, if exists
        if os.path.isfile(self._instance_name + ".opt.tour"):
            with open(self._instance_name + ".opt.tour") as best:
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
        print "Generating distance matrix..."
        if not os.path.isfile(self._instance_name + ".tsp.dm"):
            shell.run("../tsplib/create_dm.r " + instance_path)
        print "Done..."

        # Generate list of lists combination lookup
        # self._hash = np.empty((self._dimension, self._dimension),dtype=float)
        # self._dm = np.empty((self._dimension, self._dimension), dtype=float)
        self._hash = dict()
        self._dm = list()
        line_number = 1
        with open(self._instance_name + ".tsp.dm") as dm:
            for t, dist in zip(combinations(xrange(self._dimension), 2), dm):
                self._hash[t] = line_number
                self._dm.append(float(dist))
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
        dist = 0
        # Distance lookup
        for i, j in zip(aux[0::2], aux[1::2]):
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
        return float(self._shell.run(cmd))

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
        return float(self._shell.run(cmd))
