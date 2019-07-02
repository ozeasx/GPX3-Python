#!/usr/bin/python
# ozeasx@gmail.com

from collections import defaultdict
from tsp import TSPLIB
from shell import Shell


class VRPLIB(TSPLIB):
    def __init__(self, instance_path):
        # Call super init
        super(VRPLIB, self).__init__(instance_path)

        # Instance type
        self._type = 'vrp'

        # Set capacity
        self._capacity = float(Shell.run("grep CAPACITY " + instance_path
                                         + " | cut -d':' -f2").strip())
        # Mim number of vehicles
        self._trucks = int(Shell.run("grep TRUCKS " + instance_path
                                     + " | cut -d':' -f2").strip())

        # Load demand array
        with open(instance_path) as instance:
            read = False
            self._demand = list()
            for word in instance:
                # End loop if EOF
                if word.strip() == "DEPOT_SECTION":
                    break
                # If do, store tour
                if read:
                    s = word.split()
                    self._demand.append(int(s[1]))
                # If DEMAND_SECTION, set 'read'
                if word.strip() == "DEMAND_SECTION":
                    read = True

        if self._best_solution is not None:
            self._best_solution.load = self.routes_load(
                                                self._best_solution.routes)

    # Get vehicles capacity
    @property
    def capacity(self):
        return self._capacity

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
        elif (solution.dist < self._best_solution.dist
              and all(load <= self._capacity for load in solution.load)):
            self._best_solution = solution
        else:
            return

        # Write new solution to file
        with open(self._instance_name + ".opt.tour.new", 'w') as best:
            best.write("NAME : " + self._name + ".opt.tour.new\n")
            best.write("COMMENT : ozeasx@gmail.com\n")
            best.write("TYPE : CVRP TOUR\n")
            best.write("DIMENSION : " + str(self._dimension) + "\n")
            best.write("LENGTH : " + str(solution.dist) + "\n")
            best.write("LOAD : " + str(solution.load) + "\n")
            best.write("TOUR_SECTION\n")
            for node in solution.tour:
                best.write(str(node) + "\n")
            best.write("-1\n")
            best.write("EOF\n")

    # Get client demand
    def demand(self, client):
        return self._demand[client - 1]

    # Get tour demand
    def routes_load(self, routes):
        load = defaultdict(int)
        for key in routes:
            for client in routes[key]:
                load[key] += self._demand[client - 1]
        return load

    # Calc AB_cycle distance using distance matrix (memory)
    def ab_dist(self, ab_cycle):
        # Convert deque to list
        aux = list(ab_cycle)
        # Distance
        dist = 0
        # Distance lookup
        for i, j in zip(aux[0::2], aux[1::2]):
            # Ignore ghost nodes
            t = [abs(i)-1, abs(j)-1]
            # Convert depots
            if t[0] >= self._dimension:
                t[0] = 0
            if t[1] >= self._dimension:
                t[1] = 0
            dist += self._dm[self._cindex(*sorted(t))]
        # Return result
        return dist


# Test section
if __name__ == '__main__':
    vrp = VRPLIB("../cvrp/F-n45-k4.vrp")
    # print vrp._capacity
    # print vrp._trucks
    print vrp.ab_cycle_dist([1, 2, 3, 1])
    print vrp.ab_cycle_dist([1, 2, 3, -47])
    print vrp.ab_cycle_dist([1, 2, 3, 47])
