#!/usr/bin/python
# ozeasx@gmail.com

import time
import random
import logging as log
from collections import defaultdict
from operator import attrgetter
from itertools import combinations
import numpy
from vrp_chromosome import VRP_Chromosome as Chromosome
import mut


# Class to abstract a Genetic algorithm
class GA(object):
    # GA initialization
    def __init__(self, data, cross_op, fit_func, elitism=0):
        # Parametrization
        self._data = data
        self._cross_op = cross_op
        self._elitism = elitism

        # Generation count
        self._generation = -1
        # Average fitness of the current generation
        self._avg_fitness = 0
        # Numbers of crossover
        self._cross = 0
        self._last_cross = 0
        # Numbers os mutation
        self._mut = 0
        self._last_mut = 0
        # Population was restarted
        self._pop_restart = False
        # Fitness function
        self._fit_func = fit_func
        # Timers
        self._timers = defaultdict(list)

        # Current and next population
        self._population = None
        # Best solution found
        self._best_solution = None
        # Elite population
        self._elite = list()

    # Get current generation number
    @property
    def generation(self):
        return self._generation

    # Average fitness of current generation
    @property
    def avg_fitness(self):
        return self._avg_fitness

    # Return total crossovers
    @property
    def cross(self):
        return self._cross

    # Return total mutations
    @property
    def mut(self):
        return self._mut

    # return best individuals
    @property
    def best_solution(self):
        return self._best_solution

    # Return timers
    @property
    def timers(self):
        return self._timers

    # Generate inicial population
    def gen_pop(self, size, method='random'):
        # Regiter local and global start time
        self._start_time = start_time = time.time()
        # Need even population
        assert not (size % 2), "Invalid population size. " \
                               "Must be even and greater than 0"
        # Print step
        print "Generating initial population..."
        # Population set to ensure unicity
        self._population = set()
        # Random generation
        for i in xrange(size):
            c = Chromosome(self._data.dimension, self._data.trucks)
            while c in self._population:
                c = Chromosome(self._data.dimension, self._data.trucks)
            c.dist = self._data.tour_dist(c.tour)
            # 2opt generation
            if method == '2opt':
                c = mut.vrp_2opt(c, self._data)
            c.load = self._data.routes_load(c.routes)
            self._population.add(c)
        # Convert population to list
        self._population = list(self._population)
        # Done
        print "Done..."
        # Store execution time
        self._timers['population'].append(time.time() - start_time)

    # Evaluate the entire population
    def evaluate(self):
        # Register star time
        start_time = time.time()

        # Update fitness of all population
        total_fitness = 0
        for c in self._population:
            c.fitness = self._evaluate(c)
            total_fitness += c.fitness

        # Calc average fitness
        self._pop_size = len(self._population)
        self._avg_fitness = total_fitness/float(self._pop_size)

        # Elitism
        if self._elitism:
            # Insert previous elite population
            self._population += self._elite
            # Sort
            self._population.sort(key=attrgetter('fitness'), reverse=True)
            # Save new elite
            self._elite = self._population[:self._elitism]
            # Adjust population size
            self._population = self._population[:self._pop_size]

        # Store best solution found
        if not self._best_solution:
            self._best_solution = max(self._population,
                                      key=attrgetter('fitness'))
        else:
            current_best = max(self._population, key=attrgetter('fitness'))
            if current_best.fitness > self._best_solution.fitness:
                self._best_solution = current_best

        # Increment generaion
        self._generation += 1

        # Register execution Timers
        self._timers['evaluation'].append(time.time() - start_time)

    # Tournament selection
    def select_tournament(self, k):
        # Register start time
        start_time = time.time()
        # Tournament winners
        selected = list()

        # Tournament
        for i in xrange(self._pop_size):
            # Retrieve k-sized sample
            tournament = random.sample(self._population, k)
            # Get best solution
            selected.append(max(tournament, key=attrgetter('fitness')))

        # Update population
        self._population = selected
        # Regiter execution time
        self._timers['tournament'].append(time.time() - start_time)

        # Assure population size remains the same
        assert len(self._population) == self._pop_size, "Tournament, pop size"

    # Recombination
    def recombine(self, p_cross, pairwise=None):
        # Register start time
        start_time = time.time()

        # Shuffle population
        # random.shuffle(self._population)

        # Pairwise recombination
        if pairwise == 'True':
            selected = list()
            for pair in combinations(set(self._population), 2):
                selected.extend([pair[0], pair[1]])
            self._population = selected

        # New generation
        children = list()

        # Recombination
        for p1, p2 in zip(self._population[0::2], self._population[1::2]):
            # print p1.dist
            if random.random() < p_cross:
                c1, c2 = self._cross_op.recombine(p1.to_tsp(), p2.to_tsp())
                c1 = c1.to_vrp(self._data.dimension)
                c2 = c2.to_vrp(self._data.dimension)
                c1.load = self._data.routes_load(c1.routes)
                c2.load = self._data.routes_load(c2.routes)
                children.extend([c1, c2])
                # Count cross only if there is at least one different child
                if c1 not in [p1, p2] or c2 not in [p1, p2]:
                    self._cross += 1

        # Reduce population in case of pairwise recombination
        if pairwise == 'True':
            # Reevaluate population
            for c in children:
                c.fitness = self._evaluate(c)
            children.sort(key=attrgetter('fitness'))
            self._population = children[:self._pop_size]
        else:
            self._population = children

        # Register execution time
        self._timers['recombination'].append(time.time() - start_time)

        # Assure population size remains the same
        assert len(self._population) == self._pop_size, "Cross, pop size"

    # Mutate individuals according to p_mut probability
    def mutate(self, p_mut):
        # Register start time
        start_time = time.time()
        # Is map fast?
        for i in xrange(self._pop_size):
            if random.random() < p_mut:
                self._population[i] = mut.vrp_2opt(self._population[i],
                                                   self._data)
                self._population[i].load = self._data.routes_load(
                                                    self._population[i].routes)
                self._mut += 1

        # Register execution time
        self._timers['mutation'].append(time.time() - start_time)

    # Reset population
    def restart_pop(self, ratio, pairwise=False, method='random'):
        # Register start time
        start_time = time.time()

        if not (self._cross - self._last_cross) or pairwise == 'True':
            # Reevaluate population
            for c in self._population:
                c.fitness = self._evaluate(c)
            # Population restarted flag
            self._pop_restart = True
            self._population.sort(key=attrgetter('fitness'))
            for i in xrange(int(self._pop_size * ratio)):
                c = Chromosome(self._data.dimension, self._data.trucks)
                while c in self._population:
                    c = Chromosome(self._data.dimension, self._data.trucks)
                c.dist = self._data.tour_dist(c.tour)
                if method == '2opt':
                    c = mut.vrp_2opt(c, self._data)
                c.load = self._data.routes_load(c.routes)
                self._population[i] = c

        # Register execution time
        self._timers['pop_restart'].append(time.time() - start_time)

        # Assure population size remains the same
        assert len(self._population) == self._pop_size, "restart, pop size"

    # Generation info
    def print_info(self):
        cross = self._cross - self._last_cross
        mut = self._mut - self._last_mut
        log.info("T: %i\tC: %i\tM: %i\tAvg: %f\tBest: %f\tRestart: %s",
                 self._generation, cross, mut, self._avg_fitness,
                 self._best_solution.fitness, self._pop_restart)
        self._last_cross = self._cross
        self._last_mut = self._mut
        # set pop restart flag
        self._pop_restart = False

    # Final report
    def report(self):
        self._timers['total'].append(time.time() - self._start_time)
        log.info("----------------------- Statitics -------------------------")
        log.info("Total Crossover: %i", self._cross)
        log.info("Failed: %i", self._cross_op.counters['failed'])
        parents_sum = self._cross_op.counters['parents_sum']
        children_sum = self._cross_op.counters['children_sum']
        if parents_sum != 0:
            log.info("Overall improvement: %f", (parents_sum - children_sum)
                     / float(parents_sum) * 100)
        log.info("Partitions")
        log.info(" Feasible type 1: %i", self._cross_op.counters['feasible_1'])
        log.info(" Feasible type 2: %i", self._cross_op.counters['feasible_2'])
        log.info(" Feasible type 3: %i", self._cross_op.counters['feasible_3'])
        log.info(" Infeasible: %i", self._cross_op.counters['infeasible'])
        log.info(" Fusions: %i", self._cross_op.counters['fusions'])
        log.info(" Unsolved: %i", self._cross_op.counters['unsolved'])
        log.info("Infeasible tours: %i", self._cross_op.counters['inf_tours'])
        log.info("Total mutations: %i", self._mut)
        log.info("--------------------- Time statistics----------------------")
        log.info("Total execution time: %f", sum(self._timers['total']))
        log.info("Inicial population: %f", sum(self._timers['population']))
        log.info("Evaluation: %f", sum(self._timers['evaluation']))
        log.info("Selection: %f", sum(self._timers['tournament']))
        log.info("Recombination: %f", sum(self._timers['recombination']))
        log.info(" Partitioning: %f",
                 sum(self._cross_op.timers['partitioning']))
        log.info(" Simplified graph: %f",
                 sum(self._cross_op.timers['simple_graph']))
        log.info(" Classification: %f",
                 sum(self._cross_op.timers['classification']))
        log.info(" Fusion: %f", sum(self._cross_op.timers['fusion']))
        log.info(" Build: %f", sum(self._cross_op.timers['build']))
        log.info("Mutation: %f", sum(self._timers['mutation']))
        log.info("Population restart: %f", sum(self._timers['pop_restart']))
        log.info("Capcity: %f", self._data.capacity)
        if self._data.best_solution:
            log.info("---------------- Best known solution ------------------")
            log.info("Tour: %s", (self._data.best_solution.tour,))
            log.info("Distance: %f", self._data.best_solution.dist)
            log.info("Load: %s", (self._data.best_solution.load))
        log.info("------------------- Best individual found -----------------")
        log.info("Tour: %s", (self._best_solution.tour,))
        log.info("Distance: %f", self._best_solution.dist)
        log.info("Load: %s", (self._best_solution.load))
        log.info("-----------------------------------------------------------")

    # Calculate the individual fitness
    def _evaluate(self, c):
        # Eliminate infeasible solutions
        if self._fit_func == 'a':
            if any(load > self._data.capacity for load in c.load):
                return -float("inf")
            else:
                return -c.dist
        # Standard deviation
        elif self._fit_func == 'b':
            return -c.dist * numpy.std(c.load)
        # Standard deviation squared
        elif self._fit_func == 'c':
            return -c.dist * (numpy.std(c.load) ** 2)
        # Standard deviation if infeasible
        elif self._fit_func == 'd':
            if any(load > self._data.capacity for load in c.load):
                return -c.dist * numpy.std(c.load)
            else:
                return -c.dist
        # Square of standard deviation if infeasible
        elif self._fit_func == 'e':
            if any(load > self._data.capacity for load in c.load):
                return -c.dist * (numpy.std(c.load) ** 2)
            else:
                return -c.dist
