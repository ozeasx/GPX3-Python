#!/usr/bin/python
# ozeasx@gmail.com

import time
import random
import logging as log
from collections import defaultdict
from operator import attrgetter
from chromosome import Chromosome
from gpx import GPX
import mut


# Class to abstract a Genetic algorithm
class GA(object):
    # GA initialization
    def __init__(self, data, elite=0):
        # Parametrization
        self._data = data
        self._gpx = GPX(data)
        self._elite = elite

        # Statitics
        # Population size
        self._pop_size = 0
        # Generation count
        self._generation = 0
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
        # Timers
        self._exec_time = defaultdict(list)

        # Current population
        self._population = list()
        # Best solution found
        self._best_solution = None

    # Get current generation number
    @property
    def generation(self):
        return self._generation

    # return best individuals
    @property
    def best_solution(self):
        return self._best_solution

    # Generate inicial population
    def gen_pop(self, size, method='random'):
        # Need even population
        assert not (size % 2), "Invalid population size. " \
                               "Must be even and greater than 0"
        # Print step
        print "Generating initial population..."
        # Regiter local and global start time
        self._start_time = start_time = time.time()
        # Population set to ensure unicity
        self._population = set()
        # Random generation
        if method == 'random':
            while len(self._population) < size:
                self._population.add(Chromosome(self._data.dimension))
            for c in self._population:
                c.dist = self._data.tour_dist(c.tour)
        # two_opt
        if method == 'two_opt':
            while len(self._population) < size:
                c = Chromosome(self._data.dimension)
                c.dist = self._data.tour_dist(c.tour)
                c = mut.two_opt(c, self._data)
                self._population.add(c)
        # Go back to list
        self._population = set(self._population)
        # Done
        print "Done..."
        # Store execution time
        self._exec_time['population'].append(time.time() - start_time)

    # Evaluate the entire population
    def evaluate(self):
        # Register star time
        start_time = time.time()
        # Set population size and increment generation
        self._pop_size = len(self._population)
        self._generation += 1

        # Update fitness of all population
        total_fitness = 0
        for c in self._population:
            c.fitness = self._evaluate(c)
            total_fitness += c.fitness

        # Calc average fitness
        self._avg_fitness = total_fitness/float(self._pop_size)

        # Store best solution found
        self._best_solution = max(self._population, key=attrgetter('fitness'))

        # Register execution Timers
        self._exec_time['evaluation'].append(time.time() - start_time)

    # Tournament selection
    def select_tournament(self, k):
        # Register start time
        start_time = time.time()
        # Tournament winners
        selected = list()
        # Elitism
        if self._elite:
            # Sort to get n better individuals
            selected.extend(sorted(set(self._population),
                                   key=attrgetter('fitness'))[:self._elite])

        # Tournament
        for i in xrange(self._pop_size - self._elite):
            # Retrieve k-sized sample
            tournament = random.sample(self._population, k)
            # Get best solution
            selected.append(max(tournament, key=attrgetter('fitness')))

        # Update population
        self._population = selected
        # Regiter execution time
        self._exec_time['tournament'].append(time.time() - start_time)

        # assert len(self._population) == self._pop_size

    def recombine(self, operator, p_cross, test=False):
        # Register start time
        start_time = time.time()
        # Auxiliar list
        aux = list()

        # Try to form couples?
        if test:
            couple = True
            while couple:
                couple = False
                for i in xrange(len(self._population)):
                    for j in xrange(i + 1, len(self._population)):
                        if self._population[i] != self._population[j]:
                            i, j = sorted([i, j])
                            aux.append(self._population.pop(j))
                            aux.append(self._population.pop(i))
                            couple = True
                            break
                    if couple:
                        break
            self._population += aux

        # Recombination
        for i in xrange(0, self._pop_size, 2):
            if random.random() < p_cross:
                c1, c2 = self._gpx.recombine(self._population[i],
                                             self._population[i+1])
                # Replace p1 and p2 only if c1 or c2 are different from parents
                if c1 not in [self._population[i], self._population[i+1]] \
                   or c2 not in [self._population[i], self._population[i+1]]:
                    self._population[i], self._population[i+1] = c1, c2
                    self._cross += 1

        # Register execution time
        self._exec_time['recombination'].append(time.time() - start_time)

        assert len(self._population) == self._pop_size

    # Mutate individuals according to p_mut probability
    def mutate(self, p_mut):
        # Register start time
        start_time = time.time()
        # Is map fast?
        for i in xrange(self._pop_size):
            if random.random() < p_mut:
                self._population[i] = mut.two_opt(self._population[i],
                                                  self._data)
                self._mut += 1

        # Register execution time
        self._exec_time['mutation'].append(time.time() - start_time)

        # assert len(self._population) == self._pop_size

    # Reset population
    def restart_pop(self, ratio):
        # Register start time
        start_time = time.time()

        if not (self._cross - self._last_cross):
            # Population restarted flag
            self._pop_restart = True
            self._population.sort(key=attrgetter('fitness'))
            for i in xrange(int(self._pop_size * ratio)):
                c = Chromosome(self._data.dimension)
                while c in self._population:
                    c = Chromosome(self._data.dimension)
                c.dist = self._data.tour_dist(c.tour)
                self._population[i] = c

        # Register execution time
        self._exec_time['pop restart'].append(time.time() - start_time)

        # assert len(self._population) == self._pop_size

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
        log.info("----------------------- Statitics -------------------------")
        log.info("Total Crossover: %i", self._cross)
        log.info("Failed crossovers: %i", self._gpx.failed)
        log.info("Total mutations: %i", self._mut)
        log.info("--------------------- Time statistics----------------------")
        log.info("Total execution time: %f", time.time() - self._start_time)
        log.info("Inicial population: %f", sum(self._exec_time['population']))
        log.info("Evaluation: %f", sum(self._exec_time['evaluation']))
        log.info("Selection: %f", sum(self._exec_time['tournament']))
        log.info("Recombination: %f", sum(self._exec_time['recombination']))
        log.info("  Partitioning: %f", sum(self._gpx.exec_time['partition']))
        log.info("  Simplified graph: %f",
                 sum(self._gpx.exec_time['simple graph']))
        log.info("  Classification: %f", sum(self._gpx.exec_time['classify']))
        log.info("  Fusion: %f", sum(self._gpx.exec_time['fusion']))
        log.info("  Build: %f", sum(self._gpx.exec_time['build']))
        log.info("Mutation: %f", sum(self._exec_time['mutation']))
        log.info("Population restart: %f", sum(self._exec_time['pop restart']))
        if self._data.best_tour:
            log.info("---------------- Best known solution ------------------")
            log.info("Tour: %s", (self._data.best_tour,))
            log.info("Distance: %f",
                     self._data.tour_dist(self._data.best_tour))
        log.info("------------------- Best individual found -----------------")
        log.info("Tour: %s", (self._best_solution.tour,))
        log.info("Distance: %f", self._best_solution.dist)
        log.info("-----------------------------------------------------------")

    # Calculate the individual fitness
    def _evaluate(self, c):
        return -c.dist
