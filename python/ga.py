#!/usr/bin/python
# ozeasx@gmail.com

import time
import random
import pp
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
        # Generations
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
        self._execution_time = defaultdict(list)

        # List to store current population
        self._population = list()
        # Best and worst solutions found
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
    # N = Number of individuals
    def gen_pop(self, size, method='random'):
        # Need even population
        assert not (size % 2), "Invalid population size. " \
                               "Must be even and greater than 0"
        # Print step
        print "Generating initial population..."
        # Regiter start time
        start_time = time.time()
        # Random generation
        if method == 'random':
            while len(self._population) < size:
                c = Chromosome(self._data.dimension)
                if c not in self._population:
                    c.dist = self._data.tour_dist(c.tour)
                    self._population.append(c)
                    print '\r', len(self._population),
        # two_opt
        if method == 'two_opt':
            while len(self._population) < size:
                c = Chromosome(self._data.dimension)
                c = mut.two_opt(c, self._data)
                if c not in self._population:
                    self._population.append(c)
                    print '\r', len(self._population),
        # Done
        print "Done..."
        # Store execution time
        self._execution_time['gen_pop'].append(time.time() - start_time)
        # Global start time
        self._start_time = time.time()

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
        self._execution_time['evaluate'].append(time.time() - start_time)

    # Tournament selection
    def select_tournament(self, k):
        # Register start time
        start_time = time.time()
        # Tournament winners
        selected = list()

        # Elitism
        if self._elite:
            for _ in xrange(self._elite):
                selected.append(max(self._population,
                                    key=attrgetter('fitness')))

        # Tournament
        for i in xrange(self._pop_size - self._elite):
            # Retrieve k-sized sample
            tournament = random.sample(self._population, k)
            # Get best solution
            selected.append(max(tournament, key=attrgetter('fitness')))

        # Update population
        self._population = selected
        # Regiter execution time
        self._execution_time['select_tournament'].append(time.time()
                                                         - start_time)

        # assert len(self._population) == self._pop_size

    def recombine(self, p_cross):
        # Register start time
        start_time = time.time()

        # Recombination
        for i in xrange(0, self._pop_size, 2):
            if random.random() < p_cross:
                c1, c2 = self._gpx.recombine(self._population[i],
                                             self._population[i+1])
                # Replace p1 and p2 only if c1 or c2 are different from parents
                if (c1 not in [self._population[i], self._population[i+1]]
                    or c2 not in [self._population[i], self._population[i+1]]):
                    self._population[i], self._population[i+1] = c1, c2
                    self._cross += 1

        # Register execution time
        self._execution_time['recombine'].append(time.time() - start_time)

        # assert len(self._population) == self._pop_size

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
        self._execution_time['mutate'].append(time.time() - start_time)

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
        self._execution_time['restart_pop'].append(time.time() - start_time)

    # Generation info
    def print_info(self):
        cross = self._cross - self._last_cross
        mut = self._mut - self._last_mut
        print "T: %i\tCross: %i\tMut: %i\tAverage: %f\tBest: %f\tRestart: %s" \
              % (self._generation, cross, mut, self._avg_fitness,
                 self._best_solution.fitness, self._pop_restart)
        self._last_cross = self._cross
        self._last_mut = self._mut
        # set pop restart flag
        self._pop_restart = False

    # Final report
    def report(self):
        print "------------------------ Statitics ----------------------------"
        print "Total Crossover:", self._cross
        print "Total mutations:", self._mut
        print "---------------------- Time statistics-------------------------"
        print "Execution time:", time.time() - self._start_time
        print "Inicial population:", sum(self._execution_time['gen_pop'])
        print "Evaluation:", sum(self._execution_time['evaluate'])
        print "Selection:", sum(self._execution_time['select_tournament'])
        print "Recombination:", sum(self._execution_time['recombine'])
        print "Mutation:", sum(self._execution_time['mutate'])
        print "Population restart:", sum(self._execution_time['restart_pop'])
        if self._data.best_tour:
            print "----------------- Best known solution ---------------------"
            print "Tour:", self._data.best_tour
            print "Distance:", self._data.tour_dist(self._data.best_tour)
        print "-------------------- Best individual found --------------------"
        print "Tour:", self._best_solution.tour
        print "Distance:", self._best_solution.dist
        print "---------------------------------------------------------------"

    # Calculate the individual fitness
    def _evaluate(self, c):
        return -c.dist

    # Calculate avg distance from individual to population
    def _max_dist(self, c):
        aux = 0
        for p in self._population:
            aux = max(aux, c-p)
        return aux
