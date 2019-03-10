#!/usr/bin/python
# ozeasx@gmail.com

import math
import time
import random
import logging as log
from collections import defaultdict
from operator import attrgetter
from itertools import combinations
from chromosome import Chromosome
import functions


# Class to abstract a Genetic algorithm
class GA(object):
    # GA initialization
    def __init__(self, data, cross_op, elitism=0):
        # Instance data
        self._data = data
        # Crossover operator
        self._cross_op = cross_op

        # Number of elite population
        self._elitism = elitism
        # Generation count
        self._generation = -1
        # Average fitness of the current generation
        self._avg_fitness = 0
        # To indicate if pop should be restarted
        self._restart_pop = False
        # To indicate if population was restarted
        self._pop_restarted = False
        # Counters amd timers
        self._counters = defaultdict(list)
        self._timers = defaultdict(list)

        # Current population
        self._population = list()
        # Elite population
        self._elite = list()
        # Best solution found
        self._best_solution = None

        # Initialize counters
        self._counters['cross'].append(0)
        self._counters['mut'].append(0)

    # Get current generation number
    @property
    def generation(self):
        return self._generation

    # Average fitness of current generation
    @property
    def avg_fitness(self):
        return self._avg_fitness

    # return best individuals
    @property
    def best_solution(self):
        return self._best_solution

    # Return counters
    @property
    def counters(self):
        return self._counters

    # Return timers
    @property
    def timers(self):
        return self._timers

    # =========================================================================
    # Insert unique solutions into population
    def _insert_pop(self, number, method='random', eval=False):
        # Do nothing
        if number == 0:
            return
        # Individuals to be inserted
        elif 0 < number < 1:
            number = 1

        # ---------------------------------------------------------------------
        # Sub functions to generate unique chromosome
        def random():
            c = Chromosome(self._data.dimension)
            # Avoid duplicates
            while c in self._population:
                c = Chromosome(self._data.dimension)
            c.dist = self._data.tour_dist(c.tour)
            return c

        def two_opt():
            c = random()
            c = functions.two_opt(c, self._data)
            while c in self._population:
                c = random()
                c = functions.two_opt(c, self._data)
            return c

        def nn():
            c = functions.nn(self._data, method)
            # Avoid duplicates
            while c in self._population or c is None:
                c = functions.nn(self._data, method)
        # ---------------------------------------------------------------------

        for i in xrange(int(number)):
            # Random
            if method == 'random':
                c = random()
            # 2opt
            elif method == '2opt':
                c = two_opt()
            # NN and NN with 2opt
            elif method in ['nn', 'nn2opt']:
                c = nn()
            # Insert c in population
            assert c.dist is not None, "_insert_pop, 'dist is none'"
            # Avaliate before insertion
            if eval:
                c.fitness = self._evaluate(c)
            # Insert unique individual into population
            self._population.append(c)

    # =========================================================================
    # Generate inicial population
    def gen_pop(self, size, method='random', ratio=1):
        # Regiter local and global start time
        self._start_time = start_time = time.time()
        # Need even population
        assert not (size % 2), "Invalid population size. " \
                               "Must be even and greater than 0"
        # Print step
        print "Generating initial population..."
        # Population generation
        if method == 'random':
            self._insert_pop(size, method)
        else:
            self._insert_pop(size - ratio * size, 'random')
            self._insert_pop(ratio * size, method)
        # Done
        print "Done..."
        # Assert population size
        self._pop_size = len(self._population)
        assert self._pop_size == size, "gen_pop, pop_size"
        # Store execution time
        self._timers['population'].append(time.time() - start_time)

    # =========================================================================
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
        self._counters['avg_fit'].append(total_fitness/float(self._pop_size))

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

        self._counters['best_fit'].append(self._best_solution.fitness)

        # Increment generaion
        self._generation += 1

        # Register execution Timers
        self._timers['evaluation'].append(time.time() - start_time)

    # =========================================================================
    # Calculate the individual fitness
    def _evaluate(self, c):
        return -c.dist

    # =========================================================================
    # Tournament selection
    def tournament_selection(self, k):
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

    # =========================================================================
    # Pairwise selection
    def pairwise_selection(self):
        selected = list()
        for p1, p2 in combinations(set(self._population), 2):
            selected.extend([p1, p2])
        self._population = selected

    # =========================================================================
    # Ranking selection
    # https://dl.acm.org/citation.cfm?id=93169
    def rank_selection(self, b=1.5):
        def select_parent():
            index = int(self._pop_size * (b - math.sqrt(b ** 2 - 4*(b - 1)
                                                        * random.random()))
                        / 2 / (b - 1))
            return self._population[index]

        selected = list()
        self._population.sort(key=attrgetter('fitness'))
        for i in xrange(self._pop_size):
            selected.append(select_parent())

        self._population = selected

    # =========================================================================
    # Recombination
    def recombine(self, p_cross, pairwise=False):
        # Register start time
        start_time = time.time()

        # New generation
        children = list()
        # Counters
        cross = 0
        # Recombination
        for p1, p2 in zip(self._population[0::2], self._population[1::2]):
            # print p1.dist
            if random.random() < p_cross:
                c1, c2 = self._cross_op.recombine(p1, p2)
                children.extend([c1, c2])
                # Count cross only if there is at least one different child
                if c1 not in [p1, p2] or c2 not in [p1, p2]:
                    cross += 1

        # Reduce population in case of pairwise recombination
        if pairwise:
            # Remove duplicates
            children = set(children)
            children = list(children)
            # Reevaluate population
            for c in children:
                c.fitness = self._evaluate(c)
            children.sort(key=attrgetter('fitness'), reverse=True)
            self._population = children[:self._pop_size]
            if len(self._population) < self._pop_size:
                self._insert_pop(self._pop_size - len(self._population),
                                 'random', eval=True)
        else:
            self._population = children

        # Set restart based on crossover number or best_fit or avg_fit
        if self._generation > 0:
            if ((cross == 0) or (self._counters['avg_fit'][-2]
                                 == self._counters['avg_fit'][-1])
                or (self._counters['best_fit'][-2]
                    == self._counters['best_fit'][-1])):
                self._restart_pop = True

        # Assure population size remains the same
        assert len(self._population) == self._pop_size, "ga, recombination"

        # Update counters
        self._counters['cross'].append(cross)

        # Register execution time
        self._timers['recombination'].append(time.time() - start_time)

    # =========================================================================
    # Mutate individuals according to p_mut probability
    def mutate(self, p_mut, method):
        # Register start time
        start_time = time.time()
        # Mutations counter
        mut = 0
        # Is map fast?
        for i, c in enumerate(self._population):
            if random.random() < p_mut:
                if method == '2opt':
                    c = functions.two_opt(self._population[i], self._data)
                elif method == 'nn' or method == 'nn2opt':
                    c = functions.nn(self._data, method)
                    # Avoid duplicates
                    while c in self._population or c is None:
                        c = functions.nn(self._data, method)
                if c != self._population[i]:
                    self._population[i] = c
                    mut += 1

        # Update counter
        self._counters['mut'].append(mut)

        # Register execution time
        self._timers['mutation'].append(time.time() - start_time)

    # =========================================================================
    # Reset population
    def restart_pop(self, ratio, method='random'):
        # Register start time
        start_time = time.time()

        if self._restart_pop:
            # Population restarted flag
            self._restart_pop = False
            # Report population restart
            self._pop_restarted = True
            # Remove duplicates
            self._population = set(self._population)
            self._population = list(self._population)
            # Complete population
            self._insert_pop(self._pop_size - len(self._population), method,
                             eval=True)
            # Sort population
            self._population.sort(key=attrgetter('fitness'), reverse=True)
            # Reduce pop to acomodate restart
            self._population = self._population[:int(self._pop_size
                                                     - self._pop_size * ratio)]
            # Insert new population
            self._insert_pop(self._pop_size * ratio, method, eval=True)

        # Register execution time
        self._timers['restart_pop'].append(time.time() - start_time)

        # Assure population size remains the same
        assert len(self._population) == self._pop_size, "ga, restart_pop"

    # =========================================================================
    # Generation info
    def print_info(self):

        log.info("T: %i\tC: %i\tM: %i\tAvg: %f\tB: %f\tR: %s",
                 self._generation, self._counters['cross'][-1],
                 self._counters['mut'][-1], self._counters['avg_fit'][-1],
                 self._counters['best_fit'][-1], self._pop_restarted)

        # Reset restarted indicator
        self._pop_restarted = False

    # =========================================================================
    # Final report
    def report(self):
        self._timers['total'].append(time.time() - self._start_time)
        log.info("----------------------- Statitics -------------------------")
        log.info("Total Crossover: %i", sum(self._counters['cross']))
        log.info("Total fails: %i", self._cross_op.counters['failed'])
        log.info(" by duplicated parents: %i",
                 self._cross_op.counters['failed_0'])
        log.info(" by 1 or 0 partition after partitioning: %i",
                 self._cross_op.counters['failed_1'])
        log.info(" by 1 or 0 partition after fusion: %i",
                 self._cross_op.counters['failed_2'])
        log.info(" by no constructed tours: %i",
                 self._cross_op.counters['failed_3'])

        parents_sum = self._cross_op.counters['parents_sum']
        children_sum = self._cross_op.counters['children_sum']

        if parents_sum != 0:
            log.info("Overall improvement: %f", (parents_sum - children_sum)
                     / float(parents_sum) * 100)
        log.info("Feasible partitions: %i",
                 self._cross_op.counters['feasible'])
        log.info(" Feasible test 1: %i", self._cross_op.counters['feasible_1'])
        log.info(" Feasible test 2: %i", self._cross_op.counters['feasible_2'])
        log.info(" Feasible test 3: %i", self._cross_op.counters['feasible_3'])
        log.info("Infeasible: %i", self._cross_op.counters['infeasible'])
        log.info("Fusions: %i", self._cross_op.counters['fusion'])
        log.info(" Fusions test 1: %i", self._cross_op.counters['fusion_1'])
        log.info(" Fusions test 2: %i", self._cross_op.counters['fusion_2'])
        log.info(" Fusions test 3: %i", self._cross_op.counters['fusion_3'])
        log.info("Unsolved: %i", self._cross_op.counters['unsolved'])
        log.info("Infeasible tours: %i",
                 self._cross_op.counters['inf_tour'])
        log.info(" Infeasible tours 1: %i",
                 self._cross_op.counters['inf_tour_0'])
        log.info(" Infeasible tours 2: %i",
                 self._cross_op.counters['inf_tour_1'])
        log.info(" Infeasible tours 3: %i",
                 self._cross_op.counters['inf_tour_2'])
        log.info(" Infeasible tours 4: %i",
                 self._cross_op.counters['inf_tour_3'])
        log.info("Total mutations: %i", sum(self._counters['mut']))
        log.info("--------------------- Time statistics----------------------")
        log.info("Total execution time: %f", sum(self._timers['total']))
        log.info("Inicial population: %f", sum(self._timers['population']))
        log.info("Evaluation: %f", sum(self._timers['evaluation']))
        log.info("Selection: %f", sum(self._timers['tournament']))
        log.info("Recombination: %f", sum(self._timers['recombination']))
        log.info(" G Star: %f",
                 sum(self._cross_op.timers['g_star']))
        log.info(" Partitioning: %f",
                 sum(self._cross_op.timers['partitioning']))
        log.info(" Simplified graph: %f",
                 sum(self._cross_op.timers['simple_graph']))
        log.info(" Simplified graph (fusion): %f",
                 sum(self._cross_op.timers['simple_graph_f']))
        log.info(" Classification: %f",
                 sum(self._cross_op.timers['classification']))
        log.info(" Fusion: %f", sum(self._cross_op.timers['fusion']))
        log.info(" Build: %f", sum(self._cross_op.timers['build']))
        log.info("Mutation: %f", sum(self._timers['mutation']))
        log.info("Population restart: %f", sum(self._timers['restart_pop']))
        if self._data.best_solution:
            log.info("---------------- Best known solution ------------------")
            log.info("Tour: %s", (self._data.best_solution.tour,))
            log.info("Distance: %f", self._data.best_solution.dist)
        log.info("------------------- Best individual found -----------------")
        log.info("Tour: %s", (self._best_solution.tour,))
        log.info("Distance: %f", self._best_solution.dist)
        log.info("-----------------------------------------------------------")
