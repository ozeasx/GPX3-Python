#!/usr/bin/python
# Ozeas - ozeasx@gmail.com

import time
import os
import sys
import multiprocessing
import argparse
from collections import defaultdict
import logging
import csv
from ga import GA
from tsp import TSPLIB
from shell import Shell


# Redefine function to log assertion erros
# http://code.activestate.com/recipes/577074-logging-asserts/
def excepthook(*args):
    logging.getLogger().error('Uncaught exception:', exc_info=args)


# Redefine sys.excepthook
sys.excepthook = excepthook

# Arguments parser
parser = argparse.ArgumentParser(description="Genetic algorithm + GPX + 2opt")
parser.add_argument("-p", help="Initial population", type=int, default=100)
parser.add_argument("-M", help="Method to generate inicial population",
                    choices=['random', 'two_opt'], default='random')
parser.add_argument("-r", help="Percentage of population to be restarted",
                    type=float, default=0)
parser.add_argument("-e", help="Elitism. Number of individuals to preserve",
                    type=int, default=0)
parser.add_argument("-c", help="Crossover probability", type=float, default=0)
parser.add_argument("-P", help="Pairwise Recombination", default='False')
parser.add_argument("-x", help="Crossover operator", choices=['gpx'],
                    default='gpx')
parser.add_argument("-m", help="Mutation probability (2opt)", type=float,
                    default=0)
parser.add_argument("-k", help="Tournament size", type=int, default=0)
parser.add_argument("-g", help="Generation limit", type=int, default=100)
parser.add_argument("I", help="TSPLIB instance file", type=str)
parser.add_argument("-n", help="Number of iterations", type=int, default=1)
parser.add_argument("-o", help="Generate file reports", default='False',
                    choices=['True', 'False'])


# Parser
args = parser.parse_args()

# Assert arguments
assert args.p > 0 and not args.p % 2, "Invalid population size. Must be even" \
                                      " and greater than 0"
assert 0 <= args.r <= 1, "Restart percentage must be in [0,1] interval"
assert 0 <= args.e <= args.p, "Invalid number of elite individuals"
assert 0 <= args.c <= 1, "Crossover probability must be in [0,1] interval"
assert 0 <= args.m <= 1, "Mutation probability must be in [0,1] interval"
assert 0 <= args.k <= args.p - args.e, "Invalid tournament size"
assert args.g > 0, "Invalid generation limit"
assert 0 < args.n <= 100, "Invalid iteration limit [0,100]"
assert os.path.isfile(args.I), "File " + args.I + " doesn't exist"

# Logging
if args.o == 'True':
    # Create directory with timestamp
    log_dir = time.strftime("../results/%Y%m%d%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Log file
    logging.basicConfig(filename=log_dir + "/report.log", format='%(message)s',
                        level=logging.INFO)
    # stdout
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
else:
    # Only stdout
    logging.basicConfig(format='%(message)s', level=logging.INFO)

# Dict to store fitness evolution
fitness_out = defaultdict(list)

# Summary
logging.info("------------------------------GA Settings----------------------")
logging.info("Initial population: %i", args.p)
logging.info("Population restart percentage: %f", args.r)
logging.info("Elitism: %i", args.e)
logging.info("Tournament size: %i", args.k)
logging.info("Pairs formation: %s", args.P)
logging.info("Crossover probability: %f", args.c)
logging.info("Crossover operator: %s", args.x)
logging.info("Mutation probability: %f", args.m)
logging.info("Generation limit: %i", args.g)
logging.info("TSPLIB instance: %s", args.I)
logging.info("Iterations: %i", args.n)

# Needed objects
cmd = Shell()
tsp = TSPLIB(args.I, cmd)
best_solution = None

# Loop over iteration limit
loop = args.n
while loop:
    # Logging
    logging.info("\nIteration %i --------------------------------------", loop)
    # Genetic algorithm
    ga = GA(tsp, args.e)
    # Generate inicial population
    ga.gen_pop(args.p, args.M)
    # Begin GA
    while ga.generation < args.g:
        # Evaluation
        ga.evaluate()
        fitness_out[ga.generation].extend([ga.avg_fitness,
                                           ga.best_solution.fitness])
        if not best_solution:
            best_solution = ga.best_solution
        else:
            if ga.best_solution.fitness > best_solution.fitness:
                best_solution = ga.best_solution
        # Selection
        if args.k:
            ga.select_tournament(args.k)
        # Recombination
        if args.c:
            if args.P == 'False':
                ga.recombine(args.c)
            else:
                ga.recombine(args.c, True)
        # Mutation
        if args.m:
            ga.mutate(args.m)
        # Population restart
        if args.r:
            ga.restart_pop(args.r)
        # Generation info
        ga.print_info()
    # Final report
    ga.report()
    # If better solution found, store in tsp and write to file
    tsp.best_solution = best_solution
    # Decrease iteration
    loop -= 1

# File reporting
if args.o == 'True':
    with open(log_dir + '/avg_fitness.out', 'wb') as file:
        wr = csv.writer(file)
        for key in sorted(fitness_out.keys()):
            wr.writerow(fitness_out[key])

    with open(log_dir + '/best_tour.out', 'wb') as file:
        wr = csv.writer(file)
        wr.writerow(best_solution.tour)
        wr.writerow([-best_solution.fitness])
