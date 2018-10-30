#!/usr/bin/python
# Ozeas - ozeasx@gmail.com

import time
import os
import sys
import argparse
import logging as log
from ga import GA
from tsp import TSPLIB
from shell import Shell


# http://code.activestate.com/recipes/577074-logging-asserts/
def excepthook(*args):
    log.getLogger().error('Uncaught exception:', exc_info=args)


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
parser.add_argument("-x", help="Crossover operator", choices=['gpx'],
                    default='gpx')
parser.add_argument("-m", help="Mutation probability (2opt)", type=float,
                    default=0)
parser.add_argument("-k", help="Tournament size", type=int, default=0)
parser.add_argument("-g", help="Generation limit", type=int, default=100)
parser.add_argument("I", help="TSPLIB instance file", type=str)
parser.add_argument("-n", help="Number of iterations", type=int, default=1)
parser.add_argument("-o", help="Generate report output file", default="False",
                    choices=["True", "False"])

# Parser
args = parser.parse_args()

# Logging
if args.o == 'True':
    # Create directory with timestamp
    dir = time.strftime("../results/%Y%m%d%H%M%S")
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Log file
    log.basicConfig(filename=dir + "/result.log", format='%(message)s',
                    level=log.INFO)
    # stdout
    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
else:
    # Only stdout
    log.basicConfig(format='%(message)s', level=log.INFO)

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

# Summary
log.info("------------------------------GA Settings--------------------------")
log.info("Initial population: %i", args.p)
log.info("Population restart percentage: %f", args.r)
log.info("Elitism: %i", args.e)
log.info("Crossover probability: %f", args.c)
log.info("Crossover operator: %s", args.x)
log.info("Mutation probability: %f", args.m)
log.info("Tournament size: %i", args.k)
log.info("Generation limit: %i", args.g)
log.info("TSPLIB instance: %s", args.I)

# Needed objects
cmd = Shell()
tsp = TSPLIB(args.I, cmd)

# Loop over iteration limit
loop = args.n
while loop:
    # Logging
    log.info("\nIteration %i ------------------------------------------", loop)
    # Genetic algorithm
    ga = GA(tsp, args.e)
    # Generate inicial population
    ga.gen_pop(args.p, args.M)
    # Begin GA
    while ga.generation < args.g:
        # Evaluation
        ga.evaluate()
        # Selection
        if args.k:
            ga.select_tournament(args.k)
        # Recombination
        if args.c:
            ga.recombine(args.c)
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
    # Decrease iteration
    loop -= 1
