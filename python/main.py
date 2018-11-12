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
from gpx import GPX
from shell import Shell


# Argument parser
parser = argparse.ArgumentParser(description="Genetic algorithm + GPX + 2opt")

# Multual exclusive arguments
multual = parser.add_mutually_exclusive_group()
multual.add_argument("-k", help="Tournament size", type=int, default=0)
multual.add_argument("-P", help="Pairwise Recombination", default='False',
                     choices=['True', 'False'])
# Optional arguments
parser.add_argument("-p", help="Initial population", type=int, default=100)
parser.add_argument("-M", help="Method to generate inicial population",
                    choices=['random', '2opt'], default='random')
parser.add_argument("-r", help="Percentage of population to be restarted",
                    type=float, default=0)
parser.add_argument("-e", help="Elitism. Number of individuals to preserve",
                    type=int, default=0)
parser.add_argument("-c", help="Crossover probability", type=float, default=0)
parser.add_argument("-x", help="Crossover operator", choices=['GPX'],
                    default='GPX')
parser.add_argument("-m", help="Mutation probability (2opt)", type=float,
                    default=0)
parser.add_argument("-g", help="Generation limit", type=int, default=100)
parser.add_argument("-n", help="Number of iterations", type=int, default=1)
parser.add_argument("-o", help="Generate file reports", default='False',
                    choices=['True', 'False'])
parser.add_argument("-f1", help="Feasible 1 test", default='True',
                    choices=['True', 'False'])
parser.add_argument("-f2", help="Feasible 2 test", default='True',
                    choices=['True', 'False'])
parser.add_argument("-f3", help="Feasible 3 test", default='False',
                    choices=['True', 'False'])
# Mandatory argument
parser.add_argument("I", help="TSPLIB instance file", type=str)


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


# Create directory with timestamp
if args.o == 'True':
    log_dir = time.strftime("../results/%Y%m%d%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Redefine function to log assertion erros
    # http://code.activestate.com/recipes/577074-logging-asserts/
    def excepthook(*args):
        logging.getLogger().error('Uncaught exception:', exc_info=args)

    # Redefine sys.excepthook
    sys.excepthook = excepthook


# Function to call each ga run
def run_ga(id):
    # File logging
    if args.o == 'True':
        # Log file
        logging.basicConfig(filename=log_dir + "/report%i.log" % (id + 1),
                            format='%(message)s', level=logging.INFO)
        # Statistics
        fitness = defaultdict(list)
        counters = defaultdict(list)
        timers = defaultdict(list)
        # Stdout
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        # Only stdout
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Summary
    logging.info("------------------------------GA Settings------------------")
    logging.info("Initial population: %i", args.p)
    logging.info("Population restart percentage: %f", args.r)
    logging.info("Elitism: %i", args.e)
    logging.info("Tournament size: %i", args.k)
    logging.info("Pairwise formation: %s", args.P)
    logging.info("Crossover probability: %f", args.c)
    logging.info("Crossover operator: %s", args.x)
    logging.info("Mutation probability: %f", args.m)
    logging.info("Generation limit: %i", args.g)
    logging.info("TSPLIB instance: %s", args.I)
    logging.info("Iteration: %i/%i", id, args.n)

    # Needed objects
    cmd = Shell()
    tsp = TSPLIB(args.I, cmd)
    gpx = GPX(tsp)

    # Define which tests will be applied
    if args.f1 == 'False':
        gpx.f1_test = False
    if args.f2 == 'False':
        gpx.f2_test = False
    if args.f3 == 'True':
        gpx.f3_test = True

    # GA instance
    ga = GA(tsp, gpx, args.e)
    # Generate inicial population
    ga.gen_pop(args.p, args.M)
    # Fisrt population evaluation
    ga.evaluate()
    # Begin GA
    while ga.generation < args.g:
        # Store fitness evolution
        if args.o == 'True':
            fitness[ga.generation].append(ga.avg_fitness)
            fitness[ga.generation].append(ga.best_solution.fitness)
        # Generation info
        ga.print_info()
        # Selection
        if args.k:
            ga.select_tournament(args.k)
        # Recombination
        if args.c:
            ga.recombine(args.c, args.P)
        # Mutation
        if args.m:
            ga.mutate(args.m)
        # Population restart
        if args.r:
            ga.restart_pop(args.r, args.P)
        # Evaluation
        ga.evaluate()
    # Final report
    ga.report()
    if args.o == 'True':
        counters[id].extend([ga.cross, gpx.counters['failed'],
                             gpx.counters['feasible_1'],
                             gpx.counters['feasible_2'],
                             gpx.counters['feasible_3'],
                             gpx.counters['infeasible'],
                             gpx.counters['fusions'],
                             gpx.counters['unsolved'],
                             gpx.counters['inf_tours'], ga.mut])
        timers[id].extend([sum(ga.timers['total']),
                           sum(ga.timers['population']),
                           sum(ga.timers['evaluation']),
                           sum(ga.timers['selection']),
                           sum(ga.timers['recombination']),
                           sum(gpx.timers['partition']),
                           sum(gpx.timers['simple graph']),
                           sum(gpx.timers['classify']),
                           sum(gpx.timers['fusion']),
                           sum(gpx.timers['build']),
                           sum(ga.timers['mutation']),
                           sum(ga.timers['pop restart'])])
        return fitness, counters, timers


# Execution decision
if args.n > 1:
    # Execute all runs
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(run_ga, xrange(args.n))
    pool.close()
    pool.join()
else:
    result = run_ga(0)

# Statistics files
if args.o == 'True' and args.n > 1:
    # Consolidate data
    fitness = defaultdict(list)
    counters = defaultdict(list)
    timers = defaultdict(list)

    for f, c, t in result:
        for key, values in f.items():
            fitness[key].extend(values)
        for key, values in c.items():
            counters[key].extend(values)
        for key, values in t.items():
            timers[key].extend(values)

    # Write data
    with open(log_dir + "/fitness.out", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key in sorted(fitness):
            writer.writerow([key, ','.join(map(str, fitness[key]))])

    with open(log_dir + "/counters.out", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key in sorted(counters):
            writer.writerow(["run", "crossovers", "failed", "feasible 1",
                             "feasible 2", "feasible 3",
                             "infeasible", "fusions", "unsolved",
                             "infeasible tours", "mutations"])
            writer.writerow([key, ','.join(map(str, counters[key]))])

    with open(log_dir + "/timers.out", 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run", "total", "population", "evaluation",
                         "selection", "recombination", "partition",
                         "simple graph", "classify", "fusion", "build",
                         "mutation", "pop restart"])
        for key in sorted(timers):
            writer.writerow([key, ','.join(map(str, timers[key]))])
