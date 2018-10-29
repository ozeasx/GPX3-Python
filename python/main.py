#!/usr/bin/python
# Ozeas - ozeasx@gmail.com

import os
import argparse
from ga import GA
from tsp import TSPLIB
from shell import Shell

# Arguments parser
parser = argparse.ArgumentParser(description="Genetic algorithm")
parser.add_argument("-p", help="inicial population", type=int, default=100)
parser.add_argument("-t", help="Inicial populatio creation method",
                    choices=['random', 'two_opt'], default='random')
parser.add_argument("-r", help="population restart percentage", type=float,
                    default=0)

parser.add_argument("-e", help="Elitism. Number of individuals", type=int,
                    default=0)

parser.add_argument("-c", help="crossover probability", type=float, default=0)
parser.add_argument("-x", help="crossover operator", choices=['gpx'],
                    default='gpx')

parser.add_argument("-m", help="mutation probability", type=float, default=0)
parser.add_argument("-k", help="tournament size", type=int, default=0)
parser.add_argument("-g", help="Generation limit", type=int, default=100)
parser.add_argument("i", help="TSPLIB instance file", type=str)
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
assert os.path.isfile(args.i), "File %s doesn't exist" % args.i

# Summary
print "------------------------------GA Settings------------------------------"
print "Initial population:", args.p
print "Population restart percentage:", args.r
print "Elitism:", args.e
print "Crossover probability:", args.c
print "Crossover operator:", args.x
print "Mutation probability:", args.m
print "Tournament size:", args.k
print "Generation limit:", args.g
print "TSPLIB instance:", args.i
print "-----------------------------------------------------------------------"

# Needed objects
cmd = Shell()
tsp = TSPLIB(args.i, cmd)

# Genetic algorithm
ga = GA(tsp, args.e)

ga.gen_pop(args.p, args.t)

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
