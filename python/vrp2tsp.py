#!/usr/bin/python
# ozeasx@gmail.com

from chromosome import Chromosome
from vrp_chromosome import VRP_Chromosome


def vrp2tsp(vrp):

    tsp_tour = list()
    ghost = vrp.dimension + 1

    for i in vrp.tour:
        if i == 1:
            tsp_tour.append(ghost)
            ghost += 1
        else:
            tsp_tour.append(i)

    return Chromosome(tsp_tour, vrp.dist)


def tsp2vrp(tsp, trucks):
    vrp_tour = list()
    dimension = len(tsp.tour) - trucks
    ghost_depots = range(dimension + 1, trucks + 1)
    for i in tsp.tour:
        if i in ghost_depots:
            vrp_tour.append(1)
        else:
            vrp_tour.append(i)

    return VRP_Chromosome(vrp_tour)
