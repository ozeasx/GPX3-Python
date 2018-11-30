#!/usr/bin/python
# ozeasx@gmail.com

from chromosome import Chromosome
from vrp_chromosome import VRP_Chromosome


# Given a VRP Chromosome, returns a TSP Chromosome
def vrp2tsp(vrp):
    # TSP tour
    tsp_tour = list()

    # Ghost depots numbering
    ghost = vrp.dimension + 1

    first = False
    for i in vrp.tour:
        if i == 1:
            # If first depot, append 1
            if not first:
                first = True
                tsp_tour.append(i)
                continue
            # Ghost depots
            else:
                tsp_tour.append(ghost)
                ghost += 1
        else:
            tsp_tour.append(i)

    return Chromosome(tsp_tour)


# Given a vrp in TSP format, returns a VRP Chromosome
def tsp2vrp(tsp, trucks):
    vrp_tour = list()
    dimension = len(tsp.tour) - trucks + 1
    ghost_depots = range(dimension + 1, dimension + trucks + 2)
    for i in tsp.tour:
        if i in ghost_depots:
            vrp_tour.append(1)
        else:
            vrp_tour.append(i)

    return VRP_Chromosome(vrp_tour)
