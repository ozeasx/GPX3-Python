#!/usr/bin/python
# ozeasx@gmail.com

import random
from chromosome import Chromosome


class VRP_chromosome(Chromosome):
    def __init__(self, dimension, trucks):
        # Assert valid dimension/trucks relations
        assert dimension - 1 >= trucks, "Invalid dimension/truck relation"
        # Random tour
        self._tour = range(2, dimension + 1)
        random.shuffle(self._tour)
        # Insert 'get back's to deposit



if __name__ == '__main__':
    p1 = VRP_chromosome(4, 2)
    print p1._tour
