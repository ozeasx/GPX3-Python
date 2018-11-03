#!/usr/bin/python
# ozeasx@gmail.com

from shell import Shell
from tsp import TSPLIB
from chromosome import Chromosome
from gpx import GPX

# Snipet code to test a lot of random cases
# cmd = Shell()
# tsp = TSPLIB("../tsplib/ulysses16.tsp", cmd)
# gpx = GPX(tsp)
gpx = GPX()

# p1 = Chromosome(16)
# p1.dist = tsp.tour_dist(p1.tour)

# r1 = mut.two_opt(p1, tsp)

# print p1.tour
# print p1.dist
# print r1.tour
# print r1.dist

# def test(data, limit, dimension = None):
#     for x in xrange(limit):
#         p1 = Chromosome(data)
#         p2 = Chromosome(data)
#         while (p1 == p2):
#             p2 = Chromosome(data)
#         c1, c2 = p1 * p2
#         print (c1.get_dist() + c2.get_dist(]) - (p1.get_dist() +
#                p2.get_dist(])
#         print c1.get_tour()
#         print c2.get_tour()
#         print p1.get_tour()
#         print p1.get_dist()
#         p1.two_opt()
#         print p1.get_tour()
#         print p1.get_dist()
#         print '\r', x,

# test(tsp, 10000)

# Whitley2010-F1
# p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
# p2 = Chromosome([1,13,14,12,11,3,10,9,6,8,7,5,4,2])

# Tinos2014-F1
# p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
# p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
# p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
# p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
# p1 = Chromosome((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#                 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32))
# p2 = Chromosome((1, 32, 31, 11, 12, 28, 27, 26, 25, 20, 19, 17, 18, 15, 16, 14,
#                 13, 29, 30, 10, 9, 7, 8, 5, 6, 4, 3, 22, 21, 24, 23, 2))
# p2 = Chromosome([1,2,23,24,21,22,3,4,6,5,8,7,9,10,30,29,13,14,16,15,18,17,19,
#                 20,25,26,27,28,12,11,31,32])

# Hains 2011F-2.3
# p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
# p2 = Chromosome([1,13,12,10,9,7,6,8,5,4,11,3,2,14])

#  Whitley2011-F1
# p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
#               24,25,26,27,28])
# p2 = Chromosome([1,3,2,5,4,6,7,11,12,10,13,14,15,17,16,19,18,20,21,9,8,22,23,
#                  25,24,27,26,28])

# F1 2i
# p1 = Chromosome([1, 7, 5, 6, 2, 10, 9, 3, 8, 4])
# p2 = Chromosome([1, 3, 8, 9, 7, 10, 2, 4, 5, 6])

# F2 3i (2f1)
# p1 = Chromosome([1,10,4,6,2,8,5,9,3,7])
# p2 = Chromosome([1,3,2,9,7,5,6,10,4,8])

# F6 2f1
# p1 = Chromosome([1,3,6,7,10,4,8,11,2,5,9])
# p2 = Chromosome([1,4,8,2,11,5,9,6,7,10,3])

# F7 2i
# p1 = Chromosome([1,7,4,8,6,5,2,3])
# p2 = Chromosome([1,5,2,8,6,7,4,3])

# F8 3i
# p1 = Chromosome([1,2,3,4,5,6,7,8,9])
# p2 = Chromosome([1,8,6,4,2,9,7,5,3])

# F9 1f, 3i (1f, 1f2, 2i) execption
# p1 = Chromosome([1,9,8,10,7,2,5,4,3,6])
# p2 = Chromosome([1,8,6,2,3,10,5,9,4,7])

# F10 2f2 2if
# p1 = Chromosome([1,2,6,7,11,4,9,3,8,5,10,12])
# p2 = Chromosome([1,11,8,3,7,12,10,6,9,4,5,2])

# F11 4i (2 fusions f2) execption
# p1 = Chromosome([1,2,9,7,10,12,3,6,5,4,11,8])
# p2 = Chromosome([1,5,12,11,9,3,10,8,7,6,2,4])

# F12 5i (one fusion)
# p1 = Chromosome([1,3,10,8,11,2,5,7,6,4,9,12])
# p2 = Chromosome([1,4,2,8,12,6,3,9,11,10,5,7])

# F13 1f1, 2if
# p1 = Chromosome([1,4,3,9,6,2,8,7,10,5])
# p2 = Chromosome([9,7,10,6,3,8,5,4,2,1])

#  F14 2f2 2if
# p1 = Chromosome([1, 9, 5, 8, 7, 11, 4, 10, 3, 12, 6, 2])
# p2 = Chromosome([1, 10, 7, 11, 6, 9, 3, 12, 5, 8, 4, 2])

#  F15 (1f1, 1f2)
# p1 = Chromosome([1, 8, 4, 5, 6, 7, 9, 2, 10, 3])
# p2 = Chromosome([1, 6, 9, 8, 3, 7, 5, 10, 4, 2])

#  F16
# p1 = Chromosome([1, 8, 10, 2, 6, 9, 5, 3, 7, 4])
# p2 = Chromosome([1, 4, 5, 2, 3, 9, 6, 10, 8, 7])

#  Teste
# p1 = Chromosome([1,2,3,4,5,6])
# p2 = Chromosome([1,2,3,6,4,5])

#  Force test
# p1 = Chromosome(100)
# p2 = Chromosome(100)

#  Minimal exemple
# p1 = Chromosome([1,2,3,4])
# p2 = Chromosome([1,3,2,4])

#  FUck me
# p1 = Chromosome([1, 3, 2, 4, 8, 16, 15, 14, 13, 12, 10, 9, 11, 5, 6, 7])
# p2 = Chromosome([1, 15, 3, 11, 5, 9, 7, 16, 12, 13, 14, 6, 10, 4, 2, 8])

# F17
# p1 = Chromosome((1,-1,15,-15,10,-10,16,-16,2,-2,4,-4,8,-8,7,-7,13,-13,12,-12,
#                 14,-14,3,-3,5,-5,6,-6,11,-11,9,-9))
# p2 = Chromosome((1,-1,7,-7,9,-9,8,-8,13,-13,15,-15,6,-6,4,-4,5,-5,14,-14,16,
#                 -16,11,-11,12,-12,10,-10,2,-2,3,-3))

# F18
# p1 = Chromosome((1, -1, 7, -7, 6, -6, 5, -5, 11, -11, 9, -9, 10, -10, 12,
#                   -12, 13, -13, 14, -14, 15, -15, 16, -16, 8, -8, 4, -4, 2,
#                   -2, 3, -3))
# p2 = Chromosome((1, -1, 14, -14, 2, -2, 13, -13, 9, -9, 8, -8, 6, -6, 16,
#                  -16, 11, -11, 3, -3, 5, -5, 10, -10, 4, -4, 7, -7, 12, -12,
#                   15, -15))

# F19
p1 = Chromosome((1, 12, 7, 15, 4, 5, 11, 6, 13, 14, 3, 2, 10, 16, 8, 9))
p2 = Chromosome((1, 5, 12, 10, 16, 13, 9, 11, 3, 7, 2, 15, 14, 6, 8, 4))

# F20
p1 = Chromosome((1, -1, 4, -4, 11, -11, 10, -10, 16, -16, 13, -13, 3, -3, 9, -9, 6, -6, 8, -8, 2, -2, 14, 15, 12, -12, 7, -7, 5, -5))
p2 = Chromosome((-1, 1, -7, 7, -6, 6, -5, 5, -11, 11, -9, 9, -10, 10, -12, 12, -13, 13, 14, 15, -16, 16, -8, 8, -4, 4, -2, 2, -3, 3))


# p1.dist = tsp.tour_dist(p1.tour)
# p2.dist = tsp.tour_dist(p2.tour)
# p1 = Chromosome(1000)
# p2 = Chromosome(1000)
r = gpx.recombine(p1, p2)

print "Results -------------------------------------------------------------"
# print
# print "Tour 1: ", p1.tour, ", Distance: "  # , p1.dist
# print "Tour 2: ", p2.tour, ", Distance: "  # , p2.dist
# print
# print "Internal tour a: ", gpx.tour_a
# print "Internal tour b: ", gpx.tour_b
# print
print "Execution Time ------------------------------------------------------"
print
print "Partitioning: ", sum(gpx.exec_time['partition'])
print "simple graphs: ", sum(gpx.exec_time['simple graph'])
print "Classification: ", sum(gpx.exec_time['classify'])
print "Fusion: ", sum(gpx.exec_time['fusion'])
print "Build: ", sum(gpx.exec_time['build'])
print "Recombination: ", sum(gpx.exec_time['recombination'])
print
print "Partitions ------------------------------------------------------"
print
print "Vertices: ", gpx.partitions['vertices']
print
print "AB_cycles: ", gpx.partitions['ab_cycles']
print
print "simple graph a: ", gpx.partitions['simple_graph_a']
print "simple graph b: ", gpx.partitions['simple_graph_b']
print
print "Feasible: ", gpx.partitions['feasible']
print "Infeasible: ", gpx.partitions['infeasible']
if r and p1 != r[0]:
    print
    print "Solutions -------------------------------------------------------"
    print
    print "Solution 1: ", r[0].tour, ", Distance: ", r[0].dist
    print
    print "Solution 2: ", r[1].tour, ", Distance: ", r[1].dist
