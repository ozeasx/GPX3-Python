#!/usr/bin/python
# ozeasx@gmail.com

from tsp import TSPLIB
from chromosome import Chromosome
from gpx import GPX

# Snipet code to test a lot of random cases
tsp = TSPLIB("../tsplib/ulysses16.tsp")
gpx = GPX(tsp)

# Whitley2010-F1
# p1 = Chromosome([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# p2 = Chromosome([1, 13, 14, 12, 11, 3, 10, 9, 6, 8, 7, 5, 4, 2])

# Ulysses test f1, f2 problem
# p1 = Chromosome([6, 12, 9, 13, 14, 8, 4, 16, 15, 3, 1, 7, 10, 5, 2, 11])
# p2 = Chromosome([12, 4, 2, 1, 14, 15, 13, 6, 3, 16, 9, 5, 8, 11, 10, 7])

# Ulysses test f1, f2 problem
# p1 = Chromosome([3, 1, 13, 11, 15, 9, 16, 5, 12, 14, 2, 4, 10, 7, 6, 8])
# p2 = Chromosome([14, 11, 3, 9, 6, 10, 13, 2, 7, 12, 1, 4, 5, 16, 8, 15])

# p1 = Chromosome([10, 14, 9, 6, 2, 11, 16, 1, 4, 3, 7, 8, 12, 15, 5, 13])
# p2 = Chromosome([8, 5, 4, 10, 14, 16, 3, 12, 9, 11, 15, 1, 2, 13, 7, 6])

# p1 = Chromosome([2, 16, 11, 6, 8, 15, 4, 3, 12, 14, 7, 5, 10, 13, 1, 9])
# p2 = Chromosome([2, 16, 12, 10, 15, 14, 9, 6, 11, 3, 7, 4, 13, 8, 1, 5])

# p1 = Chromosome([5, 6, 1, 8, 2, 9, 3, 7, 4, 12, 16, 11, 15, 14, 10, 13])
# p2 = Chromosome([11, 16, 5, 9, 7, 6, 15, 3, 14, 10, 13, 2, 12, 8, 1, 4])

# Tinos2014-F1
# p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11])
# p2 = Chromosome([1,11,9,10,7,8,6,4,5,2,3])

# Tinos2014-F2
# p1 = Chromosome([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
# p2 = Chromosome([1,14,11,12,10,13,9,15,8,7,5,6,3,4,2])

# Tinos2018b-F5
# p1 = Chromosome((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
#                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32))
# p2 = Chromosome((1, 32, 31, 11, 12, 28, 27, 26, 25, 20, 19, 17, 18, 15, 16,
#                 14, 13, 29, 30, 10, 9, 7, 8, 5, 6, 4, 3, 22, 21, 24, 23, 2))
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
# p1 = Chromosome([1, 9, 8, 10, 7, 2, 5, 4, 3, 6])
# p2 = Chromosome([1, 8, 6, 2, 3, 10, 5, 9, 4, 7])

# F10 2f2 2if
# p1 = Chromosome([1,2,6,7,11,4,9,3,8,5,10,12])
# p2 = Chromosome([1,11,8,3,7,12,10,6,9,4,5,2])

# F11 4i ?
# p1 = Chromosome([1,2,9,7,10,12,3,6,5,4,11,8])
# p2 = Chromosome([1,5,12,11,9,3,10,8,7,6,2,4])

# F12 5i (2 fusions)
# p1 = Chromosome([1, 3, 10, 8, 11, 2, 5, 7, 6, 4, 9, 12])
# p2 = Chromosome([1, 4, 2, 8, 12, 6, 3, 9, 11, 10, 5, 7])

# F13 1f1, 2if
# p1 = Chromosome([1,4,3,9,6,2,8,7,10,5])
# p2 = Chromosome([9,7,10,6,3,8,5,4,2,1])

#  F14 2f2 2if
# p1 = Chromosome([1, 9, 5, 8, 7, 11, 4, 10, 3, 12, 6, 2])
# p2 = Chromosome([1, 10, 7, 11, 6, 9, 3, 12, 5, 8, 4, 2])

#  F15 (1f1, 1inf)
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

# ????
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
# p1 = Chromosome((1, 12, 7, 15, 4, 5, 11, 6, 13, 14, 3, 2, 10, 16, 8, 9))
# p2 = Chromosome((1, 5, 12, 10, 16, 13, 9, 11, 3, 7, 2, 15, 14, 6, 8, 4))

# F20
# p1 = Chromosome((1, 4, 11, 10, 16, 13, 3, 9, 6, 8, 2, 14, 15, 12, 7, 5))
# p2 = Chromosome((1, 7, 6, 5, 11, 9, 10, 12, 13, 14, 15, 16, 8, 4, 2, 3))

# F21
# p1 = Chromosome((12, 2, 1, 9, 5, 3, 8, 7, 4, 11, 10, 6))
# p2 = Chromosome((11, 8, 6, 5, 4, 1, 10, 3, 2, 7, 9, 12))

# F22
# p1 = Chromosome((9, 14, 2, 5, 3, 11, 6, 1, 4, 12, 15, 7, 10, 13, 16, 8))
# p2 = Chromosome([6, 1, 15, 3, 16, 2, 4, 8, 9, 11, 10, 7, 13, 14, 12, 5])

p1 = Chromosome([11, 1, 3, 16, 4, 9, 14, 10, 7, 2, 13, 5, 6, 8, 12, 15])
p2 = Chromosome([3, 1, 14, 7, 8, 2, 16, 9, 15, 5, 10, 11, 6, 13, 12, 4])

# vrp1
# p1 = Chromosome((1, 2, 3, 10, 4, 5, 6, 11, 7, 8, 9))
# p2 = Chromosome((1, 2, 3, 4, 5, 10, 6, 7, 8, 9))

# vrp2
# p1 = Chromosome((15, 25, 41, 42, 21, 39, 38, 33, 45, 32, 43, 30, 9, 34, 2, 24,
#                 1, 20, 46, 6, 3, 40, 29, 12, 4, 44, 35, 27, 31, 19, 14, 26,
#                 11, 23, 16, 22, 5, 37, 47, 7, 10, 48, 13, 28, 36, 8, 17, 18))

# p2 = Chromosome((4, 1, 3, 24, 14, 9, 45, 38, 26, 46, 13, 32, 44, 20, 35, 12, 7,
#                 16, 19, 5, 2, 34, 6, 30, 47, 18, 27, 33, 43, 29, 8, 48, 22,
#                 25, 42, 40, 36, 41, 11, 23, 39, 21, 31, 15, 17, 10, 37, 28))

# p1 = Chromosome((1, 10, 16, 2, 37, 4, 3, 17, 18, 15, 14, 13, 12, 19, 11, 27,
#                 21, 22, 1, 26, 24, 23, 20, 9, 45, 44, 35, 43, 39, 1, 25, 38,
#                 5, 36, 8, 6, 7, 28, 30, 29, 34, 32, 31, 42, 41, 40, 1, 33))

# p2 = Chromosome((1, 22, 21, 20, 27, 23, 24, 26, 11, 18, 4, 37, 3, 2, 16, 10,
#                 1, 19, 12, 13, 14, 15, 17, 38, 39, 43, 40, 41, 35, 32, 42,
#                 44, 1, 9, 45, 31, 33, 7, 6, 36, 1, 25, 5, 8, 28, 30, 34,
#                 29))

p1.dist = tsp.tour_dist(p1.tour)
p2.dist = tsp.tour_dist(p2.tour)

gpx.f1_test = True
gpx.f2_test = False
gpx.f3_test = False
gpx.ff1_test = True
gpx.ff2_test = False
gpx.ff3_test = False
gpx.fusion_on = False

c1, c2 = gpx.recombine(p1, p2)

print "Results ---------------------------------------------------------------"
print
print "Tour 1: ", p1.tour, ", Distance: ", p1.dist
print "Tour 2: ", p2.tour, ", Distance: ", p2.dist
print
print "Internal tour a: ", gpx.info['tour_a']
print "Internal tour b: ", gpx.info['tour_b']
print
print "Execution Time --------------------------------------------------------"
print
print "\tPartitioning: ", sum(gpx.timers['partition'])
print "\tsimple graphs: ", sum(gpx.timers['simple graph'])
print "\tClassification: ", sum(gpx.timers['classify'])
print "\tFusion: ", sum(gpx.timers['fusion'])
print "\tBuild: ", sum(gpx.timers['build'])
print "\tRecombination: ", sum(gpx.timers['recombination'])
print
print "Partitioning ----------------------------------------------------------"
print
print "\tFeasible type 1: ", gpx.counters['feasible_1']
print "\tFeasible type 2: ", gpx.counters['feasible_2']
print "\tFeasible type 3: ", gpx.counters['feasible_3']
print "\tInfeasible: ", gpx.counters['infeasible']
print "\tFusions: ", gpx.counters['fusion']
print "\tUnsolved: ", gpx.counters['unsolved']
print "\tInfeasible tour: ", gpx.counters['inf_tour']
print
print "Partitions vertices: --------------------------------------------------"
print
for key, value in gpx.info['vertices'].items():
    print key, value
print
print "AB_cycles: -----------------------------------------------------------"
print
for key, value in gpx.info['ab_cycles'].items():
    print key, value
print
print "simple graph a"
for key in gpx.info['simple_a']:
    print
    for k in gpx.info['simple_a'][key]:
        print key, k, gpx.info['simple_a'][key][k]
print
print "simple graph b"
for key in gpx.info['simple_b']:
    print
    for k in gpx.info['simple_b'][key]:
        print key, k, gpx.info['simple_b'][key][k]
print
print "Feasible: ", gpx.info['feasible']
print "Infeasible: ", gpx.info['infeasible']
if c1:
    print
    print "Solutions ---------------------------------------------------------"
    print
    print "Solution 1: ", c1.tour, ", Distance: ", c1.dist
    print
    print "Solution 2: ", c2.tour, ", Distance: ", c2.dist
    print
    print "Improvement: ", (p1.dist + p2.dist) - (c1.dist + c2.dist)
