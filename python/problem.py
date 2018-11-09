#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations
import random

vertices = set(xrange(1, 10 + 1))
edges = set(combinations(vertices, 2))

visited = set()
result = set()

proceed = True
while len(result) < 946:
    aux = list(edges)
    random.shuffle(aux)
    gen = set()
    for edge in aux:
        if edge[0] not in visited and edge[1] not in visited:
            visited.update([edge[0], edge[1]])
            gen.add(edge)
    visited = set()
    gen = frozenset(gen)
    if gen not in result:
        result.add(gen)

for gen in sorted(result):
    print sorted(gen)

#test = combinations(result, 2)
#result2 = set()
#for t in test:
#    print t
#    if len(t[0] & t[1]) == 0:
#        result2.add(t)

#for gen in sorted(result2):
#    print sorted(gen)
#print len(result2)
