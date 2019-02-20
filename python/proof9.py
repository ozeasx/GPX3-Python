#!/usr/bin/python
# ozeasx@gmail.com

from itertools import combinations



l = list()
n = 42
last = None

for i, j in combinations(range(n), 2):
    if not last:
        last = index(i, j, n)
        continue
    if last:
        current = index(i ,j, n)
        if current - last != 1:
            print i, j
        else:
            last = current
