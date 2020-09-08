import sys

import random

def solve(L,d):
    if L<=d: return 0.000

    simulation_steps=5000000
    ret = 0
    count = 0
    for i in range(simulation_steps):

        now_length=L
        while(now_length>d):
            cut_length=random.uniform(0,now_length)
            count+=1
            now_length-=cut_length


    ret=float(count)/simulation_steps


    return round(ret,4)


#
for line in sys.stdin:
    a = line.split()
    L=int(a[0])
    d=int(a[1])
    print(solve(L,d))
#