"""test_helpers.py

This file has routines needed by tests. They needed to be in a proper module.
"""
import numpy as np
import phathom.utils

def check_range(args):
    start, end, cprimes = args
    primes = np.ones(end - start, bool)
    for cprime in cprimes:
        offset = start % cprime
        idx0 = 0 if offset == 0 else cprime - offset
        if cprime >= start:
            idx0 += cprime
        primes[idx0::cprime] = False
    return np.where(primes)[0] + start


def find_primes(start, end):
    if end < 2:
        return []
    elif start <= 2 and end == 3:
        return [2]
    pstart = 2
    pend = max(2, int(np.sqrt(end)))+1
    cprimes = find_primes(pstart, pend)
    if end - start < 100000:
        see = [(start, end, cprimes)]
    else:
        see = [(_, min(_+100000, end), cprimes)
               for _ in range(start, end, 100000)]
    return np.concatenate(list(phathom.utils.parallel_map(check_range, see)))

