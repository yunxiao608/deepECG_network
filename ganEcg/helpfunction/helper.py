from __future__ import division
import math
import numpy as np

__all__ = ['frdist']

def sum(testlist):
    sum = 0
    for i in range(len(testlist)):
        sum += testlist[i]
    return sum

def listsumavg(testlist):
    sum = 0
    for i in range(len(testlist)):
        sum += testlist[i]
    return sum / len(testlist)

# mean function
def avg_list(mylist):
    sum = 0
    for i in range(len(mylist)):
        sum += mylist[i]
    return sum / len(mylist)

# variance function
def variance(mylist, mean):
    square_sum = 0
    for j in range(len(mylist)):
        square_sum += (mylist[j] - mean)**2
    return square_sum / len(mylist)

# one-dim array to two-dim array
def array_convert(testlist):
    mylist = [([0] * 2) for k in range(len(testlist))]
    for i in range(len(mylist)):
        for j in range(2):
            mylist[i][0] = i
            mylist[i][1] = testlist[i]
    return mylist

def RMSE(reallist, genlist):
    v = list(map(lambda x: (x[0] - x[1])**2, zip(reallist, genlist)))
    return avg_list(v)**0.5

def PRD(reallist, genlist):
    v = list(map(lambda x: (x[0] - x[1])**2, zip(reallist, genlist)))
    newlist = [reallist[i]**2 for i in range(len(reallist))]
    return (sum(v)/sum(newlist))**0.5*100


# calculate the discrete_frechet distance
def _c(ca, i, j, p, q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i] - q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, p, q), np.linalg.norm(p[i] - q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, p, q), np.linalg.norm(p[i] - q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max( \
            min( \
                _c(ca, i - 1, j, p, q), \
                _c(ca, i - 1, j - 1, p, q), \
                _c(ca, i, j - 1, p, q) \
                ), \
            np.linalg.norm(p[i] - q[j]) \
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]


def frdist(p, q):
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    if len_p != len_q or len(p[0]) != len(q[0]):
        raise ValueError('Input curves do not have the same dimensions.')

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    dist = _c(ca, len_p - 1, len_q - 1, p, q)
    return dist


# calculate the Hausdoff distance



