# Parallelized implementation of a simple
# iteration scheme in Spark
# instead of the full molecular dynamics with S4 Electrodynamics
# we consider a simpler scheme with some of the characteristics
# to test out Spark distributed computing
# This simplified setting will give us an indication of
# how to do full molecular simulations using Spark

#  The scheme is the following
#  We consider A as a large matrix of size M x (N+1)
#  We randomly populate A(0) = [a_1(0), ..., a_N(0), a_{N+1}(0)]
#  The iteration step does the following
#     a_k^r(t+1) = a_k^r(t) + \sum_{s=1}^r a_k^s(t) a^{s-r}_{N+1}(t)
#     a^r_{N+1}^r(t+1)  = sqrt( \sum_{p=1}^N a_k^r(t+1)^2 )
#
#  This may seem complicated but it's quite a bit cleaner than
#  the actual S4 Electrodynamics system
#
#  In this situation, we can see that the (N+1)-th column is
#  crucial for every other column and is therefore rate-limiting.
#  We can deduce that at every time step, all columns will need
#  the values of the (N+1)=th column and therefore any distributed
#  algorithm will be limited by the computation of this column

import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark import SparkConf

from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import RowMatrix

conf = SparkConf()
conf.setMaster("local")
conf.setAppName("parallelism for S4 MD")
sc = SparkContext(conf=conf)


def compute_B(B, A):
    print(B.shape)
    Ap = np.array(A.rows.collect())
    for ir in range(M):
        vec = Ap[ir, :]
        val = np.sqrt(sum(vec**2))
        B[ir, 0] = val
    return(B)


def new_row_A(row, B):
    n = len(row)
    nrow = [0]*n
    # the following is not right yet
    for k in range(1, n):
        nrow[k] = nrow[k-1]*B[n-k-1, 0]
    nrow = nrow + row
    return(nrow)


N = 5000
M = 400

rows = sc.parallelize(np.random.rand(N, M))
A = RowMatrix(rows)
Bp = np.zeros((M, 1))
Bp = compute_B(Bp, A)
B = sc.broadcast(Bp)

# each worker should be updating some columns of A
for nt in range(5):
    def f(row):
        return(new_row_A(row, B.value))
    Anext = A.rows.map(f).collect()
    AnextRows = sc.parallelize(Anext)
    A = RowMatrix(AnextRows)
    Bp = compute_B(B.value, A)
    B = sc.broadcast(Bp)
    print(nt)
