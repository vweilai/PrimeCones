import math
import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right
from tqdm import tqdm

#####################################################
# 1. 素数筛
#####################################################

GLOBAL_PRIMES = None
GLOBAL_LIMIT = 0

def sieve_primes(limit):
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit**0.5)+1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start::step] = b"\x00" * ((limit-start)//step + 1)
    return [i for i in range(limit+1) if sieve[i]]

def ensure_sieve(n):
    global GLOBAL_PRIMES, GLOBAL_LIMIT
    if GLOBAL_PRIMES is None or n > GLOBAL_LIMIT:
        print(f"[sieve] building primes up to {n}")
        GLOBAL_PRIMES = sieve_primes(n)
        GLOBAL_LIMIT = n

def primes_in_interval(a, b):
    if b < 2 or b < a:
        return []
    ensure_sieve(b)
    lo = bisect_left(GLOBAL_PRIMES, max(a,2))
    hi = bisect_right(GLOBAL_PRIMES, b)
    return GLOBAL_PRIMES[lo:hi]

#####################################################
# 2. 锥形区间
#####################################################

def conical_interval(r, alpha, k):
    L = k * (r**alpha)
    a = int((r-1)*L)
    b = int(r*L)
    mid = 0.5*(a+b)
    return a, b, L, mid

#####################################################
# 3. 高维 prime race：q = 8
#####################################################

q = 8
classes = [1,3,5,7]      # 4 个互素同余类
phi_q = 4

def compute_Z_vector(r, alpha, k):
    """返回 Z 向量：Z_1, Z_3, Z_5, Z_7"""
    a, b, L, mid = conical_interval(r, alpha, k)
    if mid < 100:
        return None

    plist = primes_in_interval(a, b)
    if not plist:
        return None

    # 统计各类
    counts = {c:0 for c in classes}
    for p in plist:
        c = p % q
        if c in counts:
            counts[c] += 1

    # 期望
    E_total = L / math.log(mid)
    E_c = E_total / phi_q
    if E_c <= 0:
        return None

    Z = [(counts[c] - E_c)/math.sqrt(E_c) for c in classes]
    return Z

#####################################################
# 4. 扫描多个 r，构造样本矩阵
#####################################################

def collect_samples(alpha, k, r_min, r_max):
    Z_list = []
    for r in range(r_min, r_max+1):
        Z = compute_Z_vector(r, alpha, k)
        if Z is not None:
            Z_list.append(Z)
    return np.array(Z_list)

#####################################################
# 5. 计算协方差矩阵、相关矩阵、特征值
#####################################################

def analyze_cov(alpha, k, r_min=500, r_max=3000):
    print(f"\n=== q=8, alpha={alpha}, k={k} ===")
    Zmat = collect_samples(alpha, k, r_min, r_max)
    n = len(Zmat)
    print("samples:", n)

    Cov = np.cov(Zmat.T)
    Corr = np.corrcoef(Zmat.T)
    vals, vecs = np.linalg.eig(Cov)

    print("\nCovariance matrix:")
    print(Cov)

    print("\nCorrelation matrix:")
    print(Corr)

    print("\nEigenvalues:")
    print(vals)

    print("\nEigenvectors (columns):")
    print(vecs)

#####################################################
# 6. 主程序：测试几个 (alpha,k)
#####################################################

tests = [
    (0.3, 4),
    (0.3, 16),
    (0.7, 4),
    (0.7, 16)
]

for alpha, k in tests:
    analyze_cov(alpha, k)






# Covariance matrix:
# [[ 0.49291157 -0.03353312 -0.02248792 -0.02810767]
#  [-0.03353312  0.4849894  -0.02622264 -0.05376792]
#  [-0.02248792 -0.02622264  0.48297469 -0.018102  ]
#  [-0.02810767 -0.05376792 -0.018102    0.47823998]]

# Correlation matrix:
# [[ 1.         -0.06858408 -0.04608956 -0.05789185]
#  [-0.06858408  1.         -0.05418113 -0.11164368]
#  [-0.04608956 -0.05418113  1.         -0.0376653 ]
#  [-0.05789185 -0.11164368 -0.0376653   1.        ]]

# Eigenvalues:
# [0.39027406 0.53704361 0.51615443 0.49564354]

# Eigenvectors (columns):
# [[-0.43109575 -0.14703764 -0.88198017  0.12102628]
#  [-0.58345872  0.77524315  0.12773242 -0.2055685 ]
#  [-0.38149268 -0.11264602  0.32308017  0.85871614]
#  [-0.57288368 -0.60389475  0.3184566  -0.45354252]]