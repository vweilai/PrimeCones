import math
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
from bisect import bisect_left, bisect_right

#####################################################
# 1. Prime sieve + fast interval prime counting
#####################################################

def sieve_primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start::step] = b'\x00' * ((limit - start)//step + 1)
    return [i for i in range(limit+1) if sieve[i]]

GLOBAL_PRIMES = None
GLOBAL_PRIME_LIMIT = None

def ensure_sieve(up_to):
    global GLOBAL_PRIMES, GLOBAL_PRIME_LIMIT
    if (GLOBAL_PRIMES is None) or (up_to > GLOBAL_PRIME_LIMIT):
        print(f"Building prime table up to {up_to} ...")
        GLOBAL_PRIMES = sieve_primes(up_to)
        GLOBAL_PRIME_LIMIT = up_to

def list_primes_interval(a, b):
    ensure_sieve(b)
    lo = bisect_left(GLOBAL_PRIMES, a)
    hi = bisect_right(GLOBAL_PRIMES, b)
    return GLOBAL_PRIMES[lo:hi]

#####################################################
# 2. Conical interval definition
#####################################################

def conical_interval_bounds(r, alpha, k):
    L = k * (r ** alpha)
    a = int((r - 1) * L)
    b = int(r * L)
    mid = (a + b) / 2
    return a, b, L, mid

#####################################################
# 3. Prime race for one (q, alpha, k)
#####################################################

def euler_phi(n):
    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1
    if x > 1:
        result -= result // x
    return result

RACE_CLASSES = {
    3: (1,2),
    4: (1,3),
    5: (1,2),
    8: (1,3),
}

def compute_prime_race_one(q, alpha, k, r_min, r_max):
    a1, a2 = RACE_CLASSES[q]
    phi_q = euler_phi(q)
    rows = []
    for r in range(r_min, r_max+1):
        a, b, L, mid = conical_interval_bounds(r, alpha, k)
        if mid <= 10:
            continue
        primes = list_primes_interval(a, b)
        if not primes:
            continue
        N1 = sum(1 for p in primes if p % q == a1)
        N2 = sum(1 for p in primes if p % q == a2)

        E = L / math.log(mid)
        E1 = E / phi_q
        E2 = E / phi_q

        Z1 = (N1 - E1) / math.sqrt(E1)
        Z2 = (N2 - E2) / math.sqrt(E2)
        D = Z1 - Z2

        rows.append(dict(r=r, Z1=Z1, Z2=Z2, D=D))
    return pd.DataFrame(rows)

#####################################################
# 4. Scan correlation for (q, alpha, k)
#####################################################

def scan_corr_for_q(q, ALPHAS, KS, r_min, r_max):
    rows = []
    for alpha in ALPHAS:
        for k in KS:
            print(f"[corr] q={q}, alpha={alpha}, k={k}")
            df = compute_prime_race_one(q, alpha, k, r_min, r_max)
            n = len(df)
            if n < 5:
                rows.append(dict(
                    q=q, alpha=alpha, k=k, n=n,
                    Z1_var=np.nan, Z2_var=np.nan, D_var=np.nan,
                    corr_Z1_Z2=np.nan,
                ))
                continue
            rows.append(dict(
                q=q, alpha=alpha, k=k, n=n,
                Z1_var=df["Z1"].var(),
                Z2_var=df["Z2"].var(),
                D_var=df["D"].var(),
                corr_Z1_Z2=df["Z1"].corr(df["Z2"]),
            ))
    return pd.DataFrame(rows)

#####################################################
# 5. Example run
#####################################################

if __name__ == "__main__":
    ALPHAS = [0.3, 0.7]
    KS = [4, 16]
    df3 = scan_corr_for_q(3, ALPHAS, KS, 500, 2000)
    df4 = scan_corr_for_q(4, ALPHAS, KS, 500, 2000)

    print("\n=== q=3 ===")
    print(df3)
    print("\n=== q=4 ===")
    print(df4)
