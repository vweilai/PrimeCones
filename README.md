# Conical Prime Intervals — Experimental Prime Noise Geometry

This repository reproduces a series of experiments showing that **prime number noise**, when measured over a new type of interval called a **conical interval**, exhibits several surprising and highly structured properties:

---

## 🔷 1. Conical Intervals

For parameters (alpha, k), define:

- Interval length:  
  L(r) = k * r^alpha

- Interval:  
  I_r = [(r-1)L(r), rL(r)]

- Standardised prime-count error:  
  Z(r) = (N(r) - E(r)) / sqrt(E(r))

Where N(r) is the number of primes in the interval and  
E(r) = L(r)/log(x_r) is the prime number theorem prediction.

---

## 🔷 2. Main Experimental Findings

### ✔ 1. Prime noise becomes Gaussian-like  
Z(r) is nearly perfectly normal with negligible autocorrelation.

### ✔ 2. Variance σ²(alpha, k) forms a smooth quadratic surface  
A remarkably stable fit:
  
  σ² ≈ A + Bα + C ln k + D α² + E (ln k)² + F α ln k

### ✔ 3. Prime races become almost uncorrelated  
For q = 3,4,5,8, the correlation corr(Z1, Z2) between different residue classes is:

  -0.07 ~ -0.15 (almost independent!)

This is unexpected — prime races normally share strongly coupled L-function noise.

### ✔ 4. The variances of Z1, Z2, and D = Z1 - Z2  
also lie on smooth quadratic surfaces, nearly identical across different q.

---

## 🔷 3. Running the Code

To run:

```bash
python3 main.py
