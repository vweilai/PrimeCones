# ============================================
#  Conical Prime Race — Real Prime Version
#  直接可在 Colab 运行
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from math import gcd
import math
import time

# -----------------------------
#  0. 全局参数（你可以按需改）
# -----------------------------
X_MAX = 2_000_000        # 素数计算的最大 x（越大越慢、越占内存）
params_q     = [3, 4, 5, 8, 10]
params_alpha = [0.4, 0.6, 0.8]   # 先用 3 个 alpha，避免窗口太大
params_k     = [8, 16, 32, 64]

R_MIN = 50_000           # r 采样最小值
R_MAX = 200_000          # r 采样最大值
N_samples_per_combo = 1500   # 每组 (q,alpha,k) 的 r 采样点数


# =============================
#  1. 筛法生成素数
# =============================
def sieve_primes(n: int):
    """简单埃氏筛，返回 <= n 的所有素数（numpy 数组）"""
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    limit = int(n**0.5) + 1
    for p in range(2, limit):
        if sieve[p]:
            sieve[p*p:n+1:p] = False
    return np.nonzero(sieve)[0]

print("开始筛素数到", X_MAX, "...")
t0 = time.time()
primes = sieve_primes(X_MAX)
print(f"素数个数 = {len(primes)}, 用时 {time.time()-t0:.2f}s\n")


# =============================
#  2. 预计算每个 q 的 π(x; q,a)
# =============================
# prime_pi_by_q[q] = (residues_array, counts_array)
#   residues_array : 模 q 的互素类，如 [1,2] 或 [1,3,5,7]
#   counts_array   : shape=(d, X_MAX+1)
#                    counts_array[j,x] = π(x; q, residues[j])
prime_pi_by_q = {}

print("开始为每个 q 预计算 π(x; q,a) 累积表 ...")
t0 = time.time()
for q in params_q:
    residues = [a for a in range(q) if gcd(a, q) == 1]
    d = len(residues)
    print(f"  q = {q}, φ(q) = {d}")

    counts = np.zeros((d, X_MAX+1), dtype=np.int32)

    # 建立一个快速映射：余数 -> 行号
    mod_map = np.full(q, -1, dtype=int)
    for idx, a in enumerate(residues):
        mod_map[a] = idx

    # 遍历所有素数，按模类加到相应行
    for p in primes:
        r = p % q
        idx = mod_map[r]
        if idx != -1:
            counts[idx, p] += 1

    # 对每一行做前缀和，得到 π(x; q,a)
    counts = counts.cumsum(axis=1)
    prime_pi_by_q[q] = (np.array(residues, dtype=int), counts)

print(f"预计算完成，用时 {time.time()-t0:.2f}s\n")


# =============================
#  3. 生成真实的锥形 prime race 样本
# =============================
def generate_race_samples(q, alpha, k, N_samples):
    """
    返回 shape = (M, d) 的矩阵 Z，
    每一行是一个 Z(r) ∈ R^d 的锥形 prime race 噪声向量
    （这里 M ≤ N_samples，取决于窗口是否超界）
    """
    residues, counts = prime_pi_by_q[q]
    d = len(residues)

    # 在 [R_MIN, R_MAX] 上取等距采样点
    rs = np.linspace(R_MIN, R_MAX, N_samples)
    Z_list = []

    for r in rs:
        L = k * (r ** alpha)     # 锥形窗口长度 L(r) = k * r^alpha
        x1 = int(r)
        x2 = int(r + L)
        if x2 > X_MAX:
            # 超出我们预计算的范围，略过
            continue

        # 各模类的素数个数：π(x2; q,a) - π(x1; q,a)
        c2 = counts[:, x2]
        c1 = counts[:, x1]
        diff = c2 - c1           # shape = (d,)

        # 近似期望：总素数 ~ L / log r，平均到 φ(q) 个模类
        x_mid = max(r, 10.0)
        mu_total = L / math.log(x_mid)
        mu_per_class = mu_total / d
        if mu_per_class <= 0:
            mu_per_class = 1e-6

        # 标准化：Z_j(r) = (N_j - μ) / sqrt(μ)
        z = (diff - mu_per_class) / math.sqrt(mu_per_class)
        Z_list.append(z.astype(np.float64))

    if not Z_list:
        raise ValueError(f"参数 (q={q}, alpha={alpha}, k={k}) 下没有有效样本（窗口可能过大）")

    Z = np.vstack(Z_list)
    return Z  # shape = (M, d)


# =============================
#  4. 主方向（最大特征向量）
# =============================
def principal_direction(Z):
    """
    对样本矩阵 Z（M x d）做协方差 + 特征分解，
    返回主特征向量 v1（方向统一到 ∑v1>0）
    """
    # 协方差矩阵
    C = (Z.T @ Z) / Z.shape[0]          # d x d，对称矩阵

    # 特征分解（对称用 eigh）
    w, V = np.linalg.eigh(C)
    idx_max = np.argmax(w)
    v1 = V[:, idx_max]
    v1 = v1 / np.linalg.norm(v1)

    # 方向统一：让分量和为正
    if v1.sum() < 0:
        v1 = -v1

    return v1, C


# =============================
#  5. 扫描参数，计算每组主方向
# =============================
print("开始对 (q, alpha, k) 扫描并计算主方向...\n")
t0 = time.time()

principal_dirs = {}  # (q,alpha,k) -> v1
eigvals_store   = {} # (q,alpha,k) -> eigenvalues

for q in params_q:
    for alpha in params_alpha:
        for k in params_k:
            print(f"  计算 q={q}, alpha={alpha}, k={k} ...")
            Z = generate_race_samples(q, alpha, k, N_samples_per_combo)
            v1, C = principal_direction(Z)
            w, _ = np.linalg.eigh(C)
            principal_dirs[(q, alpha, k)] = v1
            eigvals_store[(q, alpha, k)]   = np.sort(w)[::-1]  # 从大到小排序

print(f"\n全部计算完成，用时 {time.time()-t0:.2f}s\n")


# =============================
#  6. 在“每个 q 内部”做主方向普适性测试
# =============================
cos_table = defaultdict(dict)

for q in params_q:
    # 选同一个 q 内的某一组 (alpha,k) 作为基准，比如 alpha=0.6,k=32
    ref_key = (q, 0.6, 32)
    v_ref = principal_dirs[ref_key]
    d_ref = len(v_ref)

    print(f"======= q = {q} (维度 = {d_ref}) =======")
    for alpha in params_alpha:
        row = []
        for k in params_k:
            v = principal_dirs[(q, alpha, k)]
            assert len(v) == d_ref   # 同一个 q 必然维度相同

            cos_theta = float(np.dot(v_ref, v))
            cos_table[q][(alpha, k)] = cos_theta
            row.append(f"{cos_theta:.4f}")
        print(f"alpha={alpha}   " + "  ".join(row))
    print()


# =============================
#  7. 画热力图：展示每个 q 内部主方向的一致性
# =============================
def plot_heatmap(q):
    data = np.array([[cos_table[q][(alpha, k)] for k in params_k]
                     for alpha in params_alpha])
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap='viridis', vmin=-1, vmax=1)
    plt.title(f"Principal Direction Similarity (q={q})")
    plt.xticks(range(len(params_k)), params_k)
    plt.yticks(range(len(params_alpha)), params_alpha)
    plt.xlabel("k")
    plt.ylabel("alpha")
    plt.colorbar(label="cos(theta)")
    plt.show()

for q in params_q:
    plot_heatmap(q)


# =============================
#  8. 额外：看一下特征值谱（判断是否近似各向同性 + rank-1）
# =============================
print("部分参数下的特征值（从大到小）示例：\n")
for q in params_q:
    key = (q, 0.8, 32)
    w = eigvals_store[key]
    print(f"q={q}, alpha=0.8, k=32, eigenvalues = {w}")
