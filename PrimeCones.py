# ============================================================
# 0. 安装依赖（只需运行一次）
# ============================================================
!apt-get -qq install primesieve
!pip -q install numpy pandas scipy statsmodels tqdm matplotlib

# ============================================================
# 1. 导入库
# ============================================================
import math
import re
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
import matplotlib.pyplot as plt

rng = np.random.default_rng(20251127)

# ============================================================
# 2. 用系统 primesieve 计数区间 [a,b] 素数个数（稳健版）
# ============================================================
def prime_count_interval(a: int, b: int) -> int:
    """
    使用系统 primesieve 计算 [a,b] 内素数数量。
    自动处理所有非纯数字输出，不会炸掉。
    """
    if b < 2 or b < a:
        return 0
    a2 = max(a, 2)

    try:
        result = subprocess.run(
            ["primesieve", str(a2), str(b), "-c"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        nums = re.findall(r"\d+", output)
        if not nums:
            print("⚠ primesieve 没有输出整数，原始输出：", output)
            return 0
        return int(nums[-1])
    except Exception as e:
        print("⚠ primesieve 调用异常：", e)
        return 0

# ============================================================
# 3. 锥形区间几何 I_r
# ============================================================
def conical_interval_bounds(r: int, alpha: float, k: float):
    """
    返回: a, b, L, x_mid
    I_r = [a, b]
    L(r) = k * r^alpha
    x_mid ~ k * r^(alpha+1)
    """
    L = k * (r ** alpha)
    a = (r - 1) * L
    b = r * L
    x_mid = k * (r ** (alpha + 1))

    a_int = int(math.floor(a))
    b_int = int(math.floor(b))
    if b_int <= a_int:
        b_int = a_int + 1
    return a_int, b_int, L, x_mid

# ============================================================
# 4. 计算固定 (alpha, k) 下的 Z 序列
# ============================================================
def compute_Z_series(alpha: float, k: float, r_min: int, r_max: int):
    rs, Zs, Es = [], [], []
    for r in range(r_min, r_max + 1):
        a, b, L, x_mid = conical_interval_bounds(r, alpha, k)
        if b <= a or x_mid <= 10:
            continue

        E = L / math.log(x_mid)
        if E <= 0:
            continue

        N = prime_count_interval(a, b)
        Z = (N - E) / math.sqrt(E)

        rs.append(r)
        Zs.append(Z)
        Es.append(E)

    return np.array(rs), np.array(Zs), np.array(Es)

# ============================================================
# 5. 统计量
# ============================================================
def summarize_Z(Z: np.ndarray):
    n = len(Z)
    if n == 0:
        return dict(n=0, mean=np.nan, var=np.nan, skew=np.nan, kurt=np.nan)
    return dict(
        n=n,
        mean=float(np.mean(Z)),
        var=float(np.var(Z)),
        skew=float(skew(Z, bias=False)),
        kurt=float(kurtosis(Z, fisher=False, bias=False)),
    )

# ============================================================
# 6. 在参数网格上跑实验
# ============================================================
def experiment_grid(alphas, ks, r_min: int, r_max: int):
    rows = []
    total = len(alphas) * len(ks)
    pbar = tqdm(total=total, desc="Running (alpha,k) grid")

    for alpha in alphas:
        for k in ks:
            pbar.set_postfix(alpha=alpha, k=k)
            rs, Z, E = compute_Z_series(alpha, k, r_min, r_max)
            stats = summarize_Z(Z)
            rows.append({
                "alpha": alpha,
                "k": k,
                "r_min": r_min,
                "r_max": r_max,
                "n": stats["n"],
                "mean": stats["mean"],
                "var": stats["var"],
                "skew": stats["skew"],
                "kurt": stats["kurt"],
            })
            pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)
    return df

# ============================================================
# 7. 运行一轮实验（你可以调整参数）
# ============================================================
ALPHAS = [0.3, 0.5, 0.7, 1.0, 1.3]
KS     = [4, 8, 16, 32]

R_MIN = 2000
R_MAX = 8000   # 确认没问题后，你可以改大，比如 50000, 100000

df_result = experiment_grid(ALPHAS, KS, R_MIN, R_MAX)
print("===== 实验结果 df_result =====")
display(df_result)

# 保存一份 CSV（可下载）
df_result.to_csv("conical_results_basic.csv", index=False)
print("\n已保存到 conical_results_basic.csv")

# ============================================================
# 8. 拟合 Δ(α,k) 的线性 + 二次模型
# ============================================================
df_fit = df_result.copy()
df_fit["ln_k"] = np.log(df_fit["k"])
df_fit["Delta"] = 1 - df_fit["var"]

# ---------- 线性模型 Δ ≈ A + Bα + C ln k ----------
X_lin = df_fit[["alpha", "ln_k"]]
X_lin = sm.add_constant(X_lin)
y = df_fit["Delta"]
model_lin = sm.OLS(y, X_lin).fit()
print("\n===== 线性模型 Δ(α,k) 拟合结果 =====")
print(model_lin.summary())

# ---------- 二次模型 Δ ≈ A + Bα + C ln k + Dα² + E(lnk)² + F α ln k ----------
df_fit["alpha2"] = df_fit["alpha"]**2
df_fit["lnk2"]   = df_fit["ln_k"]**2
df_fit["alpha_lnk"] = df_fit["alpha"] * df_fit["ln_k"]

X_quad = df_fit[["alpha", "ln_k", "alpha2", "lnk2", "alpha_lnk"]]
X_quad = sm.add_constant(X_quad)
model_quad = sm.OLS(y, X_quad).fit()
print("\n===== 二次模型 Δ(α,k) 拟合结果 =====")
print(model_quad.summary())

coef = model_quad.params
A  = coef["const"]
B  = coef["alpha"]
C  = coef["ln_k"]
D  = coef["alpha2"]
E  = coef["lnk2"]
F  = coef["alpha_lnk"]

print("\n二次模型系数：")
print(f"A (const)      = {A:.6f}")
print(f"B (alpha)      = {B:.6f}")
print(f"C (ln_k)       = {C:.6f}")
print(f"D (alpha^2)    = {D:.6f}")
print(f"E (ln_k^2)     = {E:.6f}")
print(f"F (alpha*ln_k) = {F:.6f}")

# ---------- 计算二次模型残差矩阵 ----------
df_fit["Delta_pred_quad"] = (
    A
    + B*df_fit["alpha"]
    + C*df_fit["ln_k"]
    + D*df_fit["alpha2"]
    + E*df_fit["lnk2"]
    + F*df_fit["alpha_lnk"]
)
df_fit["Residual_quad"] = df_fit["Delta"] - df_fit["Delta_pred_quad"]

alphas = sorted(df_fit["alpha"].unique())
ks = sorted(df_fit["k"].unique())
res_mat = np.zeros((len(alphas), len(ks)))
for i, a in enumerate(alphas):
    for j, k in enumerate(ks):
        val = df_fit[(df_fit.alpha==a) & (df_fit.k==k)]["Residual_quad"]
        res_mat[i, j] = float(val.iloc[0])

print("\n===== 二次拟合残差矩阵 (Row=alpha, Col=k) =====")
print(res_mat)

plt.figure(figsize=(8,5))
plt.imshow(res_mat, cmap="coolwarm", origin="upper")
plt.colorbar(label="Residual (Δ - Δ_pred_quad)")
plt.xticks(range(len(ks)), ks)
plt.yticks(range(len(alphas)), alphas)
plt.xlabel("k")
plt.ylabel("alpha")
plt.title("Residual Heatmap after Quadratic Fit")
plt.tight_layout()
plt.show()
