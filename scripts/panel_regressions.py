"""
面板数据回归分析：
1. 公司+年份双向FE（当期补贴）
2. 公司+年份双向FE（滞后一期补贴）
3. 工具变量2SLS（同行业同城市其他公司补贴均值为IV）
4. FA口径管理层权力中介效应（公司FE）
"""

import csv, math, random, sys
from collections import defaultdict

random.seed(42)

# ── 读取数据 ──────────────────────────────────────────────────────────────────
DATA = '/Users/lewis/bishe/results/regression_dataset.csv'

def to_float(s):
    try:
        return float(s)
    except:
        return None

rows = []
with open(DATA) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Total rows: {len(rows)}")

# ── 构建回归样本（主回归口径：有Power_FA的样本）────────────────────────────────
def make_sample(rows, require_lag=False):
    """构建回归所需样本，按公司-年份索引"""
    # 先建一个按(Symbol, Year)→row的字典，用于匹配滞后期
    by_sym_yr = {}
    for r in rows:
        sym = r['Symbol']
        try:
            yr = int(float(r['Year']))
        except:
            continue
        by_sym_yr[(sym, yr)] = r

    sample = []
    for r in rows:
        sym = r['Symbol']
        try:
            yr = int(float(r['Year']))
        except:
            continue

        overpay = to_float(r.get('Overpay'))
        lnsub   = to_float(r.get('lnSubsidy'))
        power   = to_float(r.get('Power_FA'))
        roa     = to_float(r.get('Roa'))
        lever   = to_float(r.get('Lever'))
        top1    = to_float(r.get('Top1'))
        zone    = to_float(r.get('Zone'))
        ind     = r.get('IndustrySector', '')
        city    = r.get('City', '')

        if any(v is None for v in [overpay, lnsub, power, roa, lever, top1, zone]):
            continue
        if not ind or not city:
            continue

        lnsub_lag = None
        if require_lag:
            prev = by_sym_yr.get((sym, yr - 1))
            if prev:
                lnsub_lag = to_float(prev.get('lnSubsidy'))
            if lnsub_lag is None:
                continue

        sample.append({
            'sym': sym, 'yr': yr,
            'overpay': overpay, 'lnsub': lnsub,
            'power': power, 'roa': roa,
            'lever': lever, 'top1': top1, 'zone': zone,
            'ind': ind, 'city': city,
            'lnsub_lag': lnsub_lag,
        })
    return sample

sample_main = make_sample(rows, require_lag=False)
sample_lag  = make_sample(rows, require_lag=True)
print(f"Main sample N={len(sample_main)}, Lag sample N={len(sample_lag)}")

# ── 构造工具变量：同行业同城市同年份其他公司lnSubsidy均值（排除本公司）──────────────
def build_iv(sample):
    """为每个obs计算leave-one-out peer mean IV"""
    # 按(ind, city, yr) → [(sym, lnsub)] 建索引
    grp = defaultdict(list)
    for obs in sample:
        grp[(obs['ind'], obs['city'], obs['yr'])].append((obs['sym'], obs['lnsub']))

    iv_vals = []
    for obs in sample:
        key = (obs['ind'], obs['city'], obs['yr'])
        peers = [(s, v) for s, v in grp[key] if s != obs['sym']]
        if len(peers) < 2:
            iv_vals.append(None)
        else:
            iv_vals.append(sum(v for _, v in peers) / len(peers))
    return iv_vals

iv_list = build_iv(sample_main)
# 只保留有IV的样本
sample_iv = [(obs, iv) for obs, iv in zip(sample_main, iv_list) if iv is not None]
print(f"IV sample N={len(sample_iv)}")

# ── 矩阵运算（纯Python+numpy）────────────────────────────────────────────────
import numpy as np

def within_demean(y_arr, x_mat, group_ids):
    """
    公司层面within去均值，返回去均值后的y和X（不含截距）。
    同时对年份FE做年份虚拟变量处理。
    """
    unique_groups = sorted(set(group_ids))
    grp_idx = {g: i for i, g in enumerate(unique_groups)}
    n = len(y_arr)
    # 按group去均值
    y_dm = np.array(y_arr, dtype=float)
    X_dm = np.array(x_mat, dtype=float)
    for g in unique_groups:
        mask = np.array([group_ids[i] == g for i in range(n)])
        y_dm[mask] -= y_dm[mask].mean()
        X_dm[mask] -= X_dm[mask].mean(axis=0)
    return y_dm, X_dm

def ols_cluster(y, X, cluster_ids):
    """
    OLS with cluster-robust standard errors.
    Returns: coef, se, t-stat, N, R2, R2_within
    X should NOT include intercept (already demeaned for within estimator).
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Cluster-robust variance (HC1 style within cluster)
    unique_clusters = sorted(set(cluster_ids))
    B = np.zeros((k, k))
    for c in unique_clusters:
        mask = np.array([cluster_ids[i] == c for i in range(n)])
        Xc = X[mask]
        ec = resid[mask]
        score = Xc.T @ ec
        B += np.outer(score, score)
    # sandwich
    G = len(unique_clusters)
    scale = G / (G - 1) * n / (n - k)
    V = XtX_inv @ B @ XtX_inv * scale
    se = np.sqrt(np.diag(V))
    t_stat = beta / se
    return beta, se, t_stat, n, r2

def year_dummies(yr_arr):
    """Create year dummy matrix (drop first year as reference)"""
    years = sorted(set(yr_arr))
    ref_yr = years[0]
    result = []
    for yr in yr_arr:
        row = [1.0 if yr == y else 0.0 for y in years[1:]]
        result.append(row)
    return np.array(result), years[1:]

# ── 回归1：行业+年份FE（基准，当期补贴）─────────────────────────────────────
# 已有主回归结果，这里计算公司FE的结果

def run_twoway_fe(sample, y_var='overpay', x_var='lnsub', controls=['roa','lever','top1','zone'],
                  extra_x=None, return_resid=False):
    """
    公司+年份双向FE回归。
    extra_x: list of column names for additional regressors.
    """
    sym_list = [obs['sym'] for obs in sample]
    yr_list  = [obs['yr']  for obs in sample]

    y = np.array([obs[y_var] for obs in sample])

    # 控制变量
    ctrl_mat = np.column_stack([[obs[c] for obs in sample] for c in controls])

    # 年份虚拟变量
    yr_dum, _ = year_dummies(yr_list)

    # 主解释变量
    main_x = np.array([[obs[x_var] for obs in sample]]).T

    if extra_x:
        extra_mat = np.column_stack([[obs[c] for obs in sample] for c in extra_x])
        X_full = np.hstack([main_x, extra_mat, ctrl_mat, yr_dum])
    else:
        X_full = np.hstack([main_x, ctrl_mat, yr_dum])

    # 公司within去均值
    y_dm, X_dm = within_demean(y, X_full, sym_list)

    beta, se, t_stat, n, r2 = ols_cluster(y_dm, X_dm, sym_list)

    if return_resid:
        resid = y_dm - X_dm @ beta
        return beta, se, t_stat, n, r2, resid
    return beta, se, t_stat, n, r2

print("\n=== 回归1: 公司+年份FE，当期补贴 ===")
b1, se1, t1, n1, r2_1 = run_twoway_fe(sample_main)
print(f"lnSubsidy: β={b1[0]:.4f}, SE={se1[0]:.4f}, t={t1[0]:.4f}, N={n1}, R2={r2_1:.4f}")

print("\n=== 回归2: 公司+年份FE，滞后一期补贴 ===")
# 修改sample_lag中的x_var映射
sample_lag2 = [{**obs, 'lnsub_lag_use': obs['lnsub_lag']} for obs in sample_lag]
# 用lnsub_lag替代lnsub
for obs in sample_lag2:
    obs['lnsub'] = obs['lnsub_lag']
b2, se2, t2, n2, r2_2 = run_twoway_fe(sample_lag2)
print(f"lnSubsidy_lag: β={b2[0]:.4f}, SE={se2[0]:.4f}, t={t2[0]:.4f}, N={n2}, R2={r2_2:.4f}")

# ── 回归3：2SLS工具变量 ───────────────────────────────────────────────────────
print("\n=== 回归3: 2SLS工具变量 ===")

obs_iv  = [x[0] for x in sample_iv]
iv_vals = np.array([x[1] for x in sample_iv])
sym_iv  = [obs['sym'] for obs in obs_iv]
yr_iv   = [obs['yr']  for obs in obs_iv]

y_iv    = np.array([obs['overpay'] for obs in obs_iv])
lnsub_iv = np.array([obs['lnsub'] for obs in obs_iv])
ctrl_iv  = np.column_stack([[obs[c] for obs in obs_iv] for c in ['roa','lever','top1','zone']])
yrdum_iv, _ = year_dummies(yr_iv)

# 第一阶段：lnSubsidy ~ IV + controls（公司FE）
X1_full = np.hstack([iv_vals.reshape(-1,1), ctrl_iv, yrdum_iv])
y1_dm, X1_dm = within_demean(lnsub_iv, X1_full, sym_iv)
b_1st, se_1st, t_1st, n_1st, r2_1st = ols_cluster(y1_dm, X1_dm, sym_iv)
print(f"  第一阶段 IV系数: β={b_1st[0]:.4f}, t={t_1st[0]:.4f}")

# F统计量（IV第一阶段）：简化为t^2
f_stat = t_1st[0]**2
print(f"  第一阶段 F(IV): {f_stat:.2f}")

# 第一阶段拟合值（in-sample, company-demeaned space）
lnsub_hat_dm = X1_dm @ b_1st  # demeaned fitted values

# 第二阶段：Overpay ~ lnSubsidy_hat + controls（公司FE）
X2_full = np.hstack([lnsub_iv.reshape(-1,1), ctrl_iv, yrdum_iv])
y2_dm, X2_base = within_demean(y_iv, X2_full, sym_iv)
# 替换第一列为第一阶段拟合值（已经在demeaned space）
X2_2sls = X2_base.copy()
X2_2sls[:, 0] = lnsub_hat_dm

b_2nd, se_2nd, t_2nd, n_2nd, r2_2nd = ols_cluster(y2_dm, X2_2sls, sym_iv)
print(f"  第二阶段 lnSubsidy: β={b_2nd[0]:.4f}, SE={se_2nd[0]:.4f}, t={t_2nd[0]:.4f}, N={n_2nd}")

# ── 回归4：FA口径中介效应（公司FE）──────────────────────────────────────────
print("\n=== 回归4: FA口径中介效应（公司FE）===")

# 总效应 c：Overpay ~ lnSubsidy + controls（公司FE）
bc, sec, tc, nc, r2c = run_twoway_fe(sample_main)
c_total = bc[0]
print(f"  总效应 c: β={c_total:.4f}, t={tc[0]:.4f}")

# 路径a：Power_FA ~ lnSubsidy + controls（公司FE）
ba, sea, ta, na, r2a = run_twoway_fe(sample_main, y_var='power', x_var='lnsub')
a_path = ba[0]
print(f"  路径a (lnsub→Power): β={a_path:.4f}, t={ta[0]:.4f}")

# 直接效应 c' + 路径b：Overpay ~ lnSubsidy + Power_FA + controls（公司FE）
bcp, secp, tcp, ncp, r2cp = run_twoway_fe(sample_main, extra_x=['power'])
c_prime = bcp[0]
b_path  = bcp[1]
print(f"  直接效应 c': β={c_prime:.4f}, t={tcp[0]:.4f}")
print(f"  路径b (Power→Overpay): β={b_path:.4f}, t={tcp[1]:.4f}")

indirect = a_path * b_path
mediation_pct = indirect / c_total * 100 if c_total != 0 else 0
print(f"  间接效应 a*b: {indirect:.6f}")
print(f"  中介效应占比: {mediation_pct:.2f}%")

# Bootstrap置信区间（公司层面cluster bootstrap，300次）
print("  Bootstrap CI计算中（300次）...")
syms_main = list(set(obs['sym'] for obs in sample_main))
n_syms = len(syms_main)
sym_to_obs = defaultdict(list)
for i, obs in enumerate(sample_main):
    sym_to_obs[obs['sym']].append(i)

boot_indirect = []
for _ in range(300):
    # 有放回抽取公司
    boot_syms = random.choices(syms_main, k=n_syms)
    boot_idx = []
    for s in boot_syms:
        boot_idx.extend(sym_to_obs[s])
    boot_sample = [sample_main[i] for i in boot_idx]

    try:
        ba_b, _, _, _, _ = run_twoway_fe(boot_sample, y_var='power', x_var='lnsub')
        bcp_b, _, _, _, _ = run_twoway_fe(boot_sample, extra_x=['power'])
        boot_indirect.append(ba_b[0] * bcp_b[1])
    except:
        continue

boot_indirect.sort()
ci_lo = boot_indirect[int(len(boot_indirect) * 0.025)]
ci_hi = boot_indirect[int(len(boot_indirect) * 0.975)]
print(f"  Bootstrap 95%CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"  CI包含0: {'是' if ci_lo < 0 < ci_hi else '否'}")

# ── 汇总输出 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("汇总结果（用于论文表格）")
print("="*60)
print(f"表4-8a 行1（基准行业FE）: β=0.0582, t=14.3246, N=44831, R2=0.0372")
print(f"表4-8a 行2（公司FE当期）: β={b1[0]:.4f}, t={t1[0]:.4f}, N={n1}, R2={r2_1:.4f}")
print(f"表4-8a 行3（公司FE滞后）: β={b2[0]:.4f}, t={t2[0]:.4f}, N={n2}, R2={r2_2:.4f}")
print(f"表4-8b 第一阶段F统计量: {f_stat:.2f}")
print(f"表4-8b 第一阶段IV系数: {b_1st[0]:.4f}, t={t_1st[0]:.4f}")
print(f"表4-8b 第二阶段lnSubsidy: β={b_2nd[0]:.4f}, SE={se_2nd[0]:.4f}, t={t_2nd[0]:.4f}, N={n_2nd}")
print(f"表4-8c 路径a: β={a_path:.4f}, SE={sea[0]:.4f}")
print(f"表4-8c 路径b: β={b_path:.4f}, SE={secp[1]:.4f}")
print(f"表4-8c 间接效应: {indirect:.5f}")
print(f"表4-8c Bootstrap 95%CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"表4-8c 直接效应c': β={c_prime:.4f}, SE={secp[0]:.4f}")
print(f"表4-8c 总效应c: β={c_total:.4f}, SE={sec[0]:.4f}")
print(f"表4-8c 中介效应占比: {mediation_pct:.2f}%")
