"""
STEP 3: Analysis and charts
---------------------------
Objective 1: variability (mean, SD, CV) by month and season
Objective 2: trend (Mann-Kendall and Sen's slope) on seasonal and monthly dust
Objective 3: dust and visibility relationship (Spearman, dust classes)
Needs: pip install pandas numpy scipy matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr, kruskal

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
ALPHA = 0.05
MONTH_ORDER = [11, 12, 1, 2, 3]
MONTH_NAMES = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
DUST_LABEL = "Surface dust concentration (µg/m³)"

os.makedirs(FIG_DIR, exist_ok=True)


# ---------- Statistical tools ----------
def mann_kendall(x):
    """Mann-Kendall trend test (with tie correction). Returns S, Z, p."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    diff = np.sign(np.subtract.outer(x, x))          # x[j] - x[i]
    s = np.triu(diff.T, k=1).sum()                    # pairs with i < j
    _, counts = np.unique(x, return_counts=True)
    ties = (counts * (counts - 1) * (2 * counts + 5)).sum()
    var_s = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return s, z, p


def sens_slope(t, x):
    """Sen's slope (median of all pairwise slopes) and intercept."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    slopes = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if t[j] != t[i]:
                slopes.append((x[j] - x[i]) / (t[j] - t[i]))
    slope = np.median(slopes)
    intercept = np.median(x - slope * t)
    return slope, intercept


def trend_row(name, t, x):
    s, z, p = mann_kendall(x)
    slope, intercept = sens_slope(t, x)
    mean = np.mean(x)
    if p < ALPHA:
        verdict = "significant increase" if slope > 0 else "significant decrease"
    else:
        verdict = "no significant trend"
    return {
        "series": name, "n_seasons": len(x), "mean": round(mean, 2),
        "MK_S": int(s), "MK_Z": round(z, 3), "p_value": round(p, 4),
        "sen_slope_per_year": round(slope, 4),
        "change_pct_per_decade": round(slope * 10 / mean * 100, 2),
        "result": verdict,
    }, slope, intercept


# ---------- Load ----------
seasonal = pd.read_csv(os.path.join(OUT_DIR, "akure_dust_seasonal.csv"))
monthly = pd.read_csv(os.path.join(OUT_DIR, "akure_dust_monthly.csv"))
pairs_path = os.path.join(OUT_DIR, "akure_dust_visibility_pairs.csv")
pairs = pd.read_csv(pairs_path) if os.path.exists(pairs_path) else pd.DataFrame()

# ---------- Objective 1: variability ----------
rows = []
for m in MONTH_ORDER:
    v = monthly.loc[monthly["month"] == m, "dust_ugm3"]
    rows.append({"period": MONTH_NAMES[m], "n_seasons": len(v),
                 "mean": v.mean(), "SD": v.std(ddof=1),
                 "CV_pct": v.std(ddof=1) / v.mean() * 100,
                 "min": v.min(), "max": v.max()})
v = seasonal["dust_ugm3"]
rows.append({"period": "Whole season", "n_seasons": len(v), "mean": v.mean(),
             "SD": v.std(ddof=1), "CV_pct": v.std(ddof=1) / v.mean() * 100,
             "min": v.min(), "max": v.max()})
obj1 = pd.DataFrame(rows).round(2)
obj1.to_csv(os.path.join(OUT_DIR, "results_objective1_variability.csv"), index=False)
print("\nOBJECTIVE 1: variability of dust (ug/m3)")
print(obj1.to_string(index=False))

top = seasonal.sort_values("dust_ugm3", ascending=False)
print("\nDustiest 5 seasons:")
print(top[["label", "dust_ugm3"]].head(5).round(2).to_string(index=False))
print("Cleanest 5 seasons:")
print(top[["label", "dust_ugm3"]].tail(5).round(2).to_string(index=False))

# ---------- Objective 2: trends ----------
trend_rows = []
row, slope_s, int_s = trend_row("Whole season", seasonal["season"], seasonal["dust_ugm3"])
trend_rows.append(row)
for m in MONTH_ORDER:
    sub = monthly[monthly["month"] == m].sort_values("season")
    if len(sub) >= 8:
        r, _, _ = trend_row(MONTH_NAMES[m], sub["season"], sub["dust_ugm3"])
        trend_rows.append(r)
obj2 = pd.DataFrame(trend_rows)
obj2.to_csv(os.path.join(OUT_DIR, "results_objective2_trends.csv"), index=False)
print("\nOBJECTIVE 2: trends (Mann-Kendall and Sen's slope)")
print(obj2.to_string(index=False))

# ---------- Objective 3: dust and visibility ----------
if len(pairs) > 30:
    rho, p_rho = spearmanr(pairs["dust_ugm3"], pairs["vis_km"])
    pairs["dust_class"] = pd.qcut(pairs["dust_ugm3"], 3,
                                  labels=["Low dust", "Medium dust", "High dust"])
    cls = (pairs.groupby("dust_class", observed=True)
           .agg(days=("vis_km", "size"),
                dust_min=("dust_ugm3", "min"), dust_max=("dust_ugm3", "max"),
                mean_vis_km=("vis_km", "mean"),
                median_vis_km=("vis_km", "median")).round(2))
    groups = [g["vis_km"].values for _, g in pairs.groupby("dust_class", observed=True)]
    h, p_kw = kruskal(*groups)
    cls.to_csv(os.path.join(OUT_DIR, "results_objective3_dust_classes.csv"))
    pd.DataFrame([{"n_days": len(pairs), "spearman_rho": round(rho, 3),
                   "p_value": p_rho, "kruskal_p": p_kw}]).to_csv(
        os.path.join(OUT_DIR, "results_objective3_correlation.csv"), index=False)
    print("\nOBJECTIVE 3: dust and visibility")
    print(f"Days paired: {len(pairs)} | Spearman rho = {rho:.3f} (p = {p_rho:.3g})")
    print(cls.to_string())
    print(f"Kruskal-Wallis test across dust classes: p = {p_kw:.3g}")
else:
    print("\nOBJECTIVE 3 skipped: too few paired days.")

# ---------- Charts ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# 1. Seasonal mean dust with Sen's slope line
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(seasonal["season"], seasonal["dust_ugm3"], "o-", color="#b5651d",
        lw=1.5, ms=4, label="Seasonal mean")
xs = np.array([seasonal["season"].min(), seasonal["season"].max()])
ax.plot(xs, int_s + slope_s * xs, "--", color="black",
        label=f"Sen's slope: {slope_s:+.3f} µg/m³ per year "
              f"(p = {obj2.loc[0, 'p_value']})")
ax.set_xlabel("Harmattan season (starting year)")
ax.set_ylabel(DUST_LABEL)
ax.set_title("Harmattan dust at Akure, seasonal mean (Nov to Mar)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_seasonal_dust_trend.png"), dpi=200)
plt.close(fig)

# 2. Monthly climatology with SD bars
fig, ax = plt.subplots(figsize=(6.5, 4.5))
m_stats = obj1[obj1["period"].isin(MONTH_NAMES.values())]
ax.bar(m_stats["period"], m_stats["mean"], yerr=m_stats["SD"], capsize=4,
       color="#d9a066", edgecolor="black")
ax.set_ylabel(DUST_LABEL)
ax.set_title("Monthly mean dust (bars show ±1 SD across seasons)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_monthly_dust_climatology.png"), dpi=200)
plt.close(fig)

# 3 and 4. Dust and visibility
if len(pairs) > 30:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(pairs["dust_ugm3"], pairs["vis_km"], s=10, alpha=0.5, color="#8b4513")
    ax.set_xscale("log")
    ax.set_xlabel(DUST_LABEL + ", log scale")
    ax.set_ylabel("Daily mean visibility (km)")
    ax.set_title(f"Dust and visibility at Akure (Spearman rho = {rho:.2f}, "
                 f"n = {len(pairs)} days)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_dust_vs_visibility.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(groups, patch_artist=True, boxprops=dict(facecolor="#e8c9a0"))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Low dust", "Medium dust", "High dust"])
    ax.set_ylabel("Daily mean visibility (km)")
    ax.set_title("Visibility on low, medium and high dust days")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_visibility_by_dust_class.png"), dpi=200)
    plt.close(fig)

print(f"\nCharts saved in {FIG_DIR}")
