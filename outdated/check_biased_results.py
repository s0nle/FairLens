"""Quick inspection of biased audit results."""
import json
import pandas as pd

with open("output/audit_results.json") as f:
    r = json.load(f)

gf   = r["group_fairness"]
gaps = gf["gaps"]

print("=== GROUP FAIRNESS GAPS ===")
for dim, g in gaps.items():
    best  = g["best_group"]
    worst = g["worst_group"]
    br    = g["best_rate"]
    wr    = g["worst_rate"]
    di    = g["di_ratio"]
    sev   = g["severity"]
    flag  = g["flagged"]
    print(f"  {dim}:")
    print(f"    best  = {best!r}  {br:.1%}")
    print(f"    worst = {worst!r}  {wr:.1%}")
    print(f"    DI ratio = {di:.3f}  flagged={flag}  severity={sev}")

print()
print("=== SWAP TESTS ===")
swaps = pd.DataFrame(r["swap_tests"]["summary"])
if not swaps.empty:
    print(swaps.to_string(index=False))
else:
    print("  (empty)")

print()
print("=== TOP PROXY FEATURES (race) ===")
race_proxy = r["proxy_rankings"]["race"][:8]
for p in race_proxy:
    feat = p["feature"]
    auc  = p["auc"]
    flag = p["flagged"]
    print(f"  {feat:<45}  AUC={auc:.3f}  flagged={flag}")

print()
print("=== VERDICTS ===")
# Neighborhood: check tract_minority_level gap
if "tract_minority_level" in gaps:
    tg = gaps["tract_minority_level"]
    print(f"  Neighborhood DI: {tg['di_ratio']:.3f}  severity={tg['severity']}  flagged={tg['flagged']}")
if "derived_race" in gaps:
    rg = gaps["derived_race"]
    print(f"  Race DI:         {rg['di_ratio']:.3f}  severity={rg['severity']}  flagged={rg['flagged']}")
if "derived_sex" in gaps:
    sg = gaps["derived_sex"]
    print(f"  Sex DI:          {sg['di_ratio']:.3f}  severity={sg['severity']}  flagged={sg['flagged']}")

# Flip rate for tract_high_minority
if not swaps.empty:
    th = swaps[swaps["varied_factor"] == "tract_high_minority"]
    if not th.empty:
        row = th.iloc[0]
        print(f"  tract_high_minority flip rate: {row['flip_rate']:.1%}  ({row['n_flipped']}/{row['n_pairs']})")
    for fac in ["sex_flip", "race_white", "race_black"]:
        rx = swaps[swaps["varied_factor"] == fac]
        if not rx.empty:
            rrow = rx.iloc[0]
            print(f"  {fac} flip rate: {rrow['flip_rate']:.1%}")
