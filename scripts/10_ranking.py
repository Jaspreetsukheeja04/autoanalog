import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_csv  = snakemake.output[0]
out_plot = snakemake.output[1]
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

dock_csv   = snakemake.input[0]
admet2_csv = snakemake.input[1]
md_csv     = snakemake.input[2]
ml_pkl     = snakemake.input[3]
admet1_csv = snakemake.input[4]
ml_csv     = snakemake.input[5]

ref_ic50 = snakemake.config.get("reference_ic50_nm", 89)
ref_name = snakemake.config.get("reference_name", "Reference")
ref_src  = snakemake.config.get("reference_source", "")

print("=" * 45)
print("STEP 10: FINAL RANKING")
print("=" * 45)
print(f"Reference: {ref_name} IC50={ref_ic50}nM")
print(f"Source   : {ref_src}")
print("=" * 45)

dock_df   = pd.read_csv(dock_csv)
admet2_df = pd.read_csv(admet2_csv)
admet1_df = pd.read_csv(admet1_csv)
md_df     = pd.read_csv(md_csv)

print(f"Dock cols:   {list(dock_df.columns)}")
print(f"ADMET1 cols: {list(admet1_df.columns)}")
print(f"ADMET2 cols: {list(admet2_df.columns)}")

try:
    ml_scores = pickle.load(open(ml_pkl,"rb"))
    if isinstance(ml_scores, pd.DataFrame):
        ml_map   = dict(zip(
            ml_scores["name"],
            ml_scores["ml_score"]))
        ic50_map = dict(zip(
            ml_scores["name"],
            ml_scores.get(
                "predicted_ic50_nm",
                pd.Series([None]*len(ml_scores)))))
    else:
        ml_map, ic50_map = {}, {}
except Exception as e:
    print(f"ML pkl error: {e}")
    ml_map, ic50_map = {}, {}

try:
    ml_df = pd.read_csv(ml_csv)
    for _, r in ml_df.iterrows():
        ml_map[r["name"]]   = r.get("ml_score",0.5)
        ic50_map[r["name"]] = r.get(
            "predicted_ic50_nm", None)
    print(f"ML scores loaded: {len(ml_map)}")
except Exception as e:
    print(f"ML CSV error: {e}")

# ── MERGE ─────────────────────────────────
df = dock_df.copy()

for col in ["BBB","Bleeding_risk","CNS_ready",
            "QED","Overall_ADMET","hERG","PAINS"]:
    if col in admet1_df.columns:
        tmp = admet1_df[["name",col]].copy()
        tmp = tmp.rename(
            columns={col: col+"_a1"})
        df = df.merge(tmp, on="name", how="left")

for col in ["LigandEfficiency",
            "LipophilicEfficiency",
            "Binding_stability",
            "BBB_complex","Bleeding_risk"]:
    if col in admet2_df.columns:
        tmp = admet2_df[["name",col]].copy()
        tmp = tmp.rename(
            columns={col: col+"_a2"})
        df = df.merge(tmp, on="name", how="left")

md_keep = [c for c in
    ["name","energy_kcal","rmsd",
     "md_stable","md_time_ns"]
    if c in md_df.columns]
df = df.merge(md_df[md_keep],
    on="name", how="left")

df["ml_score"] = df["name"].map(
    ml_map).fillna(0.5)
df["predicted_ic50_nm"] = df["name"].map(
    ic50_map)

qed_col = "QED_a1" if "QED_a1" in df.columns \
    else "QED" if "QED" in df.columns else None
if qed_col:
    df["qed_score"] = pd.to_numeric(
        df[qed_col], errors="coerce").fillna(0.5)
else:
    df["qed_score"] = 0.5

print(f"QED loaded: {len(df)}")
print(f"Merged df cols: {list(df.columns)}")

# ── NORMALIZE DOCKING ─────────────────────
dmin = df["docking_score"].min()
dmax = df["docking_score"].max()
df["dock_norm"] = (
    dmin - df["docking_score"]) / (
    dmin - dmax + 1e-9)

# ── BBB SCORE ─────────────────────────────
bbb_col = None
for c in ["BBB_complex_a2","BBB_a1","BBB"]:
    if c in df.columns:
        bbb_col = c
        break

if bbb_col:
    df["bbb_score"] = df[bbb_col].apply(
        lambda x: 1.0 if x=="High"
        else 0.5 if x=="Medium" else 0.0)
else:
    df["bbb_score"] = 0.5
    print("WARNING: BBB column not found")

# ── BLEEDING SCORE ────────────────────────
bleed_col = None
for c in ["Bleeding_risk_a2",
          "Bleeding_risk_a1",
          "Bleeding_risk"]:
    if c in df.columns:
        bleed_col = c
        break

if bleed_col:
    df["bleed_score"] = df[bleed_col].apply(
        lambda x: 1.0 if x=="LOW"
        else 0.5 if x=="MEDIUM" else 0.0)
else:
    df["bleed_score"] = 0.5
    print("WARNING: Bleeding_risk not found")

# ── MD SCORE ──────────────────────────────
if "md_stable" in df.columns:
    df["md_score"] = df["md_stable"].apply(
        lambda x: 1.0 if x=="Yes"
        else 0.5 if pd.isna(x) else 0.0)
else:
    df["md_score"] = 0.5

# ── COMPOSITE ─────────────────────────────
df["composite"] = (
    0.25 * df["ml_score"]    +
    0.25 * df["dock_norm"]   +
    0.15 * df["bbb_score"]   +
    0.15 * df["bleed_score"] +
    0.10 * df["md_score"]    +
    0.10 * df["qed_score"]
)

if bleed_col:
    df.loc[df[bleed_col]=="HIGH",
           "composite"] -= 0.2

df["composite"] = df["composite"].clip(0,1)
df = df.sort_values(
    "composite", ascending=False)
df["rank"] = range(1, len(df)+1)

# ── REFERENCE COMPARISON ──────────────────
ref_row = df[df["name"].str.contains(
    "ref", case=False, na=False)]

ref_dock  = ref_row["docking_score"].values[0] \
    if len(ref_row) > 0 else None
ref_ml    = ref_row["ml_score"].values[0] \
    if len(ref_row) > 0 else None
ref_comp  = ref_row["composite"].values[0] \
    if len(ref_row) > 0 else None
ref_pred  = ref_row["predicted_ic50_nm"].values[0] \
    if len(ref_row) > 0 else None

df["better_than_ref_dock"] = df[
    "docking_score"].apply(
    lambda x: "Yes"
    if ref_dock and x < ref_dock else "No")

df["better_than_ref_ml"] = df[
    "ml_score"].apply(
    lambda x: "Yes"
    if ref_ml and x > ref_ml
    else "Equal" if ref_ml and x==ref_ml
    else "No")

df["better_than_ref_ic50"] = df[
    "predicted_ic50_nm"].apply(
    lambda x: "Yes"
    if x and ref_ic50>0 and x < ref_ic50
    else "No"
    if x and ref_ic50>0 else "Unknown")

df["reference_ic50_nm"] = ref_ic50
df["reference_name"]    = ref_name

df.to_csv(out_csv, index=False)

# ── PRINT RESULTS ─────────────────────────
print(f"\nReference {ref_name}:")
print(f"  Published IC50:  {ref_ic50} nM ({ref_src})")
print(f"  Docking score:   {ref_dock}")
print(f"  ML score:        {ref_ml}")
print(f"  Composite:       "
      f"{round(ref_comp,3) if ref_comp else 'N/A'}")
print(f"  Predicted IC50:  {ref_pred} nM")

print("\n=== FINAL RANKING ===")
for _, r in df.iterrows():
    ic50_str = ""
    if pd.notna(r.get("predicted_ic50_nm")):
        ic50_str = f" IC50~{r['predicted_ic50_nm']}nM"
    print(f"  Rank {int(r['rank'])}: {r['name']} "
          f"Score={round(r['composite'],3)} "
          f"ML={round(r['ml_score'],3)} "
          f"Dock={r['docking_score']}"
          f"{ic50_str}")

# ── PLOTS ─────────────────────────────────
fig, axes = plt.subplots(2,2,figsize=(14,10))
top = df.head(10)

axes[0,0].barh(top["name"],
    top["ml_score"], color="steelblue")
if ref_ml:
    axes[0,0].axvline(x=ref_ml,
        color="red", linestyle="--",
        label=f"{ref_name} ref")
axes[0,0].set_title("ML Activity Score")
axes[0,0].legend()

axes[0,1].barh(top["name"],
    top["docking_score"], color="coral")
if ref_dock:
    axes[0,1].axvline(x=ref_dock,
        color="red", linestyle="--",
        label=f"{ref_name} ref")
axes[0,1].set_title("Docking Score (kcal/mol)")
axes[0,1].legend()

ic50_vals = pd.to_numeric(
    top["predicted_ic50_nm"],
    errors="coerce").fillna(0)
axes[1,0].barh(top["name"],
    ic50_vals, color="purple")
axes[1,0].axvline(x=ref_ic50,
    color="red", linestyle="--",
    label=f"{ref_name} {ref_ic50}nM")
axes[1,0].set_title(
    "Predicted IC50 (nM) — lower=better")
axes[1,0].legend()

axes[1,1].barh(top["name"],
    top["composite"], color="green")
if ref_comp:
    axes[1,1].axvline(x=ref_comp,
        color="red", linestyle="--",
        label=f"{ref_name} ref")
axes[1,1].set_title("Final Composite Score")
axes[1,1].legend()

plt.suptitle(
    f"AutoAnalog v2.0 — Final Ranking\n"
    f"Reference: {ref_name} "
    f"IC50={ref_ic50}nM ({ref_src})",
    fontsize=11)
plt.tight_layout()
plt.savefig(out_plot, dpi=150,
    bbox_inches="tight")
plt.close()

print(f"\nPlot saved: {out_plot}")
print("Step 10 complete")
