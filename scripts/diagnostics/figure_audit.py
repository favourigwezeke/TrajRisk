#!/usr/bin/env python3
"""
TrajRisk Figure Asset Audit
============================
Run this BEFORE writing any plotting code.
Reads every result file in the repository and reports exactly what is available.
Produces no figures and no plots.

Usage:
    conda activate trajrisk
    cd /path/to/TrajRisk
    python scripts/figure_audit.py
"""

import os, json
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]

def hr(char="─", n=68): return char * n

print(hr("="))
print("TRAJRISK FIGURE ASSET AUDIT")
print(hr("="))

# =============================================================================
# SECTION A — CSV / JSON result files
# =============================================================================
print("\n" + hr())
print("SECTION A — Result files (CSV / JSON)")
print(hr())

search_dirs = ["results", "Data/processed"]
result_files = []

for sdir in search_dirs:
    d = BASE / sdir
    if not d.exists():
        print(f"  [missing] {sdir}/")
        continue
    for ext in ["*.csv", "*.json", "*.tsv"]:
        for f in sorted(d.glob(ext)):
            result_files.append(f)

if not result_files:
    print("  No CSV/JSON files found in results/ or Data/processed/")

for f in result_files:
    sz = f.stat().st_size
    mtime = f.stat().st_mtime
    import datetime
    mdate = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    print(f"\n  ── {f.name}")
    print(f"     Path:     {f.relative_to(BASE)}")
    print(f"     Size:     {sz:,} bytes  |  Modified: {mdate}")

    if f.suffix == ".csv":
        try:
            df = pd.read_csv(f, nrows=5)
            full = pd.read_csv(f)
            print(f"     Rows:     {len(full):,}")
            print(f"     Columns:  {full.columns.tolist()}")
            # Show dtypes summary
            num_cols = full.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                for c in num_cols[:8]:
                    print(f"               {c}: min={full[c].min():.4f}  max={full[c].max():.4f}")
        except Exception as e:
            print(f"     [Error reading: {e}]")

    elif f.suffix == ".json":
        try:
            with open(f) as fh:
                data = json.load(fh)
            def _keys(d, prefix="", depth=0):
                if depth > 3: return
                if isinstance(d, dict):
                    for k,v in d.items():
                        print(f"               {prefix}{k}: {str(v)[:80]}")
                        if isinstance(v, dict):
                            _keys(v, prefix+"  ", depth+1)
            _keys(data)
        except Exception as e:
            print(f"     [Error reading: {e}]")

# =============================================================================
# SECTION B — Existing figures (PNG / PDF / SVG)
# =============================================================================
print("\n" + hr())
print("SECTION B — Existing figures (PNG / PDF / SVG)")
print(hr())

fig_files = []
for ext in ["**/*.png", "**/*.pdf", "**/*.svg"]:
    for f in sorted(BASE.glob(ext)):
        if ".git" not in str(f) and "node_modules" not in str(f):
            fig_files.append(f)

if not fig_files:
    print("\n  No existing figure files found.")
else:
    for f in fig_files:
        sz = f.stat().st_size
        print(f"\n  ── {f.name}")
        print(f"     Path: {f.relative_to(BASE)}")
        print(f"     Size: {sz:,} bytes")
        if f.suffix == ".png":
            try:
                from PIL import Image
                img = Image.open(f)
                print(f"     Dims: {img.size[0]}×{img.size[1]} px  mode={img.mode}")
            except Exception:
                print("     Dims: (PIL not available)")

# =============================================================================
# SECTION C — Mouse validation assets
# =============================================================================
print("\n" + hr())
print("SECTION C — Mouse validation assets")
print(hr())

mouse_queries = {
    "Walesky internal validation": {
        "files": ["results/week9a_mouse_validation_table.csv",
                  "results/week9a_mouse_summary.csv"],
        "need_cols": ["test","trajrisk_knn_auroc","trajrisk_wself_auroc",
                      "seeder_auroc","b1_xgb_auroc"]
    },
    "Ben-Moshe results": {
        "files": ["results/week9a_mouse_validation_table.csv"],
        "filter_col": "test", "filter_val": "Ben-Moshe"
    },
    "Umbaugh 2h results": {
        "files": ["results/week9a_mouse_validation_table.csv"],
        "filter_col": "test", "filter_val": "Umbaugh 2h"
    },
    "Umbaugh 48h/72h results": {
        "files": ["results/week9a_mouse_validation_table.csv"],
        "filter_col": "test", "filter_val": "Umbaugh 48h"
    },
    "Cdkn1a statistics": {
        "files": ["results/week9a_mouse_p21_analysis.csv"]
    },
}

for label, info in mouse_queries.items():
    print(f"\n  {label}")
    found = False
    for fname in info["files"]:
        fp = BASE / fname
        if fp.exists():
            try:
                df = pd.read_csv(fp)
                print(f"    ✓ {fname}")
                print(f"      Columns: {df.columns.tolist()}")
                print(f"      Rows:    {len(df)}")
                if "filter_col" in info and info["filter_col"] in df.columns:
                    mask = df[info["filter_col"]].str.contains(
                        info.get("filter_val",""), case=False, na=False
                    )
                    print(f"      Matching rows: {mask.sum()}")
                found = True
            except Exception as e:
                print(f"    ✗ {fname} [Error: {e}]")
        else:
            print(f"    ✗ {fname} [NOT FOUND]")

# Check for Umbaugh 2h cell-level scores (for violin plot)
print("\n  Umbaugh 2h cell-level scores (for score distribution plot)")
# These would be in the week9a result — let's check if cell_risk_scores exist
cell_file = BASE / "results/week9a_cell_risk_scores.csv"
if cell_file.exists():
    df = pd.read_csv(cell_file)
    print(f"    ✓ {cell_file.name} — {len(df):,} rows | cols: {df.columns.tolist()}")
else:
    print(f"    ✗ results/week9a_cell_risk_scores.csv [NOT FOUND]")
    # Check the summary json for available data
    sj = BASE / "results/week9a_mouse_summary.json"
    if sj.exists():
        print(f"    ✓ results/week9a_mouse_summary.json (summary only — no cell-level scores)")

# Cdkn1a p21 file
p21 = BASE / "results/week9a_mouse_p21_analysis.csv"
if p21.exists():
    df_p21 = pd.read_csv(p21)
    print(f"\n  Cdkn1a file: {len(df_p21):,} rows | cols: {df_p21.columns.tolist()}")
else:
    print(f"\n  Cdkn1a file: results/week9a_mouse_p21_analysis.csv [NOT FOUND]")

# =============================================================================
# SECTION D — Human validation assets
# =============================================================================
print("\n" + hr())
print("SECTION D — Human validation assets")
print(hr())

human_queries = {
    "Donor-level scores (primary)":     "results/week9b_donor_primary.csv",
    "Donor-level scores (sensitivity)": "results/week9b_donor_sensitivity.csv",
    "Cell-level scores":                "results/week9b_cell_risk_scores.csv",
    "Gene signature":                   "results/week9b_gene_signature.csv",
    "Summary JSON":                     "results/week9b_summary.json",
}

for label, fname in human_queries.items():
    fp = BASE / fname
    print(f"\n  {label}")
    if fp.exists():
        if fp.suffix == ".csv":
            df = pd.read_csv(fp)
            print(f"    ✓ {fname}")
            print(f"      Rows:    {len(df):,}")
            print(f"      Columns: {df.columns.tolist()}")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for c in num_cols[:6]:
                print(f"      {c}: min={df[c].min():.4f}  max={df[c].max():.4f}")
        elif fp.suffix == ".json":
            with open(fp) as fh:
                d = json.load(fh)
            print(f"    ✓ {fname}")
            # Pull key metrics
            for key in ["primary","sensitivity","cs_trajrisk_val_auroc","integration_quality"]:
                if key in d:
                    print(f"      {key}: {str(d[key])[:120]}")
    else:
        print(f"    ✗ {fname} [NOT FOUND]")

# Check for neighbour composition data
print("\n  Neighbour composition (ALF vs Healthy by Walesky timepoint)")
cell_fp = BASE / "results/week9b_cell_risk_scores.csv"
if cell_fp.exists():
    df = pd.read_csv(cell_fp)
    comp_cols = [c for c in df.columns if c.startswith("pct_") or "neighbor" in c.lower()]
    if comp_cols:
        print(f"    ✓ Composition columns found: {comp_cols}")
    else:
        print(f"    ✗ No pct_/neighbor columns in week9b_cell_risk_scores.csv")
        print(f"      Available: {df.columns.tolist()}")
        print("      Composition data was logged but may not be saved to CSV")

# =============================================================================
# SECTION E — Panel availability check
# =============================================================================
print("\n" + hr())
print("SECTION E — Proposed panel availability")
print(hr())

panels = {}

# Mouse Figure
vt = BASE / "results/week9a_mouse_validation_table.csv"
panels["Mouse Panel A: Cross-dataset AUROC summary"] = vt.exists()

p21f = BASE / "results/week9a_mouse_p21_analysis.csv"
panels["Mouse Panel B: Umbaugh 2h score distributions"] = (
    "need cell-level Umbaugh 2h scores — check below"
)
# Check if week9a script saved cell-level scores
cell9a = BASE / "results/week9a_cell_risk_scores.csv"
if cell9a.exists():
    df_c = pd.read_csv(cell9a)
    has_umbaugh2h = "Umbaugh_2h" in df_c.get("dataset", pd.Series()).values \
                    if "dataset" in df_c.columns else False
    panels["Mouse Panel B: Umbaugh 2h score distributions"] = has_umbaugh2h
else:
    panels["Mouse Panel B: Umbaugh 2h score distributions"] = False

panels["Mouse Panel C: Cdkn1a vs TrajRisk scatter"] = p21f.exists()

# Neighbour composition — not in any saved file from week9b (was only logged)
# For mouse cross-dataset, there's no neighbour composition saved either
panels["Mouse Panel D: Neighbour composition (human)"] = False  # see note below

dp = BASE / "results/week9b_donor_primary.csv"
ds = BASE / "results/week9b_donor_sensitivity.csv"
panels["Human Panel A: Workflow schematic"] = "manual (no data needed)"
panels["Human Panel B: Donor-level ALF vs Healthy"] = dp.exists()
panels["Human Panel C: Sensitivity analysis"] = ds.exists()

# Convergence: need risk_2b and sig_net in donor primary
if dp.exists():
    ddf = pd.read_csv(dp)
    has_conv = "risk_2b" in ddf.columns and "sig_net" in ddf.columns
    panels["Human Panel D: Signature convergence scatter"] = has_conv
else:
    panels["Human Panel D: Signature convergence scatter"] = False

for panel, status in panels.items():
    icon = "✓" if status is True else ("⚠" if isinstance(status, str) else "✗")
    label = "Data available" if status is True \
            else (status if isinstance(status, str) else "Data MISSING")
    print(f"  {icon} {panel}")
    print(f"     → {label}")

# =============================================================================
# SECTION F — Recommended figure plan summary
# =============================================================================
print("\n" + hr())
print("SECTION F — Recommended figure plan (based on confirmed files)")
print(hr())

print("""
  MAIN FIGURE — Mouse Cross-Dataset Validation
    Panel A: Cross-dataset AUROC grouped bar chart
             Source: results/week9a_mouse_validation_table.csv
             Columns: test, trajrisk_knn_auroc, trajrisk_wself_auroc,
                      seeder_auroc, b1_xgb_auroc
    Panel B: Umbaugh 2h score distributions
             Source: VERIFY — need cell-level Umbaugh 2h scores
             (check whether week9a script saved cell-level outputs)
    Panel C: Cdkn1a vs TrajRisk scatter
             Source: results/week9a_mouse_p21_analysis.csv
             Columns: verify present
    Panel D: Neighbour composition
             Source: NOT SAVED to CSV during week9b run
             → Must re-run or add save step to extract composition

  MAIN FIGURE — Human Cross-Species Validation
    Panel A: Workflow schematic (manual — no data file needed)
    Panel B: Donor-level strip plot (ALF vs Healthy)
             Source: results/week9b_donor_primary.csv
             Columns: patient_id, condition, risk_2a, n_cells
    Panel C: Sensitivity analysis strip plot
             Source: results/week9b_donor_sensitivity.csv
             Columns: patient_id, condition, risk_2a
    Panel D: Convergence scatter (TrajRisk KNN vs net signature)
             Source: results/week9b_donor_primary.csv
             Columns: risk_2b, sig_net, condition — VERIFY

  SUPPLEMENTARY
    Supp Fig S1: AUPRC values
                 Source: week9a_mouse_validation_table.csv (check AUPRC columns)
    Supp Fig S2: Species integration UMAP
                 Source: Data/processed/adata_cross_species_combined.h5ad
                 (requires UMAP computation — X_scvi_cross key)
    Supp Fig S3: Gene signature analysis
                 Source: results/week9b_gene_signature.csv
    Supp Fig S4: Cell-level distributions
                 Source: results/week9b_cell_risk_scores.csv
""")

print(hr("="))
print("AUDIT COMPLETE")
print("Do NOT write plotting code until this audit is reviewed.")
print(hr("="))
