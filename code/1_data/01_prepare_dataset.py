"""
Data Inspection and Validation

"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

# ─────────────────────────────────────────────────────────────
# CONFIG — DO NOT CHANGE
# ─────────────────────────────────────────────────────────────
INPUT_FILE  = "final_jit_with_activity_new_clean.csv"   # your definitive file
OUTPUT_CSV  = "malaria_dataset_clean.csv"          # consumed by steps 1-4
REPORT_FILE = "results/diagnostics/dataset_report.txt"
FIGURE_DIR  = "results/diagnostics/dataset"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD RAW FILE
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("DATA INSPECTION & VALIDATION")
print("=" * 60)
print(f"\nInput  : {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nERROR: '{INPUT_FILE}' not found in current directory.\n"
        f"Run this from the top folder of the repository, "
        f"where the dataset file lives."
    )

# Try multiple encodings — handles UTF-8, Windows-1252, Latin-1 etc.
for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
    try:
        df = pd.read_csv(INPUT_FILE, encoding=enc)
        print(f"Encoding used  : {enc}")
        break
    except (UnicodeDecodeError, Exception):
        continue
else:
    raise ValueError(
        f"Could not read {INPUT_FILE} with any known encoding.\n"
        f"Try opening in WPS Office and re-saving as CSV UTF-8."
    )
print(f"Shape  : {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\ndtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────────────────────
# 2. STANDARDIZE COLUMN NAMES
#  Column A = SMILES, Column B = Activity
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("COLUMN DETECTION")
print("─" * 40)

df.columns = [c.strip() for c in df.columns]

# Detect SMILES column
smiles_col = None
for col in df.columns:
    if col.upper() in ['SMILES','SMILE','CANONICAL_SMILES',
                        'MOL','STRUCTURE','SMI']:
        smiles_col = col
        break
if smiles_col is None:
    smiles_col = df.columns[0]
print(f"SMILES column   : '{smiles_col}'")

# Detect Activity column
activity_col = None
for col in df.columns:
    if col.upper() in ['ACTIVITY','LABEL','ACTIVE','CLASS',
                        'Y','TARGET','OUTCOME']:
        activity_col = col
        break
if activity_col is None:
    activity_col = df.columns[1]
print(f"Activity column : '{activity_col}'")

df = df[[smiles_col, activity_col]].copy()
df.columns = ['smiles', 'activity']

# ─────────────────────────────────────────────────────────────
# 3. ACTIVITY LABEL INSPECTION
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("ACTIVITY LABEL INSPECTION")
print("─" * 40)

print(f"Unique values : {sorted(df['activity'].dropna().unique())}")
print(f"Value counts  :\n{df['activity'].value_counts().sort_index()}")

# Confirm binary
if not set(df['activity'].dropna().unique()).issubset({0,1,0.0,1.0}):
    print("WARNING: Non-binary values found — check your labels!")
else:
    df['activity'] = df['activity'].astype(int)
    print("Binary 0/1 confirmed ✓")

# Raw class counts
n_raw        = len(df)
n_active_raw = int((df['activity'] == 1).sum())
n_inactive_raw = int((df['activity'] == 0).sum())
print(f"\nRaw counts:")
print(f"  Total      : {n_raw:,}")
print(f"  Active (1) : {n_active_raw:,}  ({n_active_raw/n_raw*100:.2f}%)")
print(f"  Inactive(0): {n_inactive_raw:,}  ({n_inactive_raw/n_raw*100:.2f}%)")
print(f"  Imbalance  : 1 : {n_inactive_raw/n_active_raw:.1f}")

# ─────────────────────────────────────────────────────────────
# 4. SMILES VALIDATION & CANONICALIZATION
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("SMILES VALIDATION")
print("─" * 40)

def validate_and_canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

print("Canonicalizing SMILES...")
df['canonical_smiles'] = df['smiles'].apply(validate_and_canonicalize)

n_invalid = df['canonical_smiles'].isna().sum()
print(f"Invalid SMILES : {n_invalid:,}")
print(f"Valid SMILES   : {len(df) - n_invalid:,}")

if n_invalid > 0:
    print("Invalid examples:")
    for s in df[df['canonical_smiles'].isna()]['smiles'].head(3):
        print(f"  {s}")

df = df.dropna(subset=['canonical_smiles']).copy()

# ─────────────────────────────────────────────────────────────
# 5. DUPLICATE REMOVAL
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("DUPLICATE REMOVAL")
print("─" * 40)

n_before = len(df)

# Check conflicting labels
dup_mask  = df.duplicated('canonical_smiles', keep=False)
n_dup_total = dup_mask.sum()
if n_dup_total > 0:
    conflicts = (df[dup_mask].groupby('canonical_smiles')
                              ['activity'].nunique() > 1).sum()
    print(f"Duplicate SMILES     : {n_dup_total:,}")
    print(f"Conflicting labels   : {conflicts:,}")
    print(f"Strategy             : keep activity=1 for conflicts")

df = df.sort_values('activity', ascending=False)
df = df.drop_duplicates(subset='canonical_smiles', keep='first')
df = df.reset_index(drop=True)

n_after  = len(df)
n_removed_dup = n_before - n_after
print(f"Before           : {n_before:,}")
print(f"After            : {n_after:,}")
print(f"Removed          : {n_removed_dup:,}")

# ─────────────────────────────────────────────────────────────
# 6. PHYSICOCHEMICAL PROPERTIES
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("PHYSICOCHEMICAL PROPERTIES")
print("─" * 40)

def compute_props(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [np.nan] * 6
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
        ]
    except Exception:
        return [np.nan] * 6

print("Computing MW, logP, TPSA, HBD, HBA, RotBonds...")
props_list = df['canonical_smiles'].apply(compute_props).tolist()
props_df = pd.DataFrame(
    props_list,
    columns=['MW','logP','TPSA','HBD','HBA','RotBonds'],
    index=df.index
)
df = pd.concat([df, props_df], axis=1)

print("\nProperty means by class:")
print(df.groupby('activity')[['MW','logP','TPSA','HBD','HBA']
    ].mean().round(2).to_string())

# ─────────────────────────────────────────────────────────────
# 7. MURCKO SCAFFOLDS
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("MURCKO SCAFFOLDS")
print("─" * 40)

def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 'no_scaffold'
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc, canonical=True)
    except Exception:
        return 'no_scaffold'

print("Computing scaffolds...")
df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)

n_scaffolds = df['scaffold'].nunique()
print(f"Unique scaffolds : {n_scaffolds:,}")

# Activity cliff detection
sc_grp = df.groupby('scaffold')['activity'].agg(
    n_active='sum', n_total='count'
)
sc_grp['has_cliff'] = (
    (sc_grp['n_active'] > 0) &
    (sc_grp['n_total'] - sc_grp['n_active'] > 0)
)
n_cliff_sc = int(sc_grp['has_cliff'].sum())
n_cliff_cpd = df[
    df['scaffold'].isin(sc_grp[sc_grp['has_cliff']].index)
].shape[0]
print(f"Activity cliff scaffolds  : {n_cliff_sc:,}")
print(f"Compounds in cliff families: {n_cliff_cpd:,}")

# ─────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("FINAL SUMMARY")
print("─" * 40)

n_final      = len(df)
n_active_f   = int((df['activity'] == 1).sum())
n_inactive_f = int((df['activity'] == 0).sum())
hit_rate_f   = n_active_f / n_final * 100

print(f"Total compounds    : {n_final:,}")
print(f"Active (1)         : {n_active_f:,}  ({hit_rate_f:.2f}%)")
print(f"Inactive (0)       : {n_inactive_f:,}  ({100-hit_rate_f:.2f}%)")
print(f"Imbalance ratio    : 1 : {n_inactive_f/n_active_f:.1f}")
print(f"Unique scaffolds   : {n_scaffolds:,}")
print(f"Activity cliffs    : {n_cliff_sc:,} scaffold families")
print(f"\nTotal removed:")
print(f"  Invalid SMILES   : {n_invalid:,}")
print(f"  Duplicates       : {n_removed_dup:,}")
print(f"  Net removed      : {n_raw - n_final:,}")

# ─────────────────────────────────────────────────────────────
# 9. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
# Minimal CSV for model training (steps 1-4 read this)
df_out = df[['canonical_smiles','activity','scaffold']].copy()
df_out.columns = ['smiles','activity','scaffold']
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nClean dataset saved : {OUTPUT_CSV}  ({df_out.shape})")

# Full CSV with properties (for figures)
df_full = df[['canonical_smiles','activity','scaffold',
              'MW','logP','TPSA','HBD','HBA','RotBonds']].copy()
df_full.columns = ['smiles','activity','scaffold',
                   'MW','logP','TPSA','HBD','HBA','RotBonds']
df_full.to_csv("malaria_dataset_with_props.csv", index=False)
print(f"Props dataset saved : malaria_dataset_with_props.csv")

# ─────────────────────────────────────────────────────────────
# 10. FIGURES
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("GENERATING FIGURES")
print("─" * 40)

# Figure 1: Class distribution bar
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(
    ['Active\n(Antimalarial)', 'Inactive'],
    [n_active_f, n_inactive_f],
    color=['#C62828','#1565C0'],
    width=0.5, edgecolor='white', linewidth=1.5
)
for bar, val in zip(bars, [n_active_f, n_inactive_f]):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + n_final * 0.005,
        f'{val:,}\n({val/n_final*100:.1f}%)',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold'
    )
ax.set_ylabel('Number of Compounds', fontsize=12)
ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(n_active_f, n_inactive_f) * 1.22)
plt.tight_layout()
fig.savefig(f"{FIGURE_DIR}/class_distribution.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURE_DIR}/class_distribution.png")

# Figure 2: Physicochemical distributions
props_plot = ['MW','logP','TPSA','HBD','HBA','RotBonds']
fig, axes  = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Physicochemical Property Distributions — Active vs Inactive',
             fontsize=13, fontweight='bold')

color_map  = {1:'#C62828', 0:'#1565C0'}
label_map  = {1:f'Active (n={n_active_f:,})',
              0:f'Inactive (n={n_inactive_f:,})'}

for ax, prop in zip(axes.flatten(), props_plot):
    for cls in [0, 1]:
        vals = df[df['activity'] == cls][prop].dropna()
        ax.hist(vals, bins=40, alpha=0.65,
                color=color_map[cls], label=label_map[cls],
                density=True, edgecolor='none')
    ax.set_xlabel(prop, fontsize=11)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(prop, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(f"{FIGURE_DIR}/physicochemical_distributions.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURE_DIR}/physicochemical_distributions.png")

# ─────────────────────────────────────────────────────────────
# 11. TEXT REPORT
# ─────────────────────────────────────────────────────────────
report = f"""
ANTIMALARIAL DATASET REPORT
======================================
Input file   : {INPUT_FILE}
Output file  : {OUTPUT_CSV}

RAW DATA
--------
Total rows (raw)   : {n_raw:,}
Invalid SMILES     : {n_invalid:,}
Duplicates removed : {n_removed_dup:,}

FINAL CLEAN DATASET
-------------------
Total compounds    : {n_final:,}
Active (1)         : {n_active_f:,}  ({hit_rate_f:.2f}%)
Inactive (0)       : {n_inactive_f:,}  ({100-hit_rate_f:.2f}%)
Imbalance ratio    : 1 : {n_inactive_f/n_active_f:.1f}

STRUCTURAL DIVERSITY
--------------------
Unique Murcko scaffolds    : {n_scaffolds:,}
Activity cliff scaffolds   : {n_cliff_sc:,}
Cliff-involved compounds   : {n_cliff_cpd:,}

PHYSICOCHEMICAL PROPERTIES (Active | Inactive)
----------------------------------------------
MW      : {df[df['activity']==1]['MW'].mean():.1f} ± {df[df['activity']==1]['MW'].std():.1f}  |  {df[df['activity']==0]['MW'].mean():.1f} ± {df[df['activity']==0]['MW'].std():.1f}
logP    : {df[df['activity']==1]['logP'].mean():.2f} ± {df[df['activity']==1]['logP'].std():.2f}  |  {df[df['activity']==0]['logP'].mean():.2f} ± {df[df['activity']==0]['logP'].std():.2f}
TPSA    : {df[df['activity']==1]['TPSA'].mean():.1f} ± {df[df['activity']==1]['TPSA'].std():.1f}  |  {df[df['activity']==0]['TPSA'].mean():.1f} ± {df[df['activity']==0]['TPSA'].std():.1f}
HBD     : {df[df['activity']==1]['HBD'].mean():.2f} ± {df[df['activity']==1]['HBD'].std():.2f}  |  {df[df['activity']==0]['HBD'].mean():.2f} ± {df[df['activity']==0]['HBD'].std():.2f}
HBA     : {df[df['activity']==1]['HBA'].mean():.2f} ± {df[df['activity']==1]['HBA'].std():.2f}  |  {df[df['activity']==0]['HBA'].mean():.2f} ± {df[df['activity']==0]['HBA'].std():.2f}

NOTES FOR PAPER METHODS
-----------------------
- Primary metric    : AUPRC (justified by {n_inactive_f/n_active_f:.0f}:1 class imbalance)
- Secondary metrics : AUROC, BEDROC (alpha=20), F1
- CV strategy       : 8-fold (random, scaffold, scaffold-cluster)
- Duplicate strategy: Canonical SMILES dedup; conflicting labels → keep active
- Activity label    : Binary (0=inactive, 1=active vs P. falciparum)
"""

with open(REPORT_FILE, 'w') as f:
    f.write(report)
print(f"\nReport saved    : {REPORT_FILE}")

print("\n" + "=" * 60)
print("COMPLETE")
print(f"  Total     : {n_final:,}")
print(f"  Actives   : {n_active_f:,} ({hit_rate_f:.2f}%)")
print(f"  Imbalance : 1 : {n_inactive_f/n_active_f:.1f}")
print(f"  Scaffolds : {n_scaffolds:,}")
print("=" * 60)
print("\nNEXT: python code/1_data/02_make_splits.py")
