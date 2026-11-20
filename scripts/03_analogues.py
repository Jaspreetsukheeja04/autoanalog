import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import (AllChem, Descriptors, QED,
    BRICS, Draw, rdMolDescriptors,
    rdFingerprintGenerator)
from rdkit.Chem.rdchem import RWMol
import pandas as pd
import numpy as np
import os

out_csv = snakemake.output[0]
out_png = snakemake.output[1]
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

known_inh   = snakemake.config.get("known_inhibitor","")
ref_name    = snakemake.config.get("reference_name","Reference")
ref_ic50    = snakemake.config.get("reference_ic50_nm", 0)
max_analogs = snakemake.config.get("max_permutation_analogues", 100)

mfpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048)

print("=" * 45)
print("STEP 3: ANALOGUE DESIGN")
print("=" * 45)
print("Basis: Systematic R-group enumeration")
print("       Topliss scheme + Hansch-Craig")
print("       + Bioisostere principles")

# ── 15 SUBSTITUENTS (medicinal chem basis) ───
# Halogens: Topliss scheme
# Alkyl: Hansch-Craig logP optimization
# Polar: Bioisosteric replacement
# Special: Fluorine medicinal chemistry

SUB_SMILES = [
    ("F",        "Fluorine — metabolic stability"),
    ("Cl",       "Chlorine — membrane permeability"),
    ("Br",       "Bromine — larger pocket fill"),
    ("C",        "Methyl — blocks metabolism"),
    ("CC",       "Ethyl — lipophilicity increase"),
    ("CCC",      "Propyl — lipophilicity increase"),
    ("OC",       "Methoxy — HBA + solubility"),
    ("OCC",      "Ethoxy — HBA + solubility"),
    ("N",        "Amino — HBD+HBA addition"),
    ("NC",       "Methylamino — reduced basicity"),
    ("C(F)(F)F", "CF3 — metabolic stability"),
    ("C#N",      "Cyano — HBA + BBB penetration"),
    ("O",        "Hydroxy — HBD + solubility"),
    ("S",        "Thiol — unique binding"),
    ("C1CC1",    "Cyclopropyl — conformational constraint"),
]

RING_SWAPS = {
    "pyridine_to_thiazole":
        ("c1ccncc1","c1cscn1"),
    "pyridine_to_pyrimidine":
        ("c1ccncc1","c1cncnc1"),
    "benzene_to_pyridine":
        ("c1ccccc1","c1ccncc1"),
    "benzene_to_thiophene":
        ("c1ccccc1","c1ccsc1"),
}

BIOISOSTERES = {
    "sulfonamide_to_urea":
        ("NS(=O)(=O)","NC(=O)N"),
    "sulfonamide_to_amide":
        ("NS(=O)(=O)","NC(=O)"),
    "sulfonamide_to_carbamate":
        ("NS(=O)(=O)","NC(=O)O"),
}

def sanitize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def get_canonical_key(mol):
    """Use InChIKey to filter isomers"""
    try:
        from rdkit.Chem.inchi import MolToInchiKey
        key = MolToInchiKey(mol)
        if key:
            return key
    except Exception:
        pass
    return Chem.MolToSmiles(
        mol, canonical=True,
        isomericSmiles=True)

def lipinski_pass(mol):
    if mol is None:
        return False
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    return (mw<=500 and logp<=5
            and hbd<=5 and hba<=10)

def get_tanimoto(mol, ref_mol):
    try:
        fp1 = mfpgen.GetFingerprint(mol)
        fp2 = mfpgen.GetFingerprint(ref_mol)
        return round(
            DataStructs.TanimotoSimilarity(
                fp1, fp2), 3)
    except Exception:
        return 0.0

def get_descriptors(mol, name,
                    modification, ref_mol):
    try:
        mw   = round(Descriptors.MolWt(mol), 1)
        logp = round(Descriptors.MolLogP(mol), 2)
        qed  = round(QED.qed(mol), 3)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = round(Descriptors.TPSA(mol), 1)
        rot  = rdMolDescriptors\
            .CalcNumRotatableBonds(mol)
        lip  = "PASS" if lipinski_pass(mol) \
               else "FAIL"
        smi  = Chem.MolToSmiles(mol)
        tan  = get_tanimoto(mol, ref_mol)

        pred_ic50 = None
        vs_ref    = "Unknown"
        if ref_ic50 > 0 and tan > 0:
            pred_ic50 = round(
                ref_ic50 / max(tan, 0.01), 1)
            vs_ref = "Better" \
                if pred_ic50 < ref_ic50 \
                else "Worse"

        return {
            "name":              name,
            "smiles":            smi,
            "modification":      modification,
            "MW":                mw,
            "logP":              logp,
            "QED":               qed,
            "HBD":               hbd,
            "HBA":               hba,
            "TPSA":              tpsa,
            "RotBonds":          rot,
            "Lipinski":          lip,
            "Tanimoto":          tan,
            "predicted_ic50_nm": pred_ic50,
            "vs_ref_ic50":       vs_ref,
            "ref_ic50_nm":       ref_ic50,
        }
    except Exception as e:
        print(f"  Descriptor error {name}: {e}")
        return None

def get_aromatic_positions(mol):
    positions = []
    for atom in mol.GetAtoms():
        if (atom.GetIsAromatic() and
                atom.GetSymbol() == "C" and
                atom.GetTotalNumHs() > 0):
            positions.append(atom.GetIdx())
    return positions

def add_sub_to_position(mol, atom_idx,
                         sub_smi):
    try:
        sub = Chem.MolFromSmiles(sub_smi)
        if sub is None:
            return None
        atom = mol.GetAtomWithIdx(atom_idx)
        if not atom.GetIsAromatic():
            return None
        if atom.GetTotalNumHs() == 0:
            return None
        combined = Chem.RWMol(
            Chem.CombineMols(mol, sub))
        new_idx = mol.GetNumAtoms()
        combined.AddBond(
            atom_idx, new_idx,
            Chem.rdchem.BondType.SINGLE)
        combined.GetAtomWithIdx(
            atom_idx).SetNumExplicitHs(
            max(0,
                atom.GetTotalNumHs() - 1))
        Chem.SanitizeMol(combined)
        return combined.GetMol()
    except Exception:
        return None

# ── REFERENCE ─────────────────────────────
print(f"\nReference: {ref_name} IC50={ref_ic50}nM")

if not known_inh:
    raise ValueError(
        "known_inhibitor blank in config")

ref_mol = sanitize(known_inh)
if ref_mol is None:
    raise ValueError("Invalid reference SMILES")

ref_smi = Chem.MolToSmiles(ref_mol)
print(f"Reference SMILES valid")

analogues      = []
existing_names = set()
existing_keys  = set()  # InChIKey based

ref_desc = get_descriptors(
    ref_mol, f"{ref_name}_ref",
    "Reference compound", ref_mol)
if ref_desc:
    analogues.append(ref_desc)
    existing_names.add(f"{ref_name}_ref")
    existing_keys.add(
        get_canonical_key(ref_mol))
    print(f"  Added reference: MW={ref_desc['MW']}")

# ── RING SWAPS ────────────────────────────
print("\nGenerating ring swap analogues...")
ring_count = 0
for swap_name, (old_pat, new_pat) in \
        RING_SWAPS.items():
    try:
        old_mol = Chem.MolFromSmarts(old_pat)
        if old_mol and ref_mol\
                .HasSubstructMatch(old_mol):
            new_smi = ref_smi.replace(
                old_pat, new_pat)
            new_mol = sanitize(new_smi)
            if new_mol and lipinski_pass(new_mol):
                key = get_canonical_key(new_mol)
                if key in existing_keys:
                    continue
                short = swap_name\
                    .split("_to_")[1].capitalize()
                desc = get_descriptors(
                    new_mol, short,
                    swap_name.replace("_"," "),
                    ref_mol)
                if desc:
                    analogues.append(desc)
                    existing_names.add(short)
                    existing_keys.add(key)
                    ring_count += 1
                    print(f"  + {short}: "
                          f"MW={desc['MW']}")
    except Exception as e:
        print(f"  Ring swap error: {e}")

print(f"  Ring analogues: {ring_count}")

# ── BIOISOSTERES ──────────────────────────
print("\nGenerating bioisostere analogues...")
bio_count = 0
for bio_name, (old_pat, new_pat) in \
        BIOISOSTERES.items():
    try:
        if old_pat in ref_smi:
            new_smi = ref_smi.replace(
                old_pat, new_pat, 1)
            new_mol = sanitize(new_smi)
            if new_mol and lipinski_pass(new_mol):
                key = get_canonical_key(new_mol)
                if key in existing_keys:
                    continue
                short = bio_name\
                    .split("_to_")[1].capitalize()
                desc = get_descriptors(
                    new_mol, short,
                    bio_name.replace("_"," "),
                    ref_mol)
                if desc:
                    analogues.append(desc)
                    existing_names.add(short)
                    existing_keys.add(key)
                    bio_count += 1
                    print(f"  + {short}: "
                          f"MW={desc['MW']}")
    except Exception as e:
        print(f"  Bioisostere error: {e}")

print(f"  Bioisostere analogues: {bio_count}")

# ── KNOWN AZD1283 ANALOGUES ───────────────
print("\nAdding known AZD1283 analogues...")
known = [
    ("F_tolyl",
     "Cc1ccc(S(=O)(=O)Nc2ncc(-c3ccccn3)cn2)cc1F",
     "Fluorine on tolyl ring"),
    ("CF3_tolyl",
     "Cc1ccc(S(=O)(=O)Nc2ncc(-c3ccccn3)cn2)cc1C(F)(F)F",
     "CF3 on tolyl ring"),
    ("Cl_tolyl",
     "Cc1ccc(S(=O)(=O)Nc2ncc(-c3ccccn3)cn2)cc1Cl",
     "Chlorine on tolyl ring"),
    ("Methoxy",
     "COc1ccc(S(=O)(=O)Nc2ncc(-c3ccccn3)cn2)cc1",
     "Methoxy on tolyl ring"),
    ("F_pyridine",
     "Cc1ccc(S(=O)(=O)Nc2ncc(-c3ccccnc3F)cn2)cc1",
     "Fluorine on pyridine ring"),
    ("Urea",
     "Cc1ccc(NC(=O)Nc2ncc(-c3ccccn3)cn2)cc1",
     "Sulfonamide to urea"),
    ("Amide",
     "Cc1ccc(C(=O)Nc2ncc(-c3ccccn3)cn2)cc1",
     "Sulfonamide to amide"),
    ("Nitro",
     "Cc1ccc(S(=O)(=O)Nc2ncc(-c3ccccn3)cn2)cc1[N+](=O)[O-]",
     "Nitro on tolyl ring"),
    ("CN_tolyl",
     "Cc1ccc(S(=O)(=O)Nc2ncc(-c3ccccn3)cn2)cc1C#N",
     "Cyano on tolyl ring"),
]

known_count = 0
for name, smi, mod in known:
    if name in existing_names:
        continue
    mol = sanitize(smi)
    if mol and lipinski_pass(mol):
        key = get_canonical_key(mol)
        if key in existing_keys:
            continue
        desc = get_descriptors(
            mol, name, mod, ref_mol)
        if desc:
            analogues.append(desc)
            existing_names.add(name)
            existing_keys.add(key)
            known_count += 1
            print(f"  + {name}: MW={desc['MW']}")

print(f"  Known analogues: {known_count}")

# ── R-GROUP PERMUTATION ───────────────────
print("\nGenerating R-group permutation analogues...")
print("Basis: Topliss + Hansch-Craig + Bioisostere")
positions = get_aromatic_positions(ref_mol)
print(f"  Aromatic C-H positions: {len(positions)}")
print(f"  Substituents: {len(SUB_SMILES)}")
print(f"  Max combinations: "
      f"{len(positions)*len(SUB_SMILES)}")

perm_count = 0

for pos in positions:
    if perm_count >= max_analogs:
        break
    for sub_smi, sub_basis in SUB_SMILES:
        if perm_count >= max_analogs:
            break
        try:
            new_mol = add_sub_to_position(
                ref_mol, pos, sub_smi)
            if new_mol is None:
                continue

            # InChIKey isomer filter
            key = get_canonical_key(new_mol)
            if key in existing_keys:
                continue

            if not lipinski_pass(new_mol):
                continue

            existing_keys.add(key)
            sub_clean = sub_smi\
                .replace("(","").replace(")","") \
                .replace("#","n").replace("/","")
            name = f"p{pos}_{sub_clean}"

            if name in existing_names:
                name = f"{name}_{perm_count}"

            desc = get_descriptors(
                new_mol, name,
                f"{sub_smi} at pos {pos} "
                f"({sub_basis.split(' — ')[0]})",
                ref_mol)
            if desc:
                analogues.append(desc)
                existing_names.add(name)
                perm_count += 1
                print(f"  + {name}: "
                      f"MW={desc['MW']} "
                      f"Tan={desc['Tanimoto']} "
                      f"IC50~{desc['predicted_ic50_nm']}nM")
        except Exception:
            continue

print(f"  Permutation analogues: {perm_count}")

# ── FILTER + SAVE ─────────────────────────
df = pd.DataFrame(
    [a for a in analogues if a])
df = df.drop_duplicates(subset=["smiles"])
df = df[df["Lipinski"] == "PASS"]
df = df.sort_values(
    "Tanimoto", ascending=False)
df = df.reset_index(drop=True)

print(f"\n{'='*45}")
print(f"ANALOGUE DESIGN SUMMARY")
print(f"{'='*45}")
print(f"Reference:          1")
print(f"Ring swaps:         {ring_count}")
print(f"Bioisosteres:       {bio_count}")
print(f"Known AZD1283:      {known_count}")
print(f"R-group permutation:{perm_count}")
print(f"  Positions used:   {len(positions)}")
print(f"  Substituents:     {len(SUB_SMILES)}")
print(f"Total analogues:    {len(df)}")

if ref_ic50 > 0:
    better = (df["predicted_ic50_nm"]
              < ref_ic50).sum()
    print(f"Better IC50:        {better}/{len(df)}")

df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# ── STRUCTURE IMAGE ───────────────────────
print("\nGenerating structure image...")
try:
    mols, legends = [], []
    for _, row in df.head(12).iterrows():
        mol = sanitize(row["smiles"])
        if mol:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            ic50_str = ""
            if pd.notna(
                    row.get("predicted_ic50_nm")):
                ic50_str = (
                    f"\nIC50~"
                    f"{row['predicted_ic50_nm']}"
                    f"nM")
            legends.append(
                f"{row['name']}\n"
                f"MW:{row['MW']}"
                f"{ic50_str}")
    img = Draw.MolsToGridImage(
        mols, molsPerRow=3,
        subImgSize=(400,300),
        legends=legends)
    img.save(out_png)
    print(f"Saved: {out_png}")
except Exception as e:
    print(f"Image error: {e}")
    from PIL import Image
    Image.new("RGB",(800,600),
        color=(255,255,255)).save(out_png)

print("\nStep 3 complete")
