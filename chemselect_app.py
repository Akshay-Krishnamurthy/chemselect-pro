"""
ChemSelect Pro v2 — Medicinal Chemistry Candidate Selection Platform
=====================================================================
Systematic 7-step workflow:
  1. Upload library + optional reference compound
  2. Choose analysis mode (Basic / Advanced) + select parameters to calculate
  3. Patent Helper — Markush clustering + SciFinder labels
  4. Property Calculator — SA Score, SCScore, PAINS, Brenk, Lipinski, LE, LipE, QED…
  5. MPO + Radar Analysis — vs reference compound if provided
  6. Candidate Selection & Full Statistics
  7. AI Report & Synthesis Recommendations
"""

import io, json, math, re, warnings, string
from collections import Counter
from math import pi, log

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# RDKit
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import (
        AllChem, Descriptors, Draw, rdDepictor,
        rdMolDescriptors, FilterCatalog, QED
    )
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import rdRGroupDecomposition
    from rdkit.ML.Cluster import Butina
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

warnings.filterwarnings("ignore")

# Gemini SDK (handles 429/retries internally — same pattern as reference app)
try:
    from google import genai as _genai
    _LEGACY_GEMINI = False
except ImportError:
    try:
        import google.generativeai as _genai
        _LEGACY_GEMINI = True
    except ImportError:
        _genai = None
        _LEGACY_GEMINI = None

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChemSelect Pro",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.stApp{background:#0a0e17;color:#cdd9e5;}
[data-testid="stSidebar"]{background:#111827;border-right:1px solid #1e2d3d;}
h1{color:#4fc3f7!important;font-weight:700;letter-spacing:-0.5px;}
h2{color:#81d4fa!important;font-weight:600;}
h3{color:#b3e5fc!important;font-weight:500;}
.badge{display:inline-flex;align-items:center;gap:8px;
       background:linear-gradient(135deg,#0d47a1,#1565c0);color:#e3f2fd;
       border-radius:24px;padding:6px 18px;font-size:.78rem;font-weight:700;
       letter-spacing:.8px;text-transform:uppercase;margin-bottom:12px;
       border:1px solid #1976d2;}
.card{background:#111827;border:1px solid #1e2d3d;border-radius:12px;
      padding:18px 22px;margin:8px 0;transition:border-color .2s,box-shadow .2s;}
.card:hover{border-color:#4fc3f7;box-shadow:0 0 16px #4fc3f720;}
.info{background:#0a1929;border-left:4px solid #4fc3f7;border-radius:6px;
      padding:14px 18px;margin:12px 0;font-size:.87rem;line-height:1.7;color:#b0bec5;}
.warn{background:#1a1200;border-left:4px solid #f59e0b;border-radius:6px;
      padding:12px 16px;margin:10px 0;font-size:.87rem;color:#fcd34d;}
.good{background:#0a1f0a;border-left:4px solid #4caf50;border-radius:6px;
      padding:12px 16px;margin:10px 0;font-size:.87rem;color:#a5d6a7;}
.tag{display:inline-block;background:#1e2d3d;color:#4fc3f7;border:1px solid #1976d2;
     border-radius:6px;padding:3px 10px;font-size:.76rem;font-weight:600;margin:2px;
     font-family:'IBM Plex Mono',monospace;}
.ai-box{background:#0a1929;border:1px solid #1565c0;border-radius:10px;
        padding:22px 26px;font-size:.9rem;line-height:1.85;white-space:pre-wrap;color:#cdd9e5;}
hr.d{border:none;border-top:1px solid #1e2d3d;margin:20px 0;}
.stButton>button{background:linear-gradient(135deg,#0d47a1,#1565c0)!important;
    color:#e3f2fd!important;border:none!important;border-radius:8px!important;
    font-weight:600!important;padding:8px 20px!important;transition:all .2s!important;}
.stButton>button:hover{background:linear-gradient(135deg,#1565c0,#1976d2)!important;
    box-shadow:0 4px 20px #1976d240!important;transform:translateY(-1px);}
[data-testid="stFileUploader"]{background:#111827!important;
    border:2px dashed #1e2d3d!important;border-radius:10px!important;}
[data-testid="stDataFrame"]{border-radius:8px;overflow:hidden;}
[aria-selected="true"]{color:#4fc3f7!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "df": None, "ref_df": None,
    "mols": None, "ids": None, "ref_mols": [],
    "smiles_col": "SMILES", "id_col": "Structure_ID",
    "analysis_mode": "basic", "calc_flags": {},
    "analysis_cols": [],
    "prop_df": None, "markush_summary": None,
    "rgroup_long": None, "clusters": None,
    "mpo_df": None, "final_df": None,
    "api_key": "", "api_provider": "claude",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# CHEMISTRY HELPERS
# ─────────────────────────────────────────────────────────────────

def safe_mol(smi):
    if not RDKIT_OK or pd.isna(smi): return None
    try: return Chem.MolFromSmiles(str(smi).strip())
    except: return None

def safe_scaffold(m):
    if m is None: return None
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        return s if s and s.GetNumAtoms() > 0 else None
    except: return None

def mol_svg(mol, w=230, h=185):
    if mol is None or not RDKIT_OK: return ""
    try:
        rdDepictor.Compute2DCoords(mol)
        from rdkit.Chem.Draw import rdMolDraw2D
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        d.drawOptions().addStereoAnnotation = True
        d.drawOptions().padding = 0.15
        d.DrawMolecule(mol); d.FinishDrawing()
        return d.GetDrawingText()
    except: return ""

def rg_scifindr(col):
    """R1→Ra, R2→Rb … (SciFinder positional label convention)."""
    if not col: return "N/A"
    n = str(col).strip().lstrip("Rr")
    try:
        p = int(n)
        return f"R{string.ascii_uppercase[p-1]}" if p-1 < 26 else f"R{p}"
    except: return str(col)

# ── Lipinski Ro5 ──────────────────────────────────────────────────

# ── Common R-group representation ─────────────────────────────────
def rg_common_repr(smi):
    """
    Convert an R-group SMILES to a compact common representation.
    e.g. OC(F)F → -OCHF2, C(F)(F)F → -CF3, [*]F → -F
    Falls back to raw SMILES if no conversion possible.
    """
    if not smi or str(smi).strip() in ("","nan","None","[H]"):
        return "H"
    smi = str(smi).strip()
    import re as _re
    # Remove attachment point atoms [*:N], [*], [1*], * etc.
    clean = _re.sub(r'\[\d*\*[:\d]*\]|\*', '', smi).strip()
    # Also strip leading/trailing dots or dashes left after removal
    clean = clean.strip('-. ')
    # Known patterns → common name
    _MAP = {
        # Hydrogen (naked attachment = H substituent)
        "[H]":         "H",
        "":            "H",
        # Halogens
        "F":           "-F",
        "Cl":          "-Cl",
        "Br":          "-Br",
        "I":           "-I",
        # Simple alkyl
        "C":           "-CH3",
        "CC":          "-C2H5",
        "CCC":         "-nPr",
        "C(C)C":       "-iPr",
        "C(C)(C)C":    "-tBu",
        "CO":          "-CH2OH",
        "OC":          "-OMe",
        "OCC":         "-OEt",
        "OCCC":        "-OnPr",
        "N":           "-NH2",
        "NC":          "-NHMe",
        "NCC":         "-NHEt",
        "N(C)C":       "-NMe2",
        # Fluoromethyl groups (most common in med chem)
        "C(F)(F)F":    "-CF3",
        "C(F)F":       "-CHF2",
        "CF":          "-CH2F",
        "FC(F)F":      "-CHF2",
        "FC(F)(F)":    "-CF3",
        # Fluoroalkoxy (very common — these appear after stripping [*:N])
        "OC(F)(F)F":   "-OCF3",
        "OC(F)F":      "-OCHF2",
        "OC(F)(F)":    "-OCHF2",
        "OCF":         "-OCH2F",
        "FC(F)(F)O":   "-OCF3",
        "FC(F)O":      "-OCHF2",
        # Fluoroalkyl chains (e.g. FC(F)(F)CO[*:4] → FC(F)(F)CO → -CF3CH2O-)
        "FC(F)(F)CO":  "-CF3CH2O-",
        "C(F)(F)FCO":  "-CF3CH2O-",
        "FC(F)(F)C":   "-CF3CH2-",
        "C(F)(F)FC":   "-CF3CH2-",
        "OCC(F)(F)F":  "-OCH2CF3",
        "CC(F)(F)F":   "-CH2CF3",
        "C(C(F)(F)F)": "-CH2CF3",
        # Difluoromethoxy variants
        "OC(F)(F)CO":  "-OCF2CH2O-",
        "FC(F)CO":     "-CHF2CH2O-",
        "OC(F)CO":     "-OCHFCH2O-",
        # Oxygen
        "O":           "-OH",
        "OO":          "-OOH",
        # Carbonyl
        "C=O":         "-CHO",
        "C(=O)N":      "-CONH2",
        "C(=O)O":      "-COOH",
        "C(=O)C":      "-COCH3",
        "C(=O)OC":     "-COOMe",
        # Nitrogen
        "N=O":         "-NO",
        "N(=O)=O":     "-NO2",
        "[N+](=O)[O-]":"-NO2",
        # Sulfur
        "S":           "-SH",
        "SC":          "-SMe",
        # Nitrile
        "C#N":         "-CN",
        # Aromatic
        "c1ccccc1":    "-Ph",
        "Cc1ccccc1":   "-Bn",
        "c1ccncc1":    "-4-Py",
        "c1ccccn1":    "-2-Py",
        "c1ccncc1":    "-Pyr",
        # Charged / special groups (common in HCIE-type series)
        "O=C([O-])":   "-COO⁻",
        "C(=O)[O-]":   "-COO⁻",
        "C([O-])=O":   "-COO⁻",
        "C(O)=O":      "-COOH",
        "S(=O)(=O)N":  "-SO2NH2",
        "S(=O)(=O)O":  "-SO3H",
        "C(=O)O":      "-COOH",
        "C(=O)N":      "-CONH2",
    }
    # Direct lookup on cleaned SMILES
    if clean in _MAP:
        return _MAP[clean]
    # Try canonical SMILES lookup (handles different orderings)
    if RDKIT_OK:
        try:
            m = Chem.MolFromSmiles(clean)
            if m:
                can = Chem.MolToSmiles(m)
                if can in _MAP:
                    return _MAP[can]
                # Also try with H atoms removed
                can_noH = _re.sub(r'\[H\]','',can).strip()
                if can_noH in _MAP:
                    return _MAP[can_noH]
                # Build compact representation from atom counts
                counts = {}
                for a in m.GetAtoms():
                    sym = a.GetSymbol()
                    if sym != "*":
                        counts[sym] = counts.get(sym, 0) + 1
                        counts['H'] = counts.get('H', 0) + a.GetTotalNumHs()
                def _fmt(sym, n):
                    subs = "".join(chr(0x2080+int(d)) for d in str(n)) if n>1 else ""
                    return f"{sym}{subs}"
                parts = []
                for sym in ['C','H','N','O','F','Cl','Br','I','S','P']:
                    if sym in counts and counts[sym] > 0:
                        parts.append(_fmt(sym, counts[sym]))
                if parts:
                    return "-" + "".join(parts)
        except: pass
    # Last resort
    return f"-{clean}" if clean else "H"

# ── Core SVG with hover tooltips at R-group attachment points ──────
def core_svg_with_tooltips(core_mol, rg_opts, w=380, h=280):
    """
    Draw the core scaffold SVG. Atoms that are attachment points (*) get
    a data-rgroup attribute. JavaScript injected into the SVG shows a
    tooltip card on hover listing all R-group variants for that position.
    """
    if core_mol is None or not RDKIT_OK:
        return ""
    try:
        import re as _re
        from rdkit.Chem.Draw import rdMolDraw2D

        # Map attachment point atoms by their atom map number [*:N] → RN
        # e.g. atom with map num 4 → R4
        atom_to_rg = {}
        for a in core_mol.GetAtoms():
            if a.GetAtomicNum() == 0:          # dummy/attachment atom
                map_num = a.GetAtomMapNum()    # e.g. 1,2,3...
                if map_num > 0:
                    rg_key = f"R{map_num}"     # R1, R2, R4 ...
                    if rg_key in rg_opts:
                        atom_to_rg[a.GetIdx()] = rg_key
                    else:
                        # Try case-insensitive match
                        for k in rg_opts:
                            if k.upper() == rg_key.upper():
                                atom_to_rg[a.GetIdx()] = k
                                break
        # Fallback: if no map numbers found, assign by sorted position
        if not atom_to_rg:
            attach_atoms = [a.GetIdx() for a in core_mol.GetAtoms()
                            if a.GetAtomicNum() == 0]
            rg_names = sorted(rg_opts.keys())
            for i, idx in enumerate(attach_atoms):
                if i < len(rg_names):
                    atom_to_rg[idx] = rg_names[i]

        # Draw
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        d.drawOptions().addStereoAnnotation = True
        d.drawOptions().padding = 0.18
        d.DrawMolecule(core_mol)
        d.FinishDrawing()
        raw_svg = d.GetDrawingText()

        # Build tooltip data as JS object
        # { "R1": "Ra | F | Cl | CF3", "R2": "Rb | OCH3 | OEt" }
        tooltip_data = {}
        for rg_name, variants in rg_opts.items():
            sf = rg_scifindr(rg_name)
            common = [rg_common_repr(v) for v in sorted(variants)]
            raw    = sorted([str(v) for v in variants])
            lines  = [f"{sf} ({rg_name})"]
            for r, c in zip(raw, common):
                lines.append(f"  {c}  [{r}]")
            tooltip_data[rg_name] = "\n".join(lines)

        tooltip_js = str(tooltip_data).replace("'", '"')

        # Map atom index → pixel coords from SVG
        # We inject a transparent circle over each attachment atom that triggers hover
        # Parse atom positions from SVG <circle> or <path> elements near labels
        # Simpler: use RDKit's GetDrawCoords
        atom_circles = []
        try:
            for idx, rg_name in atom_to_rg.items():
                pt = d.GetDrawCoords(idx)
                x, y = pt.x, pt.y
                tip = tooltip_data.get(rg_name,"").replace('"','\"').replace("\n","&#10;")
                sf  = rg_scifindr(rg_name)
                atom_circles.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" '
                    f'fill="rgba(79,195,247,0.18)" stroke="#4fc3f7" stroke-width="1.5" '
                    f'style="cursor:pointer" class="rg-atom" '
                    f'data-rg="{rg_name}" data-sf="{sf}" data-tip="{tip}"/>'
                )
        except: pass

        # Tooltip div + CSS + JS (self-contained inside SVG via foreignObject is tricky;
        # instead we wrap the SVG in a div and add HTML tooltip outside)
        circles_svg = "\n".join(atom_circles)

        # Insert circles before </svg>
        raw_svg = raw_svg.replace("</svg>", circles_svg + "\n</svg>")

        # Wrap in div with tooltip machinery
        tooltip_css = """
<style>
.rg-tooltip{
  position:absolute;display:none;background:#0d2137;color:#e0f7fa;
  border:1px solid #4fc3f7;border-radius:8px;padding:10px 14px;
  font-size:12px;font-family:monospace;white-space:pre;
  pointer-events:none;z-index:9999;max-width:320px;
  box-shadow:0 4px 20px rgba(0,0,0,0.5);line-height:1.6;
}
.rg-tooltip .tip-title{
  font-weight:bold;color:#4fc3f7;font-size:13px;border-bottom:1px solid #1e2d3d;
  padding-bottom:4px;margin-bottom:6px;display:block;
}
</style>
"""
        tooltip_html = '<div class="rg-tooltip" id="rg-tip"></div>'
        js = """
<script>
(function(){
  var tip = document.getElementById('rg-tip');
  if(!tip) return;
  document.querySelectorAll('.rg-atom').forEach(function(el){
    el.addEventListener('mouseenter', function(e){
      var lines = el.getAttribute('data-tip').split('&#10;');
      var title = lines[0];
      var body  = lines.slice(1).join('\n');
      tip.innerHTML = '<span class="tip-title">'+title+'</span>' + body;
      tip.style.display = 'block';
    });
    el.addEventListener('mousemove', function(e){
      tip.style.left  = (e.pageX + 14) + 'px';
      tip.style.top   = (e.pageY - 10) + 'px';
    });
    el.addEventListener('mouseleave', function(){
      tip.style.display = 'none';
    });
  });
})();
</script>
"""
        full_html = (
            '<div style="position:relative;display:inline-block;'
            'background:white;border-radius:8px;padding:8px">'
            + tooltip_css
            + raw_svg
            + tooltip_html
            + js
            + '</div>'
        )
        return full_html
    except:
        return ""

def lipinski_ro5(mol):
    if mol is None: return None, None, {}
    try:
        mw=Descriptors.ExactMolWt(mol); lp=Descriptors.MolLogP(mol)
        hbd=rdMolDescriptors.CalcNumHBD(mol); hba=rdMolDescriptors.CalcNumHBA(mol)
        v = int(mw>500)+int(lp>5)+int(hbd>5)+int(hba>10)
        return v==0, v, {"MW":round(mw,2),"LogP":round(lp,2),"HBD":hbd,"HBA":hba}
    except: return None, None, {}

# ── PAINS catalog ─────────────────────────────────────────────────
_PAINS = None
def _pains_cat():
    global _PAINS
    if _PAINS is None and RDKIT_OK:
        try:
            p = FilterCatalogParams()
            p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            _PAINS = FilterCatalog.FilterCatalog(p)
        except: pass
    return _PAINS

def check_pains(mol):
    if mol is None: return None, ""
    cat = _pains_cat()
    if cat is None: return None, "catalog unavailable"
    try:
        e = cat.GetFirstMatch(mol)
        return (True, e.GetDescription()) if e else (False, "")
    except: return None, ""

# ── Brenk catalog ─────────────────────────────────────────────────
_BRENK = None
def _brenk_cat():
    global _BRENK
    if _BRENK is None and RDKIT_OK:
        try:
            p = FilterCatalogParams()
            p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            _BRENK = FilterCatalog.FilterCatalog(p)
        except: pass
    return _BRENK

def check_brenk(mol):
    if mol is None: return None, ""
    cat = _brenk_cat()
    if cat is None: return None, "catalog unavailable"
    try:
        e = cat.GetFirstMatch(mol)
        return (True, e.GetDescription()) if e else (False, "")
    except: return None, ""

# ── SA Score ─────────────────────────────────────────────────────
def sa_score(mol):
    """Ertl & Schuffenhauer 2009. 1=easy, 10=hard. Target ≤4."""
    if mol is None: return None
    # Try official RDKit contrib sascorer
    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sp = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sp not in sys.path: sys.path.insert(0, sp)
        import sascorer
        return round(sascorer.calculateScore(mol), 3)
    except Exception:
        pass
    # Complexity-based proxy
    try:
        mw = Descriptors.ExactMolWt(mol)
        nr = rdMolDescriptors.CalcNumRings(mol)
        ns = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nb = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nc = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        raw = 1.0 + ns*2.0 + nb*1.5 + nc*0.5 + max(0,(mw-300)/200)*1.5 + log(max(1,nr))*0.8
        return min(10.0, max(1.0, round(raw, 2)))
    except: return None

# ── SCScore (Coley/MIT 2018 proxy) ───────────────────────────────
def sc_score(mol):
    """
    SCScore 1–5 (1=cheap/easy, 5=complex/expensive).
    Coley, C.W. et al. J. Chem. Inf. Model. 2018, 58, 252-261.
    Full model: pip install scscore.
    This is a descriptor-based proxy that approximates the neural network.
    """
    if mol is None: return None
    try:
        mw   = Descriptors.ExactMolWt(mol)
        nr   = rdMolDescriptors.CalcNumRings(mol)
        ns   = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nb   = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nc   = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        na   = mol.GetNumHeavyAtoms()
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        raw = (1.0
               + min(na/30, 1.2)
               + ns*0.6 + nb*0.5 + nc*0.35
               + max(0, nr-2)*0.25
               + max(0, mw-350)/600
               - fsp3*0.3)
        return min(5.0, max(1.0, round(raw, 2)))
    except: return None

# ── QED ──────────────────────────────────────────────────────────
def qed_score(mol):
    if mol is None: return None
    try: return round(QED.qed(mol), 3)
    except: return None

# ── Fluorine / halogen alerts ─────────────────────────────────────
def fluorine_count(mol):
    """Count fluorine atoms. ≥5 F = excessive fluorination alert (metabolic/toxicity risk)."""
    if mol is None: return None
    try: return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==9)
    except: return None

def halogen_count(mol):
    """Total halogens (F+Cl+Br+I). High halogen load → toxicity, PK issues."""
    if mol is None: return None
    try: return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9,17,35,53))
    except: return None

def halogen_alert(mol):
    """
    Returns (flag, description) for halogen alerts:
    - ≥5 F: excessive fluorination
    - Any I: iodine (phototoxicity, MW penalty)
    - ≥3 Cl or ≥2 Br: polyhalogenation alert
    """
    if mol is None: return None, ""
    try:
        f  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==9)
        cl = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==17)
        br = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==35)
        i  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==53)
        alerts = []
        if f >= 5: alerts.append(f"{f}×F — excessive fluorination (metabolic instability, hERG risk)")
        if i > 0:  alerts.append(f"{i}×I — iodine present (phototoxicity, high MW penalty)")
        if cl >= 3:alerts.append(f"{cl}×Cl — poly-chlorination (toxicity, environmental persistence)")
        if br >= 2:alerts.append(f"{br}×Br — poly-bromination (toxicity alert)")
        return (len(alerts)>0, "; ".join(alerts) if alerts else "")
    except: return None, ""

# ── Estimated aqueous solubility (ESOL — Delaney 2004) ───────────
def esol_logS(mol):
    """
    ESOL (Delaney 2004) LogS estimate. 
    LogS > -2: highly soluble; -2 to -4: soluble; -4 to -6: low; < -6: insoluble.
    Returns (logS, category).
    """
    if mol is None: return None, None
    try:
        mw   = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ap   = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        na   = mol.GetNumHeavyAtoms()
        # Delaney ESOL equation
        logS = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*(ap/max(na,1))
        logS = round(logS, 3)
        if logS > -2:   cat = "Highly soluble"
        elif logS > -4: cat = "Soluble"
        elif logS > -6: cat = "Low solubility"
        else:           cat = "Insoluble"
        return logS, cat
    except: return None, None

# ── pKa proxy (basic nitrogen count) ─────────────────────────────
def pka_proxy(mol):
    """
    Proxy for basic pKa. Counts basic nitrogen types and estimates dominant pKa range.
    NOT a replacement for Epik/Marvin — use as a quick filter only.
    Returns (n_basic_N, pka_estimate_range, flag).
    """
    if mol is None: return None, None, None
    try:
        basic_n, amine_sp3, arom_n, amidine = 0, 0, 0, 0
        for a in mol.GetAtoms():
            if a.GetAtomicNum() != 7: continue
            if a.GetIsAromatic():
                arom_n += 1
            elif a.GetTotalDegree() <= 3 and not a.GetIsAromatic():
                amine_sp3 += 1
                basic_n += 1
        # Simple estimate
        if amine_sp3 >= 2:
            pka_range = "8–11 (polyamine — multiple protonation states)"
            flag = "REVIEW"
        elif amine_sp3 == 1:
            pka_range = "8–10 (aliphatic amine)"
            flag = "OK"
        elif arom_n >= 1:
            pka_range = "2–6 (aromatic/pyridine N)"
            flag = "OK"
        else:
            pka_range = "neutral / weak acid"
            flag = "OK"
        return basic_n, pka_range, flag
    except: return None, None, None

# ── Veber oral bioavailability rules (2002) ──────────────────────
def veber_rules(mol):
    """
    Veber et al. J. Med. Chem. 2002: RotBonds ≤10, TPSA ≤140.
    Better predictor of oral bioavailability than Lipinski alone.
    """
    if mol is None: return None, ""
    try:
        rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        pass_ = rb <= 10 and tpsa <= 140
        detail = f"RotBonds={rb}(≤10), TPSA={tpsa:.1f}(≤140)"
        return ("PASS" if pass_ else "FAIL"), detail
    except: return None, ""

# ── Egan egg (2000) ───────────────────────────────────────────────
def egan_egg(mol):
    """
    Egan et al. J. Med. Chem. 2000: LogP ≤5.88, TPSA ≤131.6.
    Passive intestinal absorption predictor.
    """
    if mol is None: return None, ""
    try:
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        pass_ = logp <= 5.88 and tpsa <= 131.6
        detail = f"LogP={logp:.2f}(≤5.88), TPSA={tpsa:.1f}(≤131.6)"
        return ("PASS" if pass_ else "FAIL"), detail
    except: return None, ""

# ── CNS MPO / BBB score ──────────────────────────────────────────
def cns_bbb_score(mol):
    """
    Simplified CNS-BBB penetration score based on:
    MW ≤450, LogP 1–4, HBD ≤3, TPSA ≤90, RotBonds ≤8.
    Score 0–5 (5=ideal CNS candidate).
    """
    if mol is None: return None
    try:
        mw   = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        tpsa = Descriptors.TPSA(mol)
        rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
        score = (int(mw<=450) + int(1<=logp<=4) +
                 int(hbd<=3) + int(tpsa<=90) + int(rb<=8))
        return score
    except: return None

# ── Ro3 (fragment-like, Congreve 2003) ───────────────────────────
def rule_of_3(mol):
    """Fragment-like: MW≤300, LogP≤3, HBD≤3, HBA≤3, RotBonds≤3, TPSA≤60."""
    if mol is None: return None, ""
    try:
        mw   = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        v = int(mw>300)+int(logp>3)+int(hbd>3)+int(hba>3)+int(rb>3)+int(tpsa>60)
        return ("PASS" if v==0 else "FAIL"), f"{v} violation(s)"
    except: return None, ""

# ── Muegge druglike filter (2001) ────────────────────────────────
def muegge_filter(mol):
    """
    Muegge et al. J. Med. Chem. 2001:
    200≤MW≤600, -2≤LogP≤5, HBD≤5, HBA≤10, TPSA≤150, RotBonds≤15, Rings≤7.
    Broader than Lipinski, includes natural-product space.
    """
    if mol is None: return None, ""
    try:
        mw   = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        nr   = rdMolDescriptors.CalcNumRings(mol)
        fails=[]
        if not (200<=mw<=600): fails.append(f"MW={mw:.0f}")
        if not (-2<=logp<=5):  fails.append(f"LogP={logp:.2f}")
        if hbd>5:  fails.append(f"HBD={hbd}")
        if hba>10: fails.append(f"HBA={hba}")
        if rb>15:  fails.append(f"RotBonds={rb}")
        if tpsa>150: fails.append(f"TPSA={tpsa:.0f}")
        if nr>7:   fails.append(f"Rings={nr}")
        return ("PASS" if not fails else "FAIL"), ("; ".join(fails) if fails else "")
    except: return None, ""

# ── Sp3 fraction (Fsp3) ───────────────────────────────────────────
def fsp3(mol):
    """
    Fraction of sp3 carbons. Fsp3 > 0.42 correlates with better solubility,
    lower toxicity, and higher clinical success rates (Lovering et al. 2009).
    """
    if mol is None: return None
    try: return round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
    except: return None

# ── Nitro / reactive group alerts ────────────────────────────────
_REACTIVE_SMARTS = {
    "Nitro group":           "[N+](=O)[O-]",
    "Aldehyde":              "[CX3H1](=O)",
    "Michael acceptor":      "[CX3]=[CX3][CX3]=O",
    "Epoxide":               "C1OC1",
    "Acyl halide":           "[CX3](=O)[F,Cl,Br,I]",
    "Isocyanate":            "[N]=[C]=[O]",
    "Anhydride":             "[CX3](=O)O[CX3](=O)",
    "Peroxide":              "OO",
    "Thiol":                 "[SX2H]",
    "Diazo":                 "N=N",
    "Hydrazine":             "NN",
}
_REACTIVE_MOLS = {}
def _get_reactive():
    global _REACTIVE_MOLS
    if not _REACTIVE_MOLS and RDKIT_OK:
        for name, sma in _REACTIVE_SMARTS.items():
            try:
                m = Chem.MolFromSmarts(sma)
                if m: _REACTIVE_MOLS[name] = m
            except: pass
    return _REACTIVE_MOLS

def reactive_alerts(mol):
    """
    Checks for known reactive/toxic substructures beyond Brenk.
    Returns (n_alerts, description).
    """
    if mol is None: return None, ""
    try:
        hits = []
        for name, pat in _get_reactive().items():
            if mol.HasSubstructMatch(pat):
                hits.append(name)
        return len(hits), "; ".join(hits) if hits else ""
    except: return 0, ""

# ── Ligand & lipophilic efficiency ───────────────────────────────
def le(pic50, nheavy):
    try:
        if pd.isna(pic50) or pd.isna(nheavy) or float(nheavy)==0: return None
        return round(1.37*float(pic50)/float(nheavy), 4)
    except: return None

def lipe(pic50, logp):
    try:
        if pd.isna(pic50) or pd.isna(logp): return None
        return round(float(pic50)-float(logp), 4)
    except: return None

# ── Stereocenters ────────────────────────────────────────────────
def n_stereo(mol):
    if mol is None: return None
    try: return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    except: return None

# ── Tanimoto ─────────────────────────────────────────────────────
def tanimoto(mol, ref):
    if mol is None or ref is None: return None
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(ref, 2, 2048)
        return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)
    except: return None

# ── MPO (Pfizer CNS-MPO, Wager et al. 2010) ──────────────────────
def _smw(v):
    if v<=360: return 1.0
    if v>=500: return 0.0
    return (500-v)/140
def _slp(v):
    if v<=3: return 1.0
    if v>=5: return 0.0
    return (5-v)/2
def _shbd(v):
    return {0:1.0,1:0.5}.get(int(v),0.0)
def _shba(v):
    if v<=3.5: return 1.0
    if v>=7: return 0.0
    return (7-v)/3.5
def _stpsa(v):
    if 40<=v<=90: return 1.0
    if v<40: return v/40
    if v<=120: return (120-v)/30
    return 0.0
def _spka(mol):
    bn=sum(1 for a in mol.GetAtoms()
           if a.GetAtomicNum()==7 and not a.GetIsAromatic() and a.GetTotalDegree()<4)
    return {0:0.5,1:1.0}.get(bn,0.0)

def compute_mpo(mol):
    if mol is None: return None
    try:
        mw=Descriptors.ExactMolWt(mol); lp=Descriptors.MolLogP(mol)
        hbd=rdMolDescriptors.CalcNumHBD(mol); hba=rdMolDescriptors.CalcNumHBA(mol)
        tpsa=Descriptors.TPSA(mol)
        s={"s_MW":_smw(mw),"s_LogP":_slp(lp),"s_HBD":_shbd(hbd),
           "s_HBA":_shba(hba),"s_TPSA":_stpsa(tpsa),"s_pKa":_spka(mol)}
        s["MPO_Total"]=round(sum(s.values()),3)
        return s
    except: return None

# ── Butina clustering (Akshay's pipeline) ────────────────────────
def run_clustering(mols, cutoff=0.4):
    scafs=[safe_scaffold(m) for m in mols]
    valid=[(i,s) for i,(m,s) in enumerate(zip(mols,scafs)) if s is not None]
    if not valid: return [],[]
    idxs,vscafs=zip(*valid)
    fps=[AllChem.GetMorganFingerprintAsBitVect(s,2,1024) for s in vscafs]
    dists=[]
    for i in range(1,len(fps)):
        sims=DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
    raw=Butina.ClusterData(dists,len(fps),cutoff,isDistData=True)
    clusters=sorted([tuple(idxs[j] for j in cl) for cl in raw],key=len,reverse=True)
    return clusters,list(vscafs)

def run_rgroup_decomp(core,cluster_mols):
    rg=rdRGroupDecomposition.RGroupDecomposition([core])
    for m in cluster_mols:
        try: rg.Add(m)
        except: pass
    try: rg.Process()
    except: return pd.DataFrame(),{}
    rows=[]
    for row in rg.GetRGroupsAsRows():
        e={}
        for k,v in row.items():
            try: e[k]=Chem.MolToSmiles(v) if v else None
            except: e[k]=None
        rows.append(e)
    rg_df=pd.DataFrame(rows)
    opts={}
    for col in rg_df.columns:
        if col!="Core":
            vals=sorted(set(x for x in rg_df[col].dropna() if str(x).strip()))
            if vals: opts[col]=vals
    return rg_df,opts

# ─────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────
_C=["#4fc3f7","#f48fb1","#a5d6a7","#ffcc02","#ce93d8","#ff8a65","#80deea","#ef9a9a"]

def dark_ax(fig,ax):
    fig.patch.set_facecolor("#0a0e17"); ax.set_facecolor("#111827")
    for sp in ax.spines.values(): sp.set_color("#1e2d3d")
    ax.tick_params(colors="#78909c")
    ax.xaxis.label.set_color("#78909c"); ax.yaxis.label.set_color("#78909c")
    ax.title.set_color("#cdd9e5")

def dfig(fs=(9,4)):
    fig,ax=plt.subplots(figsize=fs); dark_ax(fig,ax); return fig,ax

def radar_fig(labels, series_v, series_n, title="", ref_idx=0):
    """
    Radar plot matching the Gemini/mentor style:
    - White background
    - Reference compound = grey dashed line
    - Other series = solid coloured lines (blue for top hit)
    - Minimum 3 axes enforced externally before calling this
    """
    N = len(labels)
    if N < 2:
        return None
    angles = [n / N * 2 * pi for n in range(N)] + [0]

    fig = plt.figure(figsize=(6, 6), facecolor="white")
    ax  = fig.add_subplot(111, polar=True, facecolor="white")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Axis labels — clean black text
    ax.set_thetagrids(np.degrees(angles[:-1]), labels,
                      fontsize=11, color="black", fontweight="normal")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                       fontsize=8, color="grey")
    ax.set_rlabel_position(0)
    ax.grid(color="lightgrey", linewidth=0.8)
    ax.spines["polar"].set_color("black")

    solid_colors = ["#1565C0", "#C62828", "#2E7D32", "#F57F17",
                    "#6A1B9A", "#00838F", "#4E342E", "#AD1457"]

    for i, (vals, name) in enumerate(zip(series_v, series_n)):
        v = list(vals) + [vals[0]]
        is_ref = (i == ref_idx and len(series_v) > 1)
        if is_ref:
            # Grey dashed — reference compound style
            ax.plot(angles, v, lw=2.2, linestyle="dashed",
                    color="grey", label=f"Reference: {name}", zorder=4)
            ax.fill(angles, v, alpha=0.08, color="grey")
        else:
            cidx = (i - 1) if i > ref_idx else i
            c = solid_colors[cidx % len(solid_colors)]
            lbl = f"Top Hit: {name}" if cidx == 0 and len(series_v) > 1 else name
            ax.plot(angles, v, lw=2.5, linestyle="solid",
                    color=c, label=lbl, zorder=5)
            ax.fill(angles, v, alpha=0.12, color=c)

    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1),
              fontsize=9, frameon=True,
              facecolor="white", edgecolor="lightgrey")
    if title:
        ax.set_title(title, pad=20, color="black",
                     fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────
# AI CALL
# ─────────────────────────────────────────────────────────────────
def call_ai(prompt, provider, api_key, max_tokens=1200):
    if not api_key:
        return "⚠️ No API key set. Add it in the sidebar."
    import urllib.request, json as _j, time, urllib.error

    def _request(url, headers, body):
        """Fire request with up to 3 retries on 429."""
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=body,
                                             headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=90) as r:
                    return _j.loads(r.read()), None
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 20 * (attempt + 1)   # 20s, 40s, 60s
                    time.sleep(wait)
                    if attempt == 2:
                        return None, f"Rate limit (429) — tried 3×. Wait a minute then retry."
                else:
                    return None, f"HTTP {e.code}: {e.reason}"
            except Exception as e:
                return None, str(e)
        return None, "Max retries exceeded."

    try:
        if provider == "claude":
            data, err = _request(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
                _j.dumps({"model": "claude-sonnet-4-5",
                           "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
            )
            if err: return f"⚠️ API error (claude): {err}"
            return data["content"][0]["text"]

        elif provider == "openai":
            data, err = _request(
                "https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
                _j.dumps({"model": "gpt-4o-mini",
                           "max_tokens": max_tokens,
                           "messages": [
                               {"role": "system", "content": "You are a senior medicinal chemist."},
                               {"role": "user",   "content": prompt}
                           ]}).encode()
            )
            if err: return f"⚠️ API error (openai): {err}"
            return data["choices"][0]["message"]["content"]

        elif provider == "gemini":
            # Use google-generativeai SDK — handles 429/retries internally
            if _genai is None:
                return "⚠️ google-generativeai not installed. Run: pip install google-generativeai"
            try:
                if _LEGACY_GEMINI:
                    _genai.configure(api_key=api_key)
                    return _genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text
                else:
                    client = _genai.Client(api_key=api_key)
                    return client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt).text
            except Exception as e:
                return f"⚠️ API error (gemini): {e}"

    except Exception as e:
        return f"⚠️ API error ({provider}): {e}"
    return "Unknown provider."

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ ChemSelect Pro")
    st.caption("Medicinal Chemistry · Candidate Selection v2")
    st.markdown('<hr class="d">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Provider")
    st.selectbox("Provider",["claude","openai","gemini"],key="api_provider",
                 format_func=lambda x:{"claude":"Claude (Anthropic)",
                                        "openai":"GPT-4o (OpenAI)",
                                        "gemini":"Gemini 2.0 Flash (Google)"}[x])
    lbl={"claude":"Anthropic","openai":"OpenAI","gemini":"Google"}[st.session_state.api_provider]
    st.text_input(f"{lbl} API Key",type="password",key="api_key",placeholder="Paste key here…")
    if st.session_state.api_key: st.success("✓ API key set")
    else: st.caption("Required only for Step 7 — AI Report")
    st.markdown('<hr class="d">', unsafe_allow_html=True)
    st.markdown("### 📋 Workflow Steps")
    for n,l in [("1","Upload Library"),("2","Mode & Parameters"),
                ("3","Patent Helper"),("4","Property Calculator"),
                ("5","MPO & Radar"),("6","Selection & Stats"),("7","AI Report")]:
        st.markdown(f'<span class="badge">Step {n}</span> {l}',unsafe_allow_html=True)
    st.markdown('<hr class="d">', unsafe_allow_html=True)
    if RDKIT_OK: st.success("✓ RDKit ready")
    else: st.error("RDKit not found\n`pip install rdkit`")

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("# ⚗️ ChemSelect Pro")
st.markdown("**Systematic Medicinal Chemistry Candidate Selection** — cluster → filter → score → rank → synthesise.")
st.markdown('<hr class="d">', unsafe_allow_html=True)

with st.expander("📚 Selection Framework & All Criteria (click to expand)", expanded=False):
    st.markdown("""
<div class="info">
<strong>Industry decision framework:</strong> apply filters in order.<br>
Activity → Structural alerts (PAINS/Brenk hard filter) → MPO/Drug-likeness →
Synthetic complexity (SA + SCScore) → Similarity/LE/LipE vs reference → Patent FTO
</div>""", unsafe_allow_html=True)
    rows=[
        ("🎯 Predicted Activity","pIC₅₀/IC₅₀ from generative model or docking. ≥10× reference preferred.","0.25"),
        ("💊 MPO / Drug-Likeness","Pfizer CNS-MPO ≥4. QED > 0.6. Lipinski Ro5 pass.","0.20"),
        ("🔬 SA Score","Ertl 2009. ≤4=easy, >6=hard. Based on ring complexity & stereocenters.","0.10"),
        ("⚙️  SCScore","Coley/MIT 2018. 1–5; trained on Reaxys reactions. ≤3 preferred.","0.10"),
        ("🧬 Tanimoto + LE + LipE","Similarity vs ref: analogue >0.6, hop 0.3–0.6. LE>0.3, LipE>5.","0.15"),
        ("🚫 PAINS + Brenk","Hard filters — PAINS and Brenk/REOS alerts. Must CLEAN to proceed.","Hard"),
        ("📜 Patent FTO","Markush R-group novelty vs SciFinder labels. Novel combos preferred.","0.10"),
        ("💰 Synth Cost","Implied by SA+SCScore. Low score = fewer steps, cheaper building blocks.","0.10"),
    ]
    c1,c2=st.columns(2)
    for i,(n,d,w) in enumerate(rows):
        with c1 if i%2==0 else c2:
            st.markdown(f"""<div class="card"><strong style="color:#4fc3f7">{n}</strong>
<div style="color:#546e7a;font-size:.74rem">Weight: {w}</div>
<div style="font-size:.83rem;line-height:1.55;color:#90a4ae;margin-top:4px">{d}</div></div>""",
                        unsafe_allow_html=True)
    st.markdown("""
<div class="info" style="margin-top:10px">
<strong>LE</strong> = 1.37 × pIC₅₀ / N_heavy &nbsp;→&nbsp; target > 0.3 &nbsp;|&nbsp;
<strong>LipE</strong> = pIC₅₀ − cLogP &nbsp;→&nbsp; target > 5<br>
<strong>SCScore 1–5</strong>: MIT neural network trained on Reaxys — products more complex than reactants<br>
<strong>SA Score 1–10</strong>: Ertl complexity score — ring systems, stereocenters, rare fragments
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
TABS = st.tabs([
    "📂 1 · Upload",
    "⚙️  2 · Mode & Params",
    "🧬 3 · Patent Helper",
    "🔬 4 · Properties",
    "📊 5 · MPO & Radar",
    "🏆 6 · Selection",
    "🤖 7 · AI Report",
    "📈 8 · Visual Report",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════
with TABS[0]:
    st.markdown('<span class="badge">Step 1</span>', unsafe_allow_html=True)
    st.markdown("## 📂 Upload Compound Library")

    c_lib, c_ref = st.columns(2)

    with c_lib:
        st.markdown("### Compound Library")
        st.caption("CSV or Excel — must have a SMILES column and an ID column.")
        lf = st.file_uploader("Upload library", type=["csv","xlsx","xls"],
                              key="lib_up", label_visibility="collapsed")
        if lf:
            try:
                df = pd.read_csv(lf) if lf.name.endswith(".csv") else pd.read_excel(lf)
                st.session_state.df = df
                st.success(f"✓ {len(df)} rows · {len(df.columns)} columns")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(str(e))

        if st.session_state.df is not None:
            df = st.session_state.df
            cols = df.columns.tolist()

            st.markdown('<hr class="d">', unsafe_allow_html=True)
            st.markdown("#### Column Configuration")

            cc1, cc2 = st.columns(2)

            # ── SMILES column ──────────────────────────────────────
            smiles_candidates = [c for c in cols if "smiles" in c.lower()]
            smiles_default = smiles_candidates[0] if smiles_candidates else cols[0]
            sc = cc1.selectbox(
                "SMILES column",
                cols,
                index=cols.index(smiles_default),
                key="sc_sel"
            )

            # ── ID / Key column — explicit guidance ───────────────
            cc2.markdown("**Unique ID column** *(must be unique per row)*")
            cc2.caption("This is the key used to join tables throughout the app. "
                        "Pick a column where every row has a different value.")

            # Auto-detect best candidate: prefer cols with 'id','ID','Id' in name
            id_candidates = [c for c in cols
                             if any(k in c for k in ["ID","Id","id","_id","_ID"])]
            # Also try numeric-looking columns
            num_cols_list = df.select_dtypes(include="number").columns.tolist()

            # Rank: exact "Structure_ID" first, then id_candidates, then numeric
            if "Structure_ID" in cols:
                id_default = "Structure_ID"
            elif "Structure ID" in cols:
                id_default = "Structure ID"
            elif id_candidates:
                id_default = id_candidates[0]
            elif num_cols_list:
                id_default = num_cols_list[0]
            else:
                id_default = cols[0]

            ic = cc2.selectbox(
                "Select ID column",
                cols,
                index=cols.index(id_default),
                key="ic_sel",
                label_visibility="collapsed"
            )

            # Validate uniqueness
            n_unique = df[ic].nunique()
            n_rows   = len(df)
            if n_unique == n_rows:
                cc2.markdown(
                    f'<div class="good">✓ All {n_unique} values are unique — good key column.</div>',
                    unsafe_allow_html=True
                )
            else:
                cc2.markdown(
                    f'<div class="warn">⚠ Only {n_unique} unique values across {n_rows} rows. '
                    f'Duplicate IDs will cause join errors downstream. '
                    f'Consider choosing a different column.</div>',
                    unsafe_allow_html=True
                )
                # Show which columns ARE fully unique
                unique_cols = [c for c in cols if df[c].nunique() == n_rows]
                if unique_cols:
                    cc2.caption(f"Fully unique columns: {', '.join(unique_cols[:8])}")

            st.session_state.smiles_col = sc
            st.session_state.id_col     = ic

            if st.button("✅ Parse Molecules", key="parse_btn"):
                with st.spinner("Parsing SMILES…"):
                    df["_Mol"] = df[sc].apply(safe_mol)
                    ok = df["_Mol"].notna()
                    st.session_state.df   = df
                    st.session_state.mols = df.loc[ok,"_Mol"].tolist()
                    st.session_state.ids  = df.loc[ok,ic].astype(str).tolist()
                n_ok = len(st.session_state.mols)
                st.success(f"✓ {n_ok}/{len(df)} molecules parsed  ·  ID column: **{ic}**")
                if n_ok < len(df):
                    st.warning(f"{len(df)-n_ok} rows had invalid SMILES — skipped.")

    with c_ref:
        st.markdown("### Reference Compound *(optional)*")
        st.markdown("""<div class="info">
Select your reference / known compound directly from the uploaded library.
Its property values (docking, ADMET, etc.) are taken straight from the same CSV row —
no separate upload needed. Used for radar plot comparison and MPO benchmarking.
</div>""", unsafe_allow_html=True)

        if st.session_state.df is not None and st.session_state.ids:
            id_c = st.session_state.id_col
            df_now = st.session_state.df
            id_options = ["— None —"] + [str(x) for x in df_now[id_c].tolist()
                                          if pd.notna(x)]
            ref_choice = st.selectbox(
                "Pick reference compound by ID", id_options,
                key="ref_pick_id"
            )
            if ref_choice != "— None —":
                ref_row = df_now[df_now[id_c].astype(str) == ref_choice]
                if len(ref_row):
                    sc2 = st.session_state.smiles_col
                    ref_smi = ref_row[sc2].values[0] if sc2 in ref_row.columns else None
                    ref_mol = safe_mol(ref_smi) if ref_smi else None
                    # Store full row as ref_df so all property columns are available
                    st.session_state.ref_df   = ref_row.copy()
                    st.session_state.ref_mols = [ref_mol] if ref_mol else []
                    if ref_mol:
                        svg = mol_svg(ref_mol, 280, 210)
                        if svg: st.markdown(svg, unsafe_allow_html=True)
                    st.success(f"✓ Reference: **{ref_choice}**")
                    # Show its property values
                    num_cols_ref = ref_row.select_dtypes(include="number").columns.tolist()
                    if num_cols_ref:
                        st.caption("Reference property values:")
                        st.dataframe(ref_row[num_cols_ref[:12]].T.rename(
                            columns={ref_row.index[0]: "Value"}),
                            use_container_width=True)
            else:
                st.session_state.ref_df   = None
                st.session_state.ref_mols = []
        else:
            st.info("⬅ Upload and parse library first, then pick a reference.")

    if st.session_state.mols:
        st.markdown('<hr class="d">', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Library molecules", len(st.session_state.mols))
        m2.metric("Reference compounds", len(st.session_state.ref_mols))
        m3.metric("Columns", len(st.session_state.df.columns) if st.session_state.df is not None else 0)
        st.markdown('<div class="good">✓ Library ready — go to Step 2 to choose analysis mode.</div>',
                    unsafe_allow_html=True)

        # ── 2D Visualizer ────────────────────────────────────────
        st.markdown('<hr class="d">', unsafe_allow_html=True)
        st.markdown("### 🔍 2D Structure Viewer *(optional)*")
        viz_n = st.slider("How many structures to preview", 1, min(50, len(st.session_state.mols)), 9, key="viz_n")
        if st.button("🖼️ Show 2D Structures", key="viz_btn"):
            mols_show = st.session_state.mols[:viz_n]
            ids_show  = st.session_state.ids[:viz_n]
            # Draw in a 3-column grid
            ncols = 3
            rows  = [mols_show[i:i+ncols] for i in range(0, len(mols_show), ncols)]
            irows = [ids_show[i:i+ncols]   for i in range(0, len(ids_show), ncols)]
            for mrow, irow in zip(rows, irows):
                cols_viz = st.columns(ncols)
                for col, mol, mid in zip(cols_viz, mrow, irow):
                    svg = mol_svg(mol, 260, 200)
                    if svg:
                        col.markdown(
                            f'<div style="text-align:center">'
                            f'{svg}'
                            f'<br><small style="color:#78909c">{mid}</small></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        col.warning(f"Could not draw {mid}")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — MODE & PARAMETERS
# ══════════════════════════════════════════════════════════════════
with TABS[1]:
    st.markdown('<span class="badge">Step 2</span>', unsafe_allow_html=True)
    st.markdown("## ⚙️ Choose Analysis Mode & Parameters")

    st.markdown("""
<div class="info">
<strong>Basic Mode</strong> — your CSV already has all required property columns.
The app clusters and plots using your existing columns.<br><br>
<strong>Advanced Mode</strong> — calculate extra cheminformatics properties from SMILES
(SA Score, SCScore, PAINS, Brenk, Lipinski, LE, LipE, QED, stereocenters…).
All computed columns appear in the final output CSV.
</div>""", unsafe_allow_html=True)

    mode = st.radio("Analysis mode",
                    ["🟢 Basic — use my existing columns",
                     "🔵 Advanced — calculate additional properties"],
                    key="mode_radio")
    st.session_state.analysis_mode = "basic" if "Basic" in mode else "advanced"

    st.markdown('<hr class="d">', unsafe_allow_html=True)
    st.markdown("### Columns from your file to carry into the analysis")
    if st.session_state.df is not None:
        all_c = [c for c in st.session_state.df.columns
                 if c not in ["_Mol", st.session_state.smiles_col]]
        ucols = st.multiselect("Select columns",all_c,
                               default=[c for c in all_c if c!=st.session_state.id_col][:min(8,len(all_c))],
                               key="ucols")
        st.session_state.analysis_cols = ucols
    else:
        st.warning("⬅ Upload library in Step 1 first.")

    if st.session_state.analysis_mode == "advanced":
        st.markdown('<hr class="d">', unsafe_allow_html=True)
        st.markdown("### Select Additional Properties to Calculate")
        st.caption("Tick what you need. Results appear in the output CSV.")

        col1,col2,col3 = st.columns(3)
        with col1:
            st.markdown("**Synthetic Complexity**")
            f_sa    = st.checkbox("SA Score (Ertl 2009, 1–10)",    value=True,  key="f_sa")
            f_sc    = st.checkbox("SCScore (Coley/MIT 2018, 1–5)", value=True,  key="f_sc")
            st.markdown("**Drug-Likeness Rules**")
            f_qed   = st.checkbox("QED (Bickerton 2012, 0–1)",     value=True,  key="f_qed")
            f_lip   = st.checkbox("Lipinski Ro5 pass/fail",        value=True,  key="f_lip")
            f_veber = st.checkbox("Veber oral bioavailability",    value=True,  key="f_veber")
            f_egan  = st.checkbox("Egan egg (absorption)",         value=False, key="f_egan")
            f_mueg  = st.checkbox("Muegge drug-like filter",       value=False, key="f_mueg")
            f_ro3   = st.checkbox("Rule of 3 (fragment-like)",     value=False, key="f_ro3")
        with col2:
            st.markdown("**Structural Alerts**")
            f_pains = st.checkbox("PAINS A/B/C (pan-assay interference)",    value=True,  key="f_pains")
            f_brenk = st.checkbox("Brenk toxicophore alerts",                value=True,  key="f_brenk")
            f_halo  = st.checkbox("Halogen alerts (F≥5, I, polyCl/Br)",      value=True,  key="f_halo")
            f_react = st.checkbox("Reactive groups (nitro, aldehyde, epoxide…)", value=True, key="f_react")
            st.markdown("**Structural Descriptors**")
            f_stereo= st.checkbox("Stereocenters count",  value=True,  key="f_stereo")
            f_rings = st.checkbox("Ring counts",          value=True,  key="f_rings")
            f_fsp3  = st.checkbox("Fsp3 (sp3 fraction, Lovering 2009)", value=True, key="f_fsp3")
        with col3:
            st.markdown("**ADMET Estimates**")
            f_esol  = st.checkbox("ESOL solubility — Delaney 2004 LogS", value=True,  key="f_esol")
            f_pka   = st.checkbox("pKa proxy (basic N count + estimate)", value=True,  key="f_pka")
            f_cns   = st.checkbox("CNS/BBB score (0–5)",                  value=False, key="f_cns")
            st.markdown("**Efficiency Metrics** *(need an activity column)*")
            f_le    = st.checkbox("Ligand Efficiency LE = 1.37·pIC50/Nheavy", value=False, key="f_le")
            f_lipe  = st.checkbox("LipE = pIC50 − cLogP",                       value=False, key="f_lipe")
            if f_le or f_lipe:
                if st.session_state.df is not None:
                    act_opts = [c for c in st.session_state.df.columns if c != "_Mol"]
                    st.selectbox("Activity column (pIC50)", act_opts, key="act_col")
            st.markdown("**Similarity**")
            f_sim   = st.checkbox("Tanimoto vs reference",
                                   value=bool(st.session_state.ref_mols), key="f_sim")

        st.session_state.calc_flags = {
            "sa":f_sa,"sc":f_sc,"qed":f_qed,"lip":f_lip,
            "veber":f_veber,"egan":f_egan,"mueg":f_mueg,"ro3":f_ro3,
            "pains":f_pains,"brenk":f_brenk,"halo":f_halo,"react":f_react,
            "stereo":f_stereo,"rings":f_rings,"fsp3":f_fsp3,
            "esol":f_esol,"pka":f_pka,"cns":f_cns,
            "le":f_le,"lipe":f_lipe,"sim":f_sim,
        }
        planned=[k for k,v in st.session_state.calc_flags.items() if v]
        if planned:
            tags="".join(f'<span class="tag">{p.upper()}</span>' for p in planned)
            st.markdown(f"**Will calculate:** {tags}", unsafe_allow_html=True)
    else:
        st.session_state.calc_flags = {}

    st.markdown('<div class="good">✓ Settings saved. Go to Step 3 (Patent Helper) or Step 4 (Properties).</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — PATENT HELPER
# ══════════════════════════════════════════════════════════════════
with TABS[2]:
    st.markdown('<span class="badge">Step 3</span>', unsafe_allow_html=True)
    st.markdown("## 🧬 Patent Helper — Markush Cluster Analysis")
    st.markdown("""
<div class="info">
<strong>Akshay's Markush pipeline:</strong> Murcko scaffold extraction →
Butina clustering (Tanimoto) → R-group decomposition →
Markush summary with <strong>SciFinder labels</strong> (Ra=first position, Rb=second…).<br><br>
The <code>SciFinder_Label</code> column maps each R-group to its label in SciFinder patent
Markush drawings — open the SciFinder record and match directly by label.
</div>""", unsafe_allow_html=True)

    if not RDKIT_OK: st.error("RDKit required.")
    elif not st.session_state.mols: st.warning("⬅ Parse molecules in Step 1 first.")
    else:
        cutoff = st.slider("Butina threshold (Akshay default = 0.4)",0.2,0.8,0.4,0.05)
        if st.button("🚀 Run Markush Analysis", key="markush_btn"):
            mols=st.session_state.mols; ids=st.session_state.ids
            with st.spinner("Clustering and decomposing R-groups…"):
                clusters,_ = run_clustering(mols,cutoff)
                st.session_state.clusters = clusters
                scaf_map={i:safe_scaffold(m) for i,m in enumerate(mols)
                          if safe_scaffold(m) is not None}
                all_mk,all_rg=[],[]
                for i,cluster in enumerate(clusters):
                    cl_mols=[mols[j] for j in cluster if j<len(mols)]
                    cl_ids =[ids[j]  for j in cluster if j<len(ids)]
                    cl_scafs=[scaf_map[j] for j in cluster if j in scaf_map]
                    if not cl_scafs: continue
                    cnt=Counter()
                    for s in cl_scafs:
                        try: cnt[Chem.MolToSmiles(s)]+=1
                        except: pass
                    best=cnt.most_common(1)[0][0] if cnt else None
                    core=Chem.MolFromSmiles(best) if best else cl_scafs[0]
                    rg_df,rg_opts=run_rgroup_decomp(core,cl_mols)
                    dc=None
                    if "Core" in rg_df.columns and rg_df["Core"].notna().any():
                        try: dc=Chem.MolFromSmiles(rg_df["Core"].dropna().iloc[0])
                        except: pass
                    mk_core=dc or core
                    try: rdDepictor.Compute2DCoords(mk_core)
                    except: pass
                    cs=Chem.MolToSmiles(mk_core) if mk_core else ""
                    rg_str=" ; ".join(f"{k}: "+" | ".join(v)
                                      for k,v in sorted(rg_opts.items())) if rg_opts else ""
                    if rg_opts:
                        for rn,vals in sorted(rg_opts.items()):
                            for smi in vals:
                                all_rg.append({"Cluster":i+1,"Cluster_Size":len(cluster),
                                               "RGroup":rn,"RGroup_SMILES":smi,
                                               "SciFinder_Label":rg_scifindr(rn)})
                    else:
                        all_rg.append({"Cluster":i+1,"Cluster_Size":len(cluster),
                                       "RGroup":None,"RGroup_SMILES":None,"SciFinder_Label":"N/A"})
                    all_mk.append({"Cluster":i+1,"Cluster_Size":len(cluster),
                                   "Core_SMILES":cs,"RGroup_Names":", ".join(sorted(rg_opts.keys())),
                                   "RGroups_SMILES":rg_str,"Num_RGroups":len(rg_opts),
                                   "Members":" | ".join(cl_ids)})
                st.session_state.markush_summary=pd.DataFrame(all_mk)
                st.session_state.rgroup_long=pd.DataFrame(all_rg)
            st.success(f"✓ {len(clusters)} scaffold clusters")

        if st.session_state.markush_summary is not None:
            mk=st.session_state.markush_summary; rg=st.session_state.rgroup_long
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Clusters",len(mk)); m2.metric("Top cluster size",mk["Cluster_Size"].max())
            m3.metric("R-group positions",rg["RGroup"].nunique() if "RGroup" in rg else 0)
            m4.metric("SciFinder labels",rg["SciFinder_Label"].nunique() if "SciFinder_Label" in rg else 0)

            # ── Cluster-by-cluster viewer ─────────────────────────
            st.markdown("### 🔍 Per-Cluster Viewer")
            st.caption("Select a cluster to see its Markush definition, core structure, and R-group table.")
            cluster_ids = mk["Cluster"].tolist()
            sel_cluster = st.selectbox("Select cluster", cluster_ids,
                                       format_func=lambda x: f"Cluster {x}  (n={mk.loc[mk['Cluster']==x,'Cluster_Size'].values[0]})",
                                       key="sel_cluster")
            sel_row = mk[mk["Cluster"]==sel_cluster].iloc[0]
            sel_rg  = rg[rg["Cluster"]==sel_cluster] if "Cluster" in rg.columns else pd.DataFrame()

            # Build rg_opts dict for tooltip {R1: [smi1, smi2,...], R2: [...]}
            rg_opts_for_tip = {}
            if len(sel_rg) and "RGroup" in sel_rg.columns:
                for rg_name, grp in sel_rg.groupby("RGroup"):
                    vals = sorted(grp["RGroup_SMILES"].dropna().astype(str).tolist())
                    if vals and vals != ["None"]:
                        rg_opts_for_tip[rg_name] = vals

            col_vis, col_def = st.columns([1, 1])

            # 2D core structure
            with col_vis:
                st.markdown(f"**Core structure — Cluster {sel_cluster}**")
                core_smi = sel_row.get("Core_SMILES","") if hasattr(sel_row,"get") else sel_row["Core_SMILES"]
                core_smi = str(core_smi) if core_smi and str(core_smi) not in ("","nan","None") else ""
                drawn = False
                if core_smi:
                    # Try direct mol first
                    core_mol = safe_mol(core_smi)
                    # If failed, strip * attachment points and try again
                    if core_mol is None:
                        clean_smi = re.sub(r'\[\*[^\]]*\]|\*','c', core_smi)
                        core_mol = safe_mol(clean_smi)
                    if core_mol is None:
                        # Last resort: kekulize off
                        try:
                            core_mol = Chem.MolFromSmiles(core_smi,
                                sanitize=False) if RDKIT_OK else None
                            if core_mol:
                                Chem.SanitizeMol(core_mol,
                                    Chem.SanitizeFlags.SANITIZE_ALL ^
                                    Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        except: core_mol = None
                    if core_mol:
                        try: rdDepictor.Compute2DCoords(core_mol)
                        except: pass
                        # Use tooltip SVG if R-groups available, else plain SVG
                        if rg_opts_for_tip:
                            tip_html = core_svg_with_tooltips(core_mol, rg_opts_for_tip, 380, 280)
                            if tip_html:
                                st.markdown(tip_html, unsafe_allow_html=True)
                                st.caption("💡 Hover over highlighted atoms (blue circles) to see R-group variants")
                                drawn = True
                        if not drawn:
                            svg = mol_svg(core_mol, 320, 250)
                            if svg:
                                st.markdown(
                                    f'<div style="background:white;border-radius:8px;'
                                    f'padding:8px;display:inline-block">{svg}</div>',
                                    unsafe_allow_html=True)
                                drawn = True
                if not drawn:
                    # Fall back: draw most common member of the cluster
                    try:
                        members_str = sel_row.get("Members","") if hasattr(sel_row,"get") else sel_row["Members"]
                        mem_ids = [x.strip() for x in str(members_str).split("|") if x.strip()]
                        if mem_ids and st.session_state.ids:
                            id_to_mol = {str(mid): mol for mid, mol in
                                         zip(st.session_state.ids, st.session_state.mols)}
                            fb_mol = id_to_mol.get(mem_ids[0])
                            if fb_mol:
                                svg = mol_svg(fb_mol, 320, 250)
                                if svg:
                                    st.markdown(
                                        f'<div style="background:white;border-radius:8px;'
                                        f'padding:8px;display:inline-block">{svg}</div>',
                                        unsafe_allow_html=True)
                                    st.caption(f"⚠️ Core scaffold unavailable — showing member {mem_ids[0]}")
                                    drawn = True
                    except: pass
                if not drawn:
                    st.info("Could not render core structure for this cluster.")

            # Markush-like text definition
            with col_def:
                st.markdown("**Markush-like definition**")
                # Build R-group definition lines from rg table
                if len(sel_rg) and "RGroup" in sel_rg.columns:
                    rg_def_lines = []
                    for rg_name, grp in sel_rg.groupby("RGroup"):
                        smis = sorted(grp["RGroup_SMILES"].dropna().tolist())
                        sf_lbl = grp["SciFinder_Label"].iloc[0] if "SciFinder_Label" in grp.columns else rg_name
                        rg_def_lines.append(f"{rg_name} ({sf_lbl}) = {smis}")
                    def_text = "\n".join(rg_def_lines)
                    def_text += f"\n\nMarkush core:\n{core_smi}"
                else:
                    # Fall back to stored string
                    rg_str = sel_row.get("RGroups_SMILES","")
                    def_text = rg_str + f"\n\nMarkush core:\n{core_smi}" if rg_str else f"Markush core:\n{core_smi}"

                st.code(def_text, language="text")

            # R-group table for this cluster
            st.markdown(f"**R-Group Table — Cluster {sel_cluster}**")
            if len(sel_rg):
                # One row per variant — showing SMILES + Common_Repr side by side
                detail_rows = []
                for rg_name, grp in sel_rg.groupby("RGroup"):
                    sf = grp["SciFinder_Label"].iloc[0] if "SciFinder_Label" in grp.columns else ""
                    raw_smis = sorted(grp["RGroup_SMILES"].dropna().astype(str).tolist())
                    for smi in raw_smis:
                        detail_rows.append({
                            "R-Group":         rg_name,
                            "SciFinder_Label": sf,
                            "SMILES":          smi,
                            "Common_Repr":     rg_common_repr(smi),
                        })
                rg_table_df = pd.DataFrame(detail_rows)
                st.dataframe(rg_table_df, use_container_width=True)
                # Download enriched R-group table for this cluster
                st.download_button(
                    "⬇ Download this cluster R-group table (CSV)",
                    rg_table_df.to_csv(index=False).encode(),
                    f"cluster_{sel_cluster}_rgroups.csv", "text/csv",
                    key=f"dl_rg_{sel_cluster}"
                )
            else:
                st.info("No R-group data for this cluster.")

            st.markdown('<hr class="d">', unsafe_allow_html=True)

            # ── Full summary tables ───────────────────────────────
            st.markdown("### 📋 Full Markush Summary Table")
            st.dataframe(mk, use_container_width=True)

            st.markdown("### 📋 R-Group Table with SciFinder Labels")
            st.markdown("""<div class="info">
<code>SciFinder_Label</code>: Ra = 1st position · Rb = 2nd · Rc = 3rd…<br>
Match these labels directly to the Markush drawing in SciFinder patent records.
</div>""", unsafe_allow_html=True)
            st.dataframe(rg, use_container_width=True)

            # Cluster size chart
            fig,ax=dfig((10,3.5))
            ax.bar(mk["Cluster"].astype(str),mk["Cluster_Size"],
                   color="#4fc3f7",edgecolor="#1e2d3d",linewidth=0.5)
            ax.set_xlabel("Cluster"); ax.set_ylabel("Compounds")
            ax.set_title("Cluster Size Distribution")
            st.pyplot(fig,use_container_width=True); plt.close()

            c1,c2=st.columns(2)
            c1.download_button("⬇ Markush Summary CSV",mk.to_csv(index=False).encode(),"markush_summary.csv","text/csv")
            # Full R-group CSV — one row per variant, both SMILES and Common_Repr
            rg_export = rg.copy()
            if "RGroup_SMILES" in rg_export.columns:
                rg_export["Common_Repr"] = rg_export["RGroup_SMILES"].apply(rg_common_repr)
                # Reorder so Common_Repr sits right after SMILES
                cols = list(rg_export.columns)
                smi_idx = cols.index("RGroup_SMILES")
                cols.remove("Common_Repr")
                cols.insert(smi_idx + 1, "Common_Repr")
                rg_export = rg_export[cols]
            c2.download_button("⬇ R-Group Table (SciFinder Labels)",rg_export.to_csv(index=False).encode(),"rgroups_scifindr.csv","text/csv")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — PROPERTY CALCULATOR
# ══════════════════════════════════════════════════════════════════
with TABS[3]:
    st.markdown('<span class="badge">Step 4</span>', unsafe_allow_html=True)
    st.markdown("## 🔬 Property Calculator")

    mode = st.session_state.analysis_mode
    flags = st.session_state.calc_flags

    if mode == "basic":
        sel_cols = st.session_state.analysis_cols
        if not sel_cols:
            st.markdown("""<div class="warn">
⬅ No columns selected yet. Go back to Step 2 and choose the columns you want to carry forward.
</div>""", unsafe_allow_html=True)
        else:
            tags = "".join(f'<span class="tag">{c}</span>' for c in sel_cols)
            st.markdown(f"""<div class="info">
<strong>Basic mode</strong> — using your selected columns as-is. No properties will be
recalculated from SMILES.<br><br>
<strong>Columns carrying forward:</strong> {tags}
</div>""", unsafe_allow_html=True)

            also_core = st.checkbox(
                "Also calculate core descriptors from SMILES (MW, LogP, HBD, HBA, TPSA, RotBonds)",
                value=False, key="basic_also_core"
            )

            if st.session_state.mols and st.button("▶ Build Property Table", key="basic_desc"):
                with st.spinner("Assembling table…"):
                    df_raw = st.session_state.df
                    id_c   = st.session_state.id_col
                    # Start from selected columns
                    keep = [id_c] + [c for c in sel_cols if c in df_raw.columns and c != id_c]
                    pdf  = df_raw[keep].copy()
                    pdf.rename(columns={id_c: "Structure_ID"}, inplace=True)
                    pdf["Structure_ID"] = pdf["Structure_ID"].astype(str)

                    if also_core:
                        mol_map = {str(mid): mol for mid, mol in
                                   zip(st.session_state.ids, st.session_state.mols)}
                        core_recs = []
                        for mid in pdf["Structure_ID"]:
                            mol = mol_map.get(mid)
                            r = {"Structure_ID": mid}
                            if mol:
                                try:
                                    r["MW"]       = round(Descriptors.ExactMolWt(mol), 2)
                                    r["LogP"]     = round(Descriptors.MolLogP(mol), 2)
                                    r["HBD"]      = rdMolDescriptors.CalcNumHBD(mol)
                                    r["HBA"]      = rdMolDescriptors.CalcNumHBA(mol)
                                    r["TPSA"]     = round(Descriptors.TPSA(mol), 2)
                                    r["RotBonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                                    r["HeavyAtoms"]= mol.GetNumHeavyAtoms()
                                except: pass
                            core_recs.append(r)
                        core_df = pd.DataFrame(core_recs)
                        pdf = pdf.merge(core_df, on="Structure_ID", how="left")

                    st.session_state.prop_df = pdf
                st.success(f"✓ Table ready — {len(pdf)} compounds, {len(pdf.columns)} columns")
                st.dataframe(pdf, use_container_width=True)
    else:
        planned=[k for k,v in flags.items() if v]
        st.markdown("""<div class="info">
<strong>Advanced mode.</strong> Every selected property will be computed and added
to the output CSV as new columns.
</div>""", unsafe_allow_html=True)
        if planned:
            tags="".join(f'<span class="tag">{p.upper()}</span>' for p in planned)
            st.markdown(f"**Queued:** {tags}", unsafe_allow_html=True)
        else:
            st.info("No extra properties selected. Go back to Step 2 to tick some.")

        if not st.session_state.mols:
            st.warning("⬅ Parse molecules in Step 1 first.")
        elif st.button("🔬 Calculate All Selected Properties", key="calc_btn"):
            mols=st.session_state.mols; ids=st.session_state.ids
            ref0=st.session_state.ref_mols[0] if st.session_state.ref_mols else None
            act_col=st.session_state.get("act_col",None)
            prog=st.progress(0); stat=st.empty()
            recs=[]
            for idx,(mid,mol) in enumerate(zip(ids,mols)):
                r={"Structure_ID":mid}
                prog.progress((idx+1)/len(mols))
                stat.caption(f"{idx+1}/{len(mols)}: {mid}")
                # Always: core descriptors
                try:
                    mw=Descriptors.ExactMolWt(mol); lp=Descriptors.MolLogP(mol)
                    hbd=rdMolDescriptors.CalcNumHBD(mol); hba=rdMolDescriptors.CalcNumHBA(mol)
                    tpsa=Descriptors.TPSA(mol); rot=rdMolDescriptors.CalcNumRotatableBonds(mol)
                    na=mol.GetNumHeavyAtoms()
                    r.update({"MW":round(mw,2),"LogP":round(lp,2),"HBD":hbd,"HBA":hba,
                              "TPSA":round(tpsa,2),"RotBonds":rot,"HeavyAtoms":na})
                except: pass
                if flags.get("sa"):    r["SA_Score"] = sa_score(mol)
                if flags.get("sc"):    r["SCScore"]  = sc_score(mol)
                if flags.get("qed"):   r["QED"]      = qed_score(mol)
                if flags.get("fsp3"):  r["Fsp3"]     = fsp3(mol)
                if flags.get("lip"):
                    ok,v,det=lipinski_ro5(mol)
                    r["Lipinski_Pass"]=("PASS" if ok else "FAIL") if ok is not None else "N/A"
                    r["Lipinski_Violations"]=v
                if flags.get("veber"):
                    vp, vd = veber_rules(mol)
                    r["Veber_Pass"] = vp; r["Veber_Detail"] = vd
                if flags.get("egan"):
                    ep, ed = egan_egg(mol)
                    r["Egan_Pass"] = ep; r["Egan_Detail"] = ed
                if flags.get("mueg"):
                    mp, md = muegge_filter(mol)
                    r["Muegge_Pass"] = mp; r["Muegge_Violations"] = md
                if flags.get("ro3"):
                    rp, rd = rule_of_3(mol)
                    r["Ro3_Pass"] = rp; r["Ro3_Detail"] = rd
                if flags.get("pains"):
                    ip,pd_=check_pains(mol)
                    r["PAINS_Flag"]="PAINS" if ip else ("CLEAN" if ip is False else "N/A")
                    r["PAINS_Description"]=pd_ if ip else ""
                if flags.get("brenk"):
                    ib,bd=check_brenk(mol)
                    r["Brenk_Alert"]="ALERT" if ib else ("CLEAN" if ib is False else "N/A")
                    r["Brenk_Description"]=bd if ib else ""
                if flags.get("halo"):
                    r["F_Count"]      = fluorine_count(mol)
                    r["Halogen_Count"]= halogen_count(mol)
                    hf, hd = halogen_alert(mol)
                    r["Halogen_Alert"]= ("ALERT" if hf else "CLEAN") if hf is not None else "N/A"
                    r["Halogen_Detail"]= hd
                if flags.get("react"):
                    na_, rd_ = reactive_alerts(mol)
                    r["Reactive_Alerts"] = na_
                    r["Reactive_Detail"]  = rd_
                if flags.get("esol"):
                    ls, lcat = esol_logS(mol)
                    r["ESOL_LogS"]     = ls
                    r["Solubility_Cat"]= lcat
                if flags.get("pka"):
                    nb_, pr_, pf_ = pka_proxy(mol)
                    r["Basic_N_Count"] = nb_
                    r["pKa_Estimate"]  = pr_
                    r["pKa_Flag"]      = pf_
                if flags.get("cns"):
                    r["CNS_BBB_Score"] = cns_bbb_score(mol)
                if flags.get("stereo"): r["Stereocenters"]=n_stereo(mol)
                if flags.get("rings"):
                    try:
                        r["NumRings"]=rdMolDescriptors.CalcNumRings(mol)
                        r["NumAromaticRings"]=rdMolDescriptors.CalcNumAromaticRings(mol)
                    except: pass
                if flags.get("sim") and ref0:
                    r["Tanimoto_RefSim"]=tanimoto(mol,ref0)
                if (flags.get("le") or flags.get("lipe")) and act_col:
                    orig=st.session_state.df
                    ic2=st.session_state.id_col
                    row_df=orig[orig[ic2].astype(str)==str(mid)] if orig is not None else pd.DataFrame()
                    if len(row_df) and act_col in row_df.columns:
                        try:
                            av=float(row_df[act_col].values[0])
                            if flags.get("le"):   r["LE"]=le(av,r.get("HeavyAtoms"))
                            if flags.get("lipe"): r["LipE"]=lipe(av,r.get("LogP"))
                        except: pass
                recs.append(r)
            prog.empty(); stat.empty()
            pdf=pd.DataFrame(recs)
            # Merge user-selected extra columns
            if st.session_state.analysis_cols and st.session_state.df is not None:
                ic2=st.session_state.id_col
                ecols=[c for c in st.session_state.analysis_cols
                       if c in st.session_state.df.columns and c!=ic2]
                if ecols:
                    extra=st.session_state.df[[ic2]+ecols].copy()
                    extra[ic2]=extra[ic2].astype(str)
                    pdf["Structure_ID"]=pdf["Structure_ID"].astype(str)
                    pdf=pdf.merge(extra,left_on="Structure_ID",right_on=ic2,how="left")
                    if ic2!="Structure_ID" and ic2 in pdf.columns:
                        pdf.drop(columns=[ic2],inplace=True,errors="ignore")
            st.session_state.prop_df=pdf
            st.success(f"✓ Properties computed for {len(pdf)} compounds")

        if st.session_state.prop_df is not None:
            pdf=st.session_state.prop_df
            st.dataframe(pdf,use_container_width=True)
            # Flag summaries
            flag_c=[c for c in ["PAINS_Flag","Brenk_Alert","Lipinski_Pass"] if c in pdf.columns]
            if flag_c:
                st.markdown("### Structural Filter Summary")
                fc=st.columns(len(flag_c))
                for fcol,fax in zip(flag_c,fc):
                    vc=pdf[fcol].value_counts()
                    fax.markdown(f"**{fcol}**")
                    fax.dataframe(vc.reset_index(),use_container_width=True)
            # Distributions
            dc=[c for c in ["SA_Score","SCScore","QED","Tanimoto_RefSim"] if c in pdf.columns]
            if dc:
                st.markdown("### Score Distributions")
                n=len(dc)
                fig,axes=plt.subplots(1,n,figsize=(4*n,3.5),facecolor="#0a0e17")
                if n==1: axes=[axes]
                th={"SA_Score":(4,"≤4 easy","#a5d6a7"),
                    "SCScore":(3,"≤3 easy","#80deea"),
                    "QED":(0.6,">0.6","#ce93d8"),
                    "Tanimoto_RefSim":(0.4,">0.4","#ffcc02")}
                for ax,col in zip(axes,dc):
                    dark_ax(fig,ax)
                    v=pdf[col].dropna()
                    ax.hist(v,bins=max(6,len(v)//5),color=_C[0],edgecolor="#1e2d3d",alpha=0.85)
                    if col in th:
                        tv,tl,tc=th[col]
                        ax.axvline(tv,color=tc,lw=2,ls="--",label=tl)
                        ax.legend(fontsize=7,labelcolor="#cdd9e5",
                                  facecolor="#111827",edgecolor="#1e2d3d")
                    ax.set_title(col,fontsize=9)
                    ax.set_xlabel("Value"); ax.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig,use_container_width=True); plt.close()
            st.download_button("⬇ Download Property Table CSV",
                               pdf.to_csv(index=False).encode(),"properties.csv","text/csv")

# ══════════════════════════════════════════════════════════════════
# TAB 5 — MPO & RADAR
# ══════════════════════════════════════════════════════════════════
with TABS[4]:
    st.markdown('<span class="badge">Step 5</span>', unsafe_allow_html=True)
    st.markdown("## 📊 MPO Scoring & Radar Plots")

    with st.expander("📖 How MPO is calculated here", expanded=False):
        st.markdown("""<div class="info">
<strong>MPO = Multi-Parameter Optimization (Desirability Function approach)</strong><br><br>
Each column is converted to a 0→1 desirability score using thresholds you define:<br>
• <strong>Lower-is-better</strong> (e.g. Docking): perfect ≤ lower_threshold, zero ≥ upper_threshold<br>
• <strong>Higher-is-better</strong> (e.g. Caco-2, Papp): perfect ≥ upper_threshold, zero ≤ lower_threshold<br><br>
The final MPO = mean of all desirabilities. You can also apply weights.<br><br>
The reference compound is shown on every radar so you can see exactly where your generated
molecules beat or fall short of your benchmark.
</div>""", unsafe_allow_html=True)

    if not st.session_state.mols:
        st.warning("⬅ Parse molecules in Step 1 first.")
    else:
        df_src_full = (st.session_state.prop_df if st.session_state.prop_df is not None
                       else st.session_state.df)
        if df_src_full is None:
            st.warning("⬅ Upload library in Step 1 first.")
        else:
            # ── Cluster filter (like notebook: analyse per cluster) ───
            clusters = st.session_state.clusters
            if clusters:
                st.markdown("### 0 · Scope — whole library or one cluster?")
                st.markdown("""<div class="info">
MPO ranking and top compound selection are done
<strong>per cluster</strong> — so you get one best candidate per scaffold series.
Select a cluster below, or choose "Whole library" to rank all compounds together.
</div>""", unsafe_allow_html=True)

                cluster_opts = ["🌐 Whole library"] + [
                    f"Cluster {i+1}  (n={len(cl)}  · members: "
                    + ", ".join(str(st.session_state.ids[j])
                                for j in cl[:4] if j < len(st.session_state.ids))
                    + ("…" if len(cl) > 4 else ")")
                    for i, cl in enumerate(clusters)
                ]
                scope = st.selectbox("Scope", cluster_opts, key="mpo_scope")

                if scope == "🌐 Whole library":
                    df_src = df_src_full.copy()
                    active_ids = st.session_state.ids
                    scope_label = "Whole library"
                else:
                    cl_idx = int(scope.split()[1]) - 1
                    cl_member_indices = clusters[cl_idx]
                    active_ids = [st.session_state.ids[j]
                                  for j in cl_member_indices
                                  if j < len(st.session_state.ids)]
                    src_id_tmp = ("Structure_ID" if "Structure_ID" in df_src_full.columns
                                  else st.session_state.id_col)
                    df_src = df_src_full[
                        df_src_full[src_id_tmp].astype(str).isin([str(x) for x in active_ids])
                    ].copy()
                    scope_label = f"Cluster {cl_idx+1} ({len(active_ids)} compounds)"

                st.markdown(
                    f'<div class="good">✓ Scope: <strong>{scope_label}</strong> '
                    f'— {len(df_src)} compounds included in MPO calculation.</div>',
                    unsafe_allow_html=True
                )
            else:
                df_src = df_src_full.copy()
                active_ids = st.session_state.ids
                scope_label = "Whole library"
                st.info("ℹ Run Patent Helper (Step 3) first to enable per-cluster analysis.")

            src_id = ("Structure_ID" if "Structure_ID" in df_src.columns
                      else st.session_state.id_col)
            num_cols = [c for c in df_src.select_dtypes(include=[np.number]).columns
                        if c not in ["_Mol"] and df_src[c].notna().sum() > 1]

            st.markdown("### 1 · Select columns & MPO mode")

            mpo_mode = st.radio(
                "MPO calculation mode",
                ["⚡ Simple (min-max method)",
                 "🎯 Advanced (custom desirability thresholds)"],
                key="mpo_mode",
                horizontal=True
            )

            if mpo_mode.startswith("⚡"):
                st.markdown("""<div class="info">
<strong>Simple mode</strong> — exact same approach as your suggestion:<br>
<code>norm = (max − value) / (max − min)</code>  for lower-is-better columns (docking)<br>
<code>norm = (value − min) / (max − min)</code>  for higher-is-better columns (Caco-2, Papp)<br>
<code>MPO  = mean of all normalised columns</code><br><br>
Best value in your library = 1.0 &nbsp;·&nbsp; Worst = 0.0 &nbsp;·&nbsp; No thresholds to set.
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="info">
<strong>Advanced mode</strong> — you define what "perfect" and "zero" means for each column
using absolute thresholds (e.g. Caco-2 ≥ 15 = perfect, ≤ 5 = zero).
More meaningful when you have literature benchmarks for your targets.
</div>""", unsafe_allow_html=True)

            mpo_cols = st.multiselect(
                "Columns for MPO",
                num_cols,
                default=[c for c in ["docking score","Docking_score","CH42_Caco2_Papp_reg",
                                      "CH42_MDCK-MDR1_Papp_reg","CH42_PAMPA_Pe_reg",
                                      "CH42_Caco2_ER"] if c in num_cols]
                         or num_cols[:min(4, len(num_cols))],
                key="mpo_cols"
            )

            thresh_cfg = {}
            if mpo_cols:
                # Simple mode: just set direction per column, no threshold inputs
                if mpo_mode.startswith("⚡"):
                    st.markdown("**Direction per column** *(auto-detected — change if wrong)*")
                    dir_cols = st.columns(min(4, len(mpo_cols)))
                    for ci, col in enumerate(mpo_cols):
                        col_vals = df_src[col].dropna().astype(float)
                        auto_lower = any(k in col.lower() for k in
                                         ["dock","score","ic50","ki","ec50","energy"])
                        direction = dir_cols[ci % len(dir_cols)].radio(
                            col, ["Higher ↑", "Lower ↓"],
                            index=1 if auto_lower else 0,
                            key=f"dir_s_{col}"
                        )
                        cmin, cmax = float(col_vals.min()), float(col_vals.max())
                        thresh_cfg[col] = {
                            "direction": "Lower = better" if direction == "Lower ↓" else "Higher = better",
                            "lo": cmin, "hi": cmax, "w": 1.0,
                            "simple": True   # flag: use true min-max
                        }
                else:
                    # Advanced mode: full threshold UI
                    for col in mpo_cols:
                        col_vals = df_src[col].dropna().astype(float)
                        cmin, cmax = float(col_vals.min()), float(col_vals.max())
                        st.markdown(f"**{col}**  *(library range: {cmin:.3f} → {cmax:.3f})*")
                        tc1, tc2, tc3, tc4 = st.columns([1.2, 1.5, 1.5, 1.0])
                        direction = tc1.radio(
                            "Direction", ["Higher = better", "Lower = better"],
                            key=f"dir_{col}",
                            index=1 if any(k in col.lower() for k in
                                           ["dock","score","ic50","ki","ec50"]) else 0
                        )
                        lo_def = round(cmin + (cmax-cmin)*0.25, 3)
                        hi_def = round(cmin + (cmax-cmin)*0.75, 3)
                        lo = tc2.number_input("Zero threshold (worst)", value=lo_def,
                                              key=f"lo_{col}", format="%.3f")
                        hi = tc3.number_input("Perfect threshold (best)", value=hi_def,
                                              key=f"hi_{col}", format="%.3f")
                        w  = tc4.number_input("Weight", value=1.0, min_value=0.1,
                                              max_value=5.0, step=0.1, key=f"w_{col}")
                        thresh_cfg[col] = {"direction": direction, "lo": lo, "hi": hi,
                                           "w": w, "simple": False}
                        st.markdown('<hr class="d">', unsafe_allow_html=True)

            if mpo_cols and len(mpo_cols) >= 1:
                if st.button("▶ Calculate MPO Scores", key="mpo_btn"):
                    with st.spinner("Calculating MPO scores…"):
                        # Pre-compute library-wide min/max for simple min-max mode
                        col_stats = {}
                        for col in mpo_cols:
                            vals = df_src[col].dropna().astype(float)
                            col_stats[col] = {"min": vals.min(), "max": vals.max()}

                        norm_recs = []
                        for mid in active_ids:
                            row_data = df_src[df_src[src_id].astype(str)==str(mid)]
                            rec = {"Structure_ID": str(mid)}
                            desirs = []
                            for col, cfg in thresh_cfg.items():
                                try:
                                    v = float(row_data[col].values[0]) if len(row_data) else np.nan
                                except:
                                    v = np.nan
                                rec[col] = v
                                if pd.isna(v):
                                    d = np.nan
                                elif cfg.get("simple", False):
                                    # ── Simple mode: pure min-max (notebook approach) ──
                                    cmin = col_stats[col]["min"]
                                    cmax = col_stats[col]["max"]
                                    rng = cmax - cmin if cmax != cmin else 1.0
                                    if cfg["direction"] == "Lower = better":
                                        d = np.clip((cmax - v) / rng, 0, 1)
                                    else:
                                        d = np.clip((v - cmin) / rng, 0, 1)
                                else:
                                    # ── Advanced mode: user-defined thresholds ──
                                    lo, hi = cfg["lo"], cfg["hi"]
                                    lo2, hi2 = min(lo, hi), max(lo, hi)
                                    rng = hi2 - lo2 if hi2 != lo2 else 1.0
                                    if cfg["direction"] == "Higher = better":
                                        d = np.clip((v - lo2) / rng, 0, 1)
                                    else:
                                        if v <= lo2: d = 1.0
                                        elif v >= hi2: d = 0.0
                                        else: d = np.clip((hi2 - v) / rng, 0, 1)
                                rec[f"d_{col}"] = round(d, 4) if not pd.isna(d) else np.nan
                                desirs.append((d, cfg["w"]))
                            valid = [(d, w) for d, w in desirs if not pd.isna(d)]
                            if valid:
                                wsum = sum(w for _, w in valid)
                                rec["MPO_Score"] = round(sum(d*w for d,w in valid)/wsum, 4)
                            else:
                                rec["MPO_Score"] = np.nan
                            norm_recs.append(rec)
                        mpo_df = pd.DataFrame(norm_recs)
                        st.session_state.mpo_df = mpo_df
                    st.success(f"✓ MPO calculated for {len(mpo_df)} compounds")

                if st.session_state.mpo_df is not None:
                    mpo_df = st.session_state.mpo_df
                    d_cols = [f"d_{c}" for c in mpo_cols if f"d_{c}" in mpo_df.columns]

                    if "MPO_Score" not in mpo_df.columns:
                        st.info("Click '▶ Calculate MPO Scores' above.")
                    else:
                        vals = mpo_df["MPO_Score"].dropna()
                        m1,m2,m3,m4 = st.columns(4)
                        m1.metric("Mean MPO",   f"{vals.mean():.3f}")
                        m2.metric("Best MPO",   f"{vals.max():.3f}")
                        m3.metric("Top 25%",    int((vals >= vals.quantile(0.75)).sum()))
                        m4.metric("Median MPO", f"{vals.median():.3f}")

                        # Ranked table
                        st.markdown("### 2 · Ranked by MPO Score")
                        show_c = ["Structure_ID"] + mpo_cols + d_cols + ["MPO_Score"]
                        show_c = [c for c in show_c if c in mpo_df.columns]
                        ranked = mpo_df[show_c].sort_values("MPO_Score", ascending=False).reset_index(drop=True)
                        ranked.insert(0, "Rank", range(1, len(ranked)+1))
                        st.dataframe(ranked, use_container_width=True)

                        # Bar chart
                        st.markdown("### 3 · MPO Score Bar Chart")
                        fig, ax = dfig((12, 3.8))
                        top30 = ranked.head(30)
                        colors = plt.cm.RdYlGn(top30["MPO_Score"].values)
                        ax.bar(top30["Structure_ID"].astype(str), top30["MPO_Score"],
                               color=colors, edgecolor="#1e2d3d", linewidth=0.4)
                        ax.axhline(vals.mean(), color="#ffcc02", lw=1.8, ls="--",
                                   label=f"Mean = {vals.mean():.3f}")
                        ax.set_xlabel("Compound"); ax.set_ylabel("MPO Score (0–1)")
                        ax.set_title("MPO Ranking — top 30  (green = best)")
                        ax.set_xticklabels(top30["Structure_ID"].astype(str),
                                           rotation=90, fontsize=7)
                        ax.legend(labelcolor="#cdd9e5", facecolor="#111827", edgecolor="#1e2d3d")
                        st.pyplot(fig, use_container_width=True); plt.close()

                        # ── Radar Section ─────────────────────────
                        st.markdown("### 4 · Radar / Spider Plots")

                        has_ref = (st.session_state.ref_df is not None and
                                   len(st.session_state.ref_df) > 0)
                        radar_labels = [c[:16] for c in mpo_cols
                                        if f"d_{c}" in mpo_df.columns]
                        d_cols_r = [f"d_{c}" for c in mpo_cols if f"d_{c}" in mpo_df.columns]

                        # Build reference desirability row from same thresholds
                        ref_d_row = None
                        ref_id_label = None
                        if has_ref:
                            rdf = st.session_state.ref_df
                            src_id_ref = ("Structure_ID" if "Structure_ID" in rdf.columns
                                          else st.session_state.id_col)
                            ref_label_val = (rdf[src_id_ref].values[0]
                                             if src_id_ref in rdf.columns else "Reference")
                            ref_id_label = str(ref_label_val)
                            rv = []
                            for col in mpo_cols:
                                if f"d_{col}" not in mpo_df.columns: continue
                                try:
                                    v = float(rdf[col].values[0]) if col in rdf.columns else np.nan
                                except: v = np.nan
                                if pd.isna(v):
                                    rv.append(0.0)
                                    continue
                                cfg = thresh_cfg[col]
                                lo2, hi2 = min(cfg["lo"],cfg["hi"]), max(cfg["lo"],cfg["hi"])
                                if cfg["direction"] == "Higher = better":
                                    d = np.clip((v - cfg["lo"])/(cfg["hi"]-cfg["lo"] or 1), 0, 1)
                                else:
                                    if v <= hi2: d = 1.0
                                    elif v >= lo2: d = 0.0
                                    else: d = float(np.clip((lo2-v)/(lo2-hi2 or 1), 0, 1))
                                rv.append(round(d, 4))
                            if any(x > 0 for x in rv):
                                ref_d_row = rv

                        radar_mode_opts = [
                            "📊 Library mean",
                            "🔝 Top cluster mean",
                            "🎯 Pick specific compounds",
                            "⭐ Reference vs top hit",
                            "🌐 All compounds + reference",
                        ]
                        if not has_ref or ref_d_row is None:
                            radar_mode_opts = [o for o in radar_mode_opts
                                               if "Reference" not in o and "reference" not in o]

                        rmode = st.radio("Radar mode", radar_mode_opts,
                                         key="radar_mode", horizontal=False)

                        sv, sn = [], []

                        if "Library mean" in rmode:
                            sv.append([mpo_df[dc].mean() for dc in d_cols_r])
                            sn.append("Library mean")
                            if ref_d_row:
                                sv.append(ref_d_row); sn.append(f"⭐ {ref_id_label}")

                        elif "Top cluster" in rmode:
                            if st.session_state.clusters:
                                top_cl = st.session_state.clusters[0]
                                tids = [st.session_state.ids[j]
                                        for j in top_cl if j < len(st.session_state.ids)]
                                sub = mpo_df[mpo_df["Structure_ID"].isin([str(x) for x in tids])]
                                if len(sub):
                                    sv.append([sub[dc].mean() for dc in d_cols_r])
                                    sn.append(f"Top cluster mean (n={len(sub)})")
                                    for _, row in sub.sort_values("MPO_Score",ascending=False).head(4).iterrows():
                                        sv.append([row.get(dc,0) for dc in d_cols_r])
                                        sn.append(str(row["Structure_ID"])[:16])
                            else:
                                st.info("Run Patent Helper (Step 3) to get clusters.")
                            if ref_d_row:
                                sv.append(ref_d_row); sn.append(f"⭐ {ref_id_label}")

                        elif "Pick specific" in rmode:
                            pick = st.multiselect(
                                "Select compounds",
                                mpo_df["Structure_ID"].astype(str).tolist(),
                                default=mpo_df.sort_values("MPO_Score",ascending=False)
                                          ["Structure_ID"].astype(str).tolist()[:4],
                                key="radar_pick"
                            )
                            for pid in pick:
                                r = mpo_df[mpo_df["Structure_ID"].astype(str)==pid]
                                if len(r):
                                    sv.append([r[dc].values[0] for dc in d_cols_r])
                                    sn.append(pid)
                            if ref_d_row:
                                sv.insert(0, ref_d_row)
                                sn.insert(0, f"⭐ {ref_id_label}")

                        elif "Reference vs top hit" in rmode:
                            # Reference first, then top N by MPO
                            if ref_d_row:
                                sv.append(ref_d_row); sn.append(f"⭐ {ref_id_label}")
                            n_top = st.slider("How many top hits to compare", 1, 8, 3,
                                              key="n_top_radar")
                            for _, row in mpo_df.sort_values("MPO_Score",ascending=False).head(n_top).iterrows():
                                sv.append([row.get(dc,0) for dc in d_cols_r])
                                sn.append(f"#{_+1} {str(row['Structure_ID'])[:14]}" if False
                                          else str(row["Structure_ID"])[:18])

                        elif "All compounds" in rmode:
                            if ref_d_row:
                                sv.append(ref_d_row); sn.append(f"⭐ {ref_id_label}")
                            for _, row in mpo_df.sort_values("MPO_Score",ascending=False).iterrows():
                                sv.append([row.get(dc,0) for dc in d_cols_r])
                                sn.append(str(row["Structure_ID"])[:14])
                            st.caption(f"Plotting all {len(mpo_df)} compounds "
                                       f"— reference shown first in bold blue.")

                        if sv and d_cols_r:
                            # Determine which series is the reference (grey dashed)
                            ref_series_idx = 0
                            if has_ref and ref_id_label:
                                for si, nm in enumerate(sn[:16]):
                                    if ref_id_label in nm or nm in ref_id_label:
                                        ref_series_idx = si
                                        break

                            # Add MPO score to legend names
                            sn_labelled = []
                            for nm in sn[:16]:
                                hit_row = mpo_df[mpo_df["Structure_ID"].astype(str)==nm]
                                if len(hit_row) and "MPO_Score" in hit_row.columns:
                                    mpo_v = hit_row["MPO_Score"].values[0]
                                    sn_labelled.append(f"{nm[:20]} (MPO: {mpo_v:.2f})")
                                else:
                                    sn_labelled.append(nm[:24])

                            rc, ri = st.columns([1.2, 0.8])
                            with rc:
                                fig = radar_fig(
                                    radar_labels,
                                    sv[:16],
                                    sn_labelled[:16],
                                    title="MPO Radar Plot",
                                    ref_idx=ref_series_idx
                                )
                                if fig:
                                    st.pyplot(fig, use_container_width=True)
                                    # Download button — white bg, 300 dpi
                                    buf_r = io.BytesIO()
                                    fig.savefig(buf_r, format="png", dpi=300,
                                                bbox_inches="tight", facecolor="white")
                                    buf_r.seek(0)
                                    st.download_button("⬇ Download Radar (PNG)",
                                                       buf_r, file_name="mpo_radar.png",
                                                       mime="image/png",
                                                       key="dl_radar_tab5")
                                    plt.close(fig)
                                else:
                                    st.info("Need ≥ 2 MPO columns to draw radar. "
                                            "Add more columns in Step 5.")
                            with ri:
                                st.markdown("**MPO scores:**")
                                for nm, vv in zip(sn[:16], sv[:16]):
                                    hit_row = mpo_df[mpo_df["Structure_ID"].astype(str)==nm]
                                    mpo_val = (hit_row["MPO_Score"].values[0]
                                               if len(hit_row) else sum(vv)/max(len(vv),1))
                                    badge = "⭐ Ref" if nm == ref_id_label else "  "
                                    st.markdown(f"{badge} `{nm[:24]}` → **{mpo_val:.3f}**")

                        # Parallel coordinates
                        st.markdown("### 5 · Parallel Coordinates")
                        sub_pc = mpo_df.sort_values("MPO_Score",ascending=False).head(50)
                        fig, ax = dfig((12, 3.8))
                        cmap = plt.cm.RdYlGn
                        vmn, vmx = sub_pc["MPO_Score"].min(), sub_pc["MPO_Score"].max()
                        x = np.arange(len(d_cols_r))
                        for _, row in sub_pc.iterrows():
                            y = [row.get(dc, 0) for dc in d_cols_r]
                            nv = (row["MPO_Score"]-vmn)/(vmx-vmn+1e-9)
                            ax.plot(x, y, alpha=0.4, lw=1.2, color=cmap(nv))
                        if ref_d_row:
                            ax.plot(x, ref_d_row, lw=3, color="#4fc3f7",
                                    label=f"⭐ {ref_id_label}", zorder=5)
                        ax.set_xticks(x)
                        ax.set_xticklabels(radar_labels, color="#78909c", fontsize=9)
                        ax.set_ylim(-0.05, 1.08)
                        ax.set_ylabel("Desirability (0→1 = perfect)")
                        ax.set_title("Parallel coordinates — colour = MPO score")
                        if ref_d_row:
                            ax.legend(labelcolor="#cdd9e5", facecolor="#111827",
                                      edgecolor="#1e2d3d", fontsize=8)
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmn,vmx))
                        sm.set_array([])
                        cb = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01)
                        cb.set_label("MPO Score", color="#78909c", fontsize=8)
                        cb.ax.yaxis.set_tick_params(color="#78909c")
                        st.pyplot(fig, use_container_width=True); plt.close()

                        st.download_button("⬇ Download MPO Table CSV",
                                           mpo_df.to_csv(index=False).encode(),
                                           "mpo_scores.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════
# TAB 6 — FINAL SELECTION & SYNTHESIS RECOMMENDATION
# ══════════════════════════════════════════════════════════════════
with TABS[5]:
    st.markdown('<span class="badge">Step 6</span>', unsafe_allow_html=True)
    st.markdown("## 🏆 Final Selection & Synthesis Recommendation")
    st.markdown("""<div class="info">
The goal: <strong>which molecule should be synthesised first?</strong><br>
This step ranks all compounds by MPO score, shows similarity to your reference,
and gives a clear go/no-go recommendation for the top candidates.
</div>""", unsafe_allow_html=True)

    # Build merged frame from whatever has been computed
    frames = []
    if st.session_state.prop_df is not None:
        frames.append(st.session_state.prop_df.copy())
    if st.session_state.mpo_df is not None:
        mp = st.session_state.mpo_df.copy()
        if frames:
            new_c = [c for c in mp.columns
                     if c not in frames[0].columns or c == "Structure_ID"]
            frames[0] = frames[0].merge(mp[new_c], on="Structure_ID", how="left")
        else:
            frames.append(mp)

    if not frames:
        if st.session_state.mols:
            frames.append(pd.DataFrame({"Structure_ID": st.session_state.ids}))
        else:
            st.warning("⬅ Run Steps 4 and/or 5 first.")
            st.stop()

    merged = frames[0].copy()

    # Hard filters
    nb = len(merged)
    fc_opts = [c for c in ["PAINS_Flag","Brenk_Alert"] if c in merged.columns]
    if fc_opts:
        st.markdown("### Hard Filters")
        fcols = st.columns(len(fc_opts))
        for fc_, fax in zip(fc_opts, fcols):
            apply_f = fax.checkbox(f"Remove {fc_}", value=True, key=f"hf_{fc_}")
            if apply_f:
                merged = merged[merged[fc_] != ("PAINS" if "PAINS" in fc_ else "ALERT")]
        na_ = len(merged)
        if na_ < nb:
            st.markdown(
                f'<div class="warn">⚠ {nb-na_} compounds removed by hard filters. '
                f'{na_} remaining.</div>', unsafe_allow_html=True)

    # ── Ranked table based on MPO_Score ──────────────────────────
    sort_col = "MPO_Score" if "MPO_Score" in merged.columns else (
               "Custom_MPO" if "Custom_MPO" in merged.columns else None)

    if sort_col:
        merged = merged.sort_values(sort_col, ascending=False).reset_index(drop=True)
    merged.insert(0, "Rank", range(1, len(merged)+1))
    st.session_state.final_df = merged

    # ── Key metrics ───────────────────────────────────────────────
    st.markdown("### Key Metrics Overview")
    num_merged = merged.select_dtypes(include=[np.number]).columns.tolist()
    num_merged = [c for c in num_merged if c not in ["Rank"] and merged[c].notna().sum() > 1]

    if num_merged:
        stats_df = merged[num_merged].describe().T[["mean","min","max","50%"]]
        stats_df.columns = ["Mean","Min","Max","Median"]
        st.dataframe(stats_df.round(3), use_container_width=True)

    # ── Synthesis recommendation — top N cards ───────────────────
    st.markdown("### 🧪 Top Candidates for Synthesis")
    has_ref = (st.session_state.ref_df is not None and
               len(st.session_state.ref_df) > 0)
    ref_id_label = None
    if has_ref:
        rdf_ref = st.session_state.ref_df
        src_id_ref = ("Structure_ID" if "Structure_ID" in rdf_ref.columns
                      else st.session_state.id_col)
        ref_id_label = str(rdf_ref[src_id_ref].values[0]) if src_id_ref in rdf_ref.columns else None

    n_cards = st.slider("Show top N candidates", 3, min(20, len(merged)), 5, key="n_cards")
    mols_map = {str(mid): mol for mid, mol in
                zip(st.session_state.ids, st.session_state.mols)}

    for i, (_, row) in enumerate(merged.head(n_cards).iterrows()):
        mid = str(row["Structure_ID"])
        mpo_val = row.get(sort_col, np.nan) if sort_col else np.nan
        mol = mols_map.get(mid)
        svg = mol_svg(mol, 260, 200) if mol else None

        # Similarity to reference
        sim_str = ""
        if has_ref and st.session_state.ref_mols and mol:
            sim = tanimoto(mol, st.session_state.ref_mols[0])
            if sim is not None:
                sim_str = f"Tanimoto vs ref: **{sim:.3f}**"

        rank_badge = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"

        # Card container
        st.markdown(
            f'<div style="background:#111827;border:1px solid #1e2d3d;'
            f'border-radius:10px;padding:4px 0 4px 0;margin-bottom:4px">'
            f'</div>', unsafe_allow_html=True)

        c_img, c_info = st.columns([1, 2])
        with c_img:
            if svg:
                st.markdown(
                    f'<div style="background:white;border-radius:8px;'
                    f'padding:6px;text-align:center;margin:4px">'
                    f'{svg}'
                    f'<br><small style="color:#555;font-size:0.78rem">{mid}</small>'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:#1e2d3d;border-radius:8px;'
                    f'padding:40px 10px;text-align:center;margin:4px;color:#78909c">'
                    f'No structure<br><small>{mid}</small></div>',
                    unsafe_allow_html=True)

        with c_info:
            st.markdown(f"#### {rank_badge}  `{mid}`")
            if not pd.isna(mpo_val):
                st.markdown(f"**MPO Score: {mpo_val:.3f}** / 1.0")
                st.progress(float(mpo_val))
            if sim_str: st.markdown(sim_str)

            # Show key property values inline
            key_props = [c for c in (["docking score","Docking_score",
                                       "CH42_Caco2_Papp_reg","CH42_MDCK-MDR1_Papp_reg",
                                       "CH42_PAMPA_Pe_reg","CH42_Caco2_ER","MW","LogP",
                                       "QED","SA_Score","ESOL_LogS"])
                         if c in row.index and pd.notna(row[c])]
            if key_props:
                prop_str = "  ·  ".join(f"{c}: `{row[c]:.2f}`" for c in key_props[:6])
                st.caption(prop_str)

            # Alert flags inline
            alert_hits = []
            for ac in ["PAINS_Flag","Brenk_Alert","Halogen_Alert"]:
                if ac in row.index and str(row[ac]) in ("PAINS","ALERT"):
                    desc_col = ac.replace("_Flag","_Description").replace("_Alert","_Detail")
                    desc = f" — {row[desc_col]}" if desc_col in row.index and pd.notna(row[desc_col]) else ""
                    alert_hits.append(f"⚠ {ac}{desc}")
            if alert_hits:
                st.markdown(f'<div style="color:#ffb74d;font-size:.82rem">{"  |  ".join(alert_hits)}</div>',
                            unsafe_allow_html=True)

            # Go / no-go advice
            if not pd.isna(mpo_val):
                if mpo_val >= 0.70:
                    st.markdown('<div class="good">✅ Strong candidate — recommend for synthesis</div>',
                                unsafe_allow_html=True)
                elif mpo_val >= 0.45:
                    st.markdown('<div class="warn">⚠ Moderate candidate — consider optimisation first</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:#1a0a0a;border-left:4px solid #ef5350;'
                                'border-radius:6px;padding:10px 14px;font-size:.85rem;color:#ef9a9a">'
                                '❌ Weak candidate — likely deprioritise</div>',
                                unsafe_allow_html=True)
        st.markdown('<hr class="d">', unsafe_allow_html=True)

    # ── Reference comparison table ────────────────────────────────
    if has_ref and ref_id_label and sort_col and sort_col in merged.columns:
        st.markdown("### 📊 Top Candidates vs Reference")
        ref_row_merged = merged[merged["Structure_ID"].astype(str)==ref_id_label]
        compare_rows = pd.concat([ref_row_merged,
                                   merged[merged["Structure_ID"].astype(str)!=ref_id_label]
                                   .head(n_cards)]).reset_index(drop=True)
        num_c_show = [c for c in compare_rows.select_dtypes(include=[np.number]).columns
                      if compare_rows[c].notna().sum() > 0][:12]
        if "Structure_ID" not in num_c_show:
            num_c_show = ["Structure_ID"] + num_c_show
        else:
            num_c_show = ["Structure_ID"] + [c for c in num_c_show if c != "Structure_ID"]
        st.dataframe(compare_rows[num_c_show], use_container_width=True)

    # ── Full ranked table ─────────────────────────────────────────
    st.markdown("### Full Ranked Table")
    n_show = st.slider("Show top N", 5, len(merged), min(30, len(merged)), key="n_show_full")
    st.dataframe(merged.head(n_show), use_container_width=True)

    # ── Per-cluster top compound (notebook-style final report) ────
    if st.session_state.clusters and st.session_state.mpo_df is not None:
        st.markdown("---")
        st.markdown("### 🧬 Top Compound Per Cluster — Synthesis Shortlist")
        st.markdown("""<div class="info">
Exactly like your mentor's notebook: one best compound selected per scaffold cluster
based on MPO score. These are your primary synthesis candidates.
</div>""", unsafe_allow_html=True)

        mpo_df_for_cl = st.session_state.mpo_df
        sort_col_cl = "MPO_Score" if "MPO_Score" in mpo_df_for_cl.columns else None
        cluster_report = []

        for ci, cl in enumerate(st.session_state.clusters):
            cl_ids = [str(st.session_state.ids[j])
                      for j in cl if j < len(st.session_state.ids)]
            sub = mpo_df_for_cl[mpo_df_for_cl["Structure_ID"].isin(cl_ids)]
            if len(sub) == 0:
                continue
            if sort_col_cl:
                best_row = sub.sort_values(sort_col_cl, ascending=False).iloc[0]
            else:
                best_row = sub.iloc[0]

            best_id = str(best_row["Structure_ID"])
            mpo_v   = best_row.get(sort_col_cl, np.nan) if sort_col_cl else np.nan

            # Get raw property values from merged
            orig_row = merged[merged["Structure_ID"].astype(str) == best_id]
            sim_v = None
            if has_ref and st.session_state.ref_mols:
                mol_best = mols_map.get(best_id)
                if mol_best:
                    sim_v = tanimoto(mol_best, st.session_state.ref_mols[0])

            cluster_report.append({
                "Cluster": ci+1,
                "Cluster_Size": len(cl),
                "Top_Compound": best_id,
                "MPO_Score": round(float(mpo_v), 3) if not pd.isna(mpo_v) else None,
                "Tanimoto_vs_Ref": round(sim_v, 3) if sim_v is not None else None,
                "Recommendation": ("✅ Synthesise" if not pd.isna(mpo_v) and mpo_v >= 0.70
                                   else "⚠ Consider" if not pd.isna(mpo_v) and mpo_v >= 0.45
                                   else "❌ Deprioritise")
            })

        if cluster_report:
            cl_report_df = pd.DataFrame(cluster_report)
            st.dataframe(cl_report_df, use_container_width=True)

            # Visual: one card per cluster top compound
            cols_per_row = min(3, len(cluster_report))
            card_cols = st.columns(cols_per_row)
            for ci2, rec in enumerate(cluster_report):
                with card_cols[ci2 % cols_per_row]:
                    mol_c = mols_map.get(rec["Top_Compound"])
                    svg_c = mol_svg(mol_c, 220, 170) if mol_c else ""
                    if svg_c: st.markdown(svg_c, unsafe_allow_html=True)
                    st.markdown(f"**Cluster {rec['Cluster']}** · `{rec['Top_Compound']}`")
                    if rec["MPO_Score"] is not None:
                        st.markdown(f"MPO: **{rec['MPO_Score']:.3f}**")
                        st.progress(float(rec["MPO_Score"]))
                    st.markdown(rec["Recommendation"])

            st.download_button("⬇ Download Cluster Shortlist CSV",
                               cl_report_df.to_csv(index=False).encode(),
                               "cluster_top_compounds.csv", "text/csv")

    st.download_button("⬇ Download Full Ranked CSV",
                       merged.to_csv(index=False).encode(),
                       "ranked_candidates.csv", "text/csv")

# TAB 7 — AI REPORT
# ══════════════════════════════════════════════════════════════════
with TABS[6]:
    st.markdown('<span class="badge">Step 7</span>', unsafe_allow_html=True)
    st.markdown("## 🤖 AI-Powered Report & Synthesis Recommendations")
    st.markdown("""<div class="info">
All computed data is bundled as context for the AI — results are project-specific,
not generic advice. Configure your AI provider in the sidebar.
</div>""", unsafe_allow_html=True)

    plbl={"claude":"Claude (Anthropic)","openai":"GPT-4o (OpenAI)","gemini":"Gemini 2.0 Flash (Google)"}
    prov=st.session_state.api_provider
    st.info(f"Provider: **{plbl.get(prov,prov)}** · Key: {'✓ set' if st.session_state.api_key else '✗ missing (see sidebar)'}")

    rtype=st.selectbox("Report type",[
        "Full Candidate Selection Report",
        "MPO Analysis Deep-Dive",
        "Synthetic Accessibility & Cost (SA Score + SCScore)",
        "PAINS & Structural Alert Analysis",
        "Reference Compound Comparison (LE, LipE, Tanimoto)",
        "Patent Novelty & SciFinder R-Group FTO Strategy",
        "Top 5 Synthesis Recommendations",
        "Go/No-Go Decision Matrix",
    ],key="rtype")

    custom=st.text_area("Project context (optional — more detail = better report)",
                        placeholder="Target: EGFR · Indication: NSCLC · Reference: erlotinib IC50=2nM · Goal: oral CNS penetration",
                        height=80,key="custom")

    def build_ctx():
        p = []
        if st.session_state.mpo_df is not None:
            m = st.session_state.mpo_df
            score_col = "MPO_Score" if "MPO_Score" in m.columns else None
            if score_col:
                t3 = m.nlargest(3, score_col)["Structure_ID"].astype(str).tolist()
                p.append(f"MPO: n={len(m)}, mean={m[score_col].mean():.3f}, "
                         f"max={m[score_col].max():.3f}, "
                         f"top3={t3}.")
        if st.session_state.prop_df is not None:
            pp = st.session_state.prop_df
            for col, lbl, thr, op in [("SA_Score","SA",4,"≤"),("SCScore","SCScore",3,"≤"),
                                       ("QED","QED",0.6,">"),("Tanimoto_RefSim","Tanimoto_Ref",0.4,">")]:
                if col in pp:
                    n_pass = int((pp[col]<=thr).sum()) if op=="≤" else int((pp[col]>thr).sum())
                    p.append(f"{lbl}: mean={pp[col].mean():.3f}, {op}{thr}: {n_pass}.")
            if "PAINS_Flag"  in pp: p.append(f"PAINS: {(pp['PAINS_Flag']=='PAINS').sum()} flagged.")
            if "Brenk_Alert" in pp: p.append(f"Brenk: {(pp['Brenk_Alert']=='ALERT').sum()} alerts.")
            if "LE"          in pp: p.append(f"LE: mean={pp['LE'].mean():.3f}, >0.3={(pp['LE']>0.3).sum()}.")
            if "LipE"        in pp: p.append(f"LipE: mean={pp['LipE'].mean():.2f}.")
        if st.session_state.markush_summary is not None:
            mk = st.session_state.markush_summary
            rg = st.session_state.rgroup_long
            p.append(f"Markush: {len(mk)} clusters, top cluster size={mk['Cluster_Size'].max()}, "
                     f"R-group positions={rg['RGroup'].nunique() if 'RGroup' in rg else 'N/A'}.")
        if st.session_state.final_df is not None:
            fd = st.session_state.final_df
            t5 = fd.head(5)["Structure_ID"].astype(str).tolist() if "Structure_ID" in fd else []
            p.append(f"Top 5 ranked compounds: {t5}.")
        if custom:
            p.append(f"Project context: {custom}")
        return "\n".join(p) if p else "No computed data available yet — run Steps 4 and 5 first."

    PROMPTS={
        "Full Candidate Selection Report": lambda c: f"""
You are a senior medicinal chemist and drug discovery project leader.
Write a comprehensive candidate selection report based on this computed data:

{c}

Use these sections with clear headings:
1. Executive Summary (3 sentences — clear recommendation)
2. MPO Score Analysis (interpret distribution; highlight drug-like compounds)
3. Synthetic Accessibility (SA Score + SCScore interpretation; flag difficult compounds)
4. Structural Alert Analysis (PAINS/Brenk results and implications)
5. Reference Compound Comparison (similarity, LE, LipE improvements)
6. Patent Novelty (Markush clusters, SciFinder R-group novelty)
7. Top 5 Candidates (justify each with specific property numbers)
8. Risk Assessment (one key risk per top candidate)
9. Next Steps (specific, prioritised actions for the synthesis team)

Cite actual numbers. Use rigorous medicinal chemistry reasoning throughout.
""",
        "MPO Analysis Deep-Dive": lambda c: f"""
You are a computational medicinal chemist expert in Pfizer's CNS-MPO framework.

DATA: {c}

1. CNS-MPO framework (Wager et al. ACS Chem Neurosci 2010) — 3-sentence explanation
2. Distribution analysis — what does the histogram shape tell us about library quality?
3. Per-parameter breakdown — which of MW/LogP/HBD/HBA/TPSA/pKa is the bottleneck?
4. Radar profile — describe the ideal spoke pattern; which compounds come closest?
5. Structural recommendations — what changes would move weak compounds above MPO=4?
6. Benchmark comparison to known CNS drugs (e.g. donepezil, haloperidol)
""",
        "Synthetic Accessibility & Cost (SA Score + SCScore)": lambda c: f"""
You are an expert synthetic chemist with pharmaceutical industry experience.

DATA: {c}

1. SA Score methodology (Ertl & Schuffenhauer 2009) — interpretation guide
2. SCScore methodology (Coley et al. MIT 2018) — how it complements SA Score
3. Comparison of both scores — where do they agree/disagree and why?
4. Compounds to deprioritise (SA>4.5 or SCScore>3.5 with structural reasoning)
5. Estimated step count implications (SA 1-3 = 1-3 steps, SA 3-5 = 3-6 steps, etc.)
6. Retrosynthetic opportunities for this compound class
7. Starting material availability strategy (eMolecules, ZINC, Mcule)
8. Qualitative cost estimates: low (<€50/g), medium (€50-500/g), high (>€500/g)
9. Top 5 synthesis-friendly candidates with justification
""",
        "PAINS & Structural Alert Analysis": lambda c: f"""
You are a medicinal chemistry ADMET expert.

DATA: {c}

1. PAINS explained (Baell & Holloway 2010) — why they cause false positives in HTS
2. Brenk/REOS alerts — what toxicophores are being caught and their clinical implications
3. How many compounds pass both filters — percentage and quality interpretation
4. For flagged compounds: are any worth investigating with minor modifications?
5. Common false-positive patterns in AI-generated molecules
6. Recommendation: which flagged compounds (if any) are salvageable?
""",
        "Reference Compound Comparison (LE, LipE, Tanimoto)": lambda c: f"""
You are a medicinal chemist leading a lead optimisation campaign.

DATA: {c}

1. Tanimoto similarity distribution — analogues (>0.6) vs scaffold hops (0.3-0.6)
2. LE analysis — which generated compounds are more efficient per atom than reference?
3. LipE analysis — which achieve activity without over-relying on lipophilicity?
4. Head-to-head table: reference vs top 3 generated compounds (all key metrics)
5. Strategy: pursue analogues or scaffold hops? Data-driven justification.
6. Best candidates: novel space (low Tanimoto) with improved LE and LipE?
""",
        "Patent Novelty & SciFinder R-Group FTO Strategy": lambda c: f"""
You are a pharmaceutical IP specialist and medicinal chemist.

DATA: {c}

1. SciFinder R-group labels explained (Ra, Rb, Rc mapping to Markush positions)
2. Cluster distribution — what does it tell us about chemical space coverage?
3. FTO strategy — how to use these clusters to identify patent white space
4. Novel R-group combinations — which combinations are most likely outside existing claims?
5. Step-by-step SciFinder search strategy for the medicinal chemist
6. Patent drafting recommendations — what structural features are worth protecting?
""",
        "Top 5 Synthesis Recommendations": lambda c: f"""
You are a medicinal chemistry project leader at a synthesis committee meeting.

DATA: {c}

For each of the top 5 ranked compounds provide:
- Structure_ID and composite score
- Property summary: MPO, SA, SCScore, QED, PAINS status, similarity, LE
- Why it ranks highly — multi-criteria justification with specific numbers
- Proposed retrosynthetic strategy (2-3 sentence overview)
- Key synthesis challenges and mitigation strategies
- Estimated timeline and difficulty (★ to ★★★★★)
- First assay recommendation and rationale

End with a prioritised synthesis matrix (compound vs criterion, Green/Amber/Red).
""",
        "Go/No-Go Decision Matrix": lambda c: f"""
You are the decision-maker in a drug discovery project.

DATA: {c}

Produce a structured Go/No-Go assessment:
1. Decision criteria table with thresholds (MPO≥4, SA≤4, SCScore≤3, PAINS=CLEAN, etc.)
2. Traffic-light matrix for top 10 compounds (Green/Amber/Red per criterion)
3. Overall Go/No-Go per compound with confidence (High/Medium/Low)
4. What additional data would change a No-Go to Go?
5. Immediate action items — ordered by priority
6. Portfolio recommendation — how many to take into synthesis and which ones

Be definitive. A synthesis committee needs clear decisions, not ambiguity.
""",
    }

    if st.button(f"🤖 Generate Report", type="primary", key="gen"):
        ctx=build_ctx()
        fn=PROMPTS.get(rtype)
        prompt=fn(ctx) if fn else f"Analyse this drug discovery data:\n{ctx}"
        with st.spinner(f"Generating '{rtype}'…"):
            rep=call_ai(prompt,prov,st.session_state.api_key,max_tokens=2400)
        st.markdown(f"### 📋 {rtype}")
        st.markdown(f'<div class="ai-box">{rep}</div>',unsafe_allow_html=True)
        st.download_button("⬇ Download Report (TXT)",rep.encode(),
                           f"report_{rtype[:25].replace(' ','_').lower()}.txt","text/plain")

    st.markdown('<hr class="d">', unsafe_allow_html=True)
    st.markdown("### 📖 Property Thresholds Reference")
    st.markdown("""
| Property | Source | ✅ Good | ⚠️ Caution | ❌ Poor |
|----------|--------|---------|-----------|--------|
| MPO Total | Wager 2010 | ≥ 4.0 | 3–4 | < 3 |
| SA Score | Ertl 2009 | ≤ 4 | 4–6 | > 6 |
| SCScore | Coley 2018 | ≤ 3 | 3–4 | > 4 |
| QED | Bickerton 2012 | > 0.6 | 0.4–0.6 | < 0.4 |
| LE | Hopkins 2004 | > 0.3 | 0.2–0.3 | < 0.2 |
| LipE | Leeson 2007 | > 5 | 3–5 | < 3 |
| Tanimoto | Morgan r=2 | Analogue >0.6 / Hop 0.3–0.6 | — | < 0.3 |
| PAINS | Baell 2010 | CLEAN | — | PAINS |
| Brenk | Brenk 2008 | CLEAN | — | ALERT |
| Lipinski | Lipinski 1997 | PASS (0 violations) | 1 violation | ≥ 2 |
| Stereocenters | — | 0–1 | 2–3 | > 3 |
""")

# ══════════════════════════════════════════════════════════════════
# TAB 8 — VISUAL REPORT (publication-quality, white background)
# ══════════════════════════════════════════════════════════════════
with TABS[7]:
    st.markdown('<span class="badge">Step 8</span>', unsafe_allow_html=True)
    st.markdown("## 📈 Visual Report — Full Pipeline Summary")
    st.markdown("""<div class="info">
Every plot uses a <strong>white background</strong> for publication/presentation use.
Each panel has an individual download button (PNG, 300 dpi).
Plots are generated from whatever data has been computed in Steps 3–6.
</div>""", unsafe_allow_html=True)

    def pub_fig(figsize=(8,5)):
        """White-background figure for publication."""
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color("black")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        return fig, ax

    def dl_btn(fig, filename, label="⬇ Download PNG"):
        """Render a download button for a matplotlib figure."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        st.download_button(label, buf, file_name=filename, mime="image/png",
                           key=f"dl_{filename}")

    has_any = any([
        st.session_state.mols,
        st.session_state.prop_df is not None,
        st.session_state.mpo_df  is not None,
        st.session_state.clusters,
        st.session_state.markush_summary is not None,
    ])
    if not has_any:
        st.warning("⬅ Run Steps 1–6 first to generate data for the visual report.")
    else:
        mols_vr  = st.session_state.mols
        ids_vr   = st.session_state.ids
        prop_vr  = st.session_state.prop_df
        mpo_vr   = st.session_state.mpo_df
        mk_vr    = st.session_state.markush_summary
        final_vr = st.session_state.final_df
        mols_map_vr = {str(mid): mol for mid, mol in zip(ids_vr, mols_vr)}

        # ── Panel 1: Library 2D grid ──────────────────────────────
        st.markdown("---")
        st.markdown("### 📌 Panel 1 — Library Overview (2D Structures)")
        n_grid = st.slider("Compounds to show in grid", 4, min(48, len(mols_vr)), 12, key="vr_grid")
        if st.button("Generate 2D Grid", key="vr_grid_btn"):
            ncols = 4
            rows_ = [mols_vr[:n_grid][i:i+ncols] for i in range(0, n_grid, ncols)]
            irows_= [ids_vr[:n_grid][i:i+ncols]  for i in range(0, n_grid, ncols)]
            for mrow_, irow_ in zip(rows_, irows_):
                gc = st.columns(ncols)
                for gc_, mol_, mid_ in zip(gc, mrow_, irow_):
                    svg_ = mol_svg(mol_, 220, 170)
                    if svg_:
                        gc_.markdown(
                            f'<div style="text-align:center">{svg_}'
                            f'<br><small style="color:#555">{mid_}</small></div>',
                            unsafe_allow_html=True)

        # ── Panel 2: Cluster bar chart ────────────────────────────
        if mk_vr is not None:
            st.markdown("---")
            st.markdown("### 📌 Panel 2 — Scaffold Cluster Sizes")
            fig, ax = pub_fig((8, 4))
            colors_ = plt.cm.Set2(np.linspace(0, 1, len(mk_vr)))
            ax.bar([f"Cluster {r['Cluster']}" for _, r in mk_vr.iterrows()],
                   mk_vr["Cluster_Size"], color=colors_, edgecolor="black", linewidth=0.6)
            ax.set_xlabel("Scaffold Cluster"); ax.set_ylabel("Number of compounds")
            ax.set_title("Butina Scaffold Cluster Distribution", fontweight="bold")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            dl_btn(fig, "panel2_clusters.png"); plt.close()

        # ── Panel 3: Property distributions ──────────────────────
        if prop_vr is not None:
            st.markdown("---")
            st.markdown("### 📌 Panel 3 — Property Distributions")
            dist_cols = [c for c in ["MW","LogP","TPSA","HBD","HBA","RotBonds",
                                      "SA_Score","SCScore","QED","Fsp3",
                                      "ESOL_LogS","CNS_BBB_Score"]
                         if c in prop_vr.columns and prop_vr[c].notna().sum() > 1]
            if dist_cols:
                n = len(dist_cols)
                ncols_ = min(4, n)
                nrows_ = math.ceil(n / ncols_)
                fig, axes = plt.subplots(nrows_, ncols_,
                                          figsize=(4*ncols_, 3.5*nrows_),
                                          facecolor="white")
                axes = np.array(axes).flatten()
                thresholds = {
                    "MW":(500,"Lipinski MW≤500"),
                    "LogP":(5,"Lipinski LogP≤5"),
                    "TPSA":(140,"Veber TPSA≤140"),
                    "HBD":(5,"HBD≤5"), "HBA":(10,"HBA≤10"),
                    "SA_Score":(4,"SA≤4 (easy)"),
                    "SCScore":(3,"SCScore≤3"),
                    "QED":(0.6,"QED>0.6"),
                    "Fsp3":(0.42,"Fsp3>0.42"),
                    "ESOL_LogS":(-4,"LogS>-4 (soluble)"),
                    "CNS_BBB_Score":(3,"CNS≥3"),
                }
                for ax_, col_ in zip(axes[:n], dist_cols):
                    ax_.set_facecolor("white")
                    for sp in ax_.spines.values(): sp.set_color("black")
                    v_ = prop_vr[col_].dropna()
                    ax_.hist(v_, bins=min(20, len(v_)), color="#4472C4",
                              edgecolor="white", alpha=0.85)
                    if col_ in thresholds:
                        tv, tlbl = thresholds[col_]
                        ax_.axvline(tv, color="red", lw=1.8, ls="--", label=tlbl)
                        ax_.legend(fontsize=7, frameon=False)
                    ax_.set_title(col_, fontweight="bold", fontsize=10, color="black")
                    ax_.set_xlabel("Value", fontsize=8, color="black")
                    ax_.set_ylabel("Count",  fontsize=8, color="black")
                    ax_.tick_params(colors="black", labelsize=7)
                for ax_ in axes[n:]: ax_.set_visible(False)
                plt.suptitle("Property Distributions", fontsize=13, fontweight="bold",
                              color="black", y=1.01)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                dl_btn(fig, "panel3_distributions.png"); plt.close()

        # ── Panel 4: Structural alert summary ────────────────────
        if prop_vr is not None:
            alert_cols = [c for c in ["PAINS_Flag","Brenk_Alert",
                                       "Halogen_Alert","Lipinski_Pass",
                                       "Veber_Pass","Muegge_Pass","Ro3_Pass"]
                          if c in prop_vr.columns]
            if alert_cols:
                st.markdown("---")
                st.markdown("### 📌 Panel 4 — Structural Alert & Filter Summary")
                fig, axes = plt.subplots(1, len(alert_cols),
                                          figsize=(3*len(alert_cols), 4),
                                          facecolor="white")
                if len(alert_cols)==1: axes=[axes]
                palette = {"PAINS":"#D32F2F","CLEAN":"#388E3C",
                            "ALERT":"#D32F2F","PASS":"#388E3C",
                            "FAIL":"#D32F2F","REVIEW":"#F57C00",
                            "OK":"#388E3C","N/A":"#9E9E9E"}
                for ax_, col_ in zip(axes, alert_cols):
                    ax_.set_facecolor("white")
                    for sp in ax_.spines.values(): sp.set_color("black")
                    vc = prop_vr[col_].value_counts()
                    colors_ = [palette.get(str(k),"#607D8B") for k in vc.index]
                    bars = ax_.bar(vc.index.astype(str), vc.values,
                                   color=colors_, edgecolor="black", linewidth=0.5)
                    for bar_, val_ in zip(bars, vc.values):
                        ax_.text(bar_.get_x()+bar_.get_width()/2,
                                  bar_.get_height()+0.3, str(val_),
                                  ha="center", va="bottom", fontsize=9, fontweight="bold",
                                  color="black")
                    ax_.set_title(col_.replace("_"," "), fontweight="bold",
                                   fontsize=9, color="black")
                    ax_.tick_params(colors="black", labelsize=8)
                    ax_.set_ylabel("Count", fontsize=8, color="black")
                plt.suptitle("Structural Filters & Alerts", fontsize=12,
                              fontweight="bold", color="black")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                dl_btn(fig, "panel4_alerts.png"); plt.close()

        # ── Panel 5: Halogen / fluorine profile ──────────────────
        if prop_vr is not None and "F_Count" in prop_vr.columns:
            st.markdown("---")
            st.markdown("### 📌 Panel 5 — Halogen & Fluorine Profile")
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
            for ax_ in axes:
                ax_.set_facecolor("white")
                for sp in ax_.spines.values(): sp.set_color("black")
                ax_.tick_params(colors="black")

            # F count histogram
            fc = prop_vr["F_Count"].dropna()
            axes[0].hist(fc, bins=range(int(fc.max())+2), color="#1565C0",
                          edgecolor="white", alpha=0.85)
            axes[0].axvline(5, color="red", lw=2, ls="--", label="≥5F alert")
            axes[0].set_title("Fluorine Count per Compound", fontweight="bold", color="black")
            axes[0].set_xlabel("F atoms"); axes[0].set_ylabel("Count")
            axes[0].legend(fontsize=8, frameon=False)

            # Halogen count
            if "Halogen_Count" in prop_vr.columns:
                hc = prop_vr["Halogen_Count"].dropna()
                axes[1].hist(hc, bins=range(int(hc.max())+2), color="#6A1B9A",
                              edgecolor="white", alpha=0.85)
                axes[1].set_title("Total Halogen Count", fontweight="bold", color="black")
                axes[1].set_xlabel("Halogen atoms"); axes[1].set_ylabel("Count")

            # Alert pie
            if "Halogen_Alert" in prop_vr.columns:
                vc2 = prop_vr["Halogen_Alert"].value_counts()
                c2  = ["#D32F2F" if x=="ALERT" else "#388E3C" for x in vc2.index]
                axes[2].pie(vc2.values, labels=vc2.index, colors=c2,
                             autopct="%1.0f%%", startangle=90,
                             textprops={"color":"black","fontsize":10})
                axes[2].set_title("Halogen Alert Distribution", fontweight="bold", color="black")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            dl_btn(fig, "panel5_halogens.png"); plt.close()

        # ── Panel 6: ESOL solubility ──────────────────────────────
        if prop_vr is not None and "ESOL_LogS" in prop_vr.columns:
            st.markdown("---")
            st.markdown("### 📌 Panel 6 — Estimated Aqueous Solubility (ESOL)")
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
            for ax_ in axes:
                ax_.set_facecolor("white")
                for sp in ax_.spines.values(): sp.set_color("black")
                ax_.tick_params(colors="black")

            ls_vals = prop_vr["ESOL_LogS"].dropna()
            axes[0].hist(ls_vals, bins=15, color="#00838F", edgecolor="white", alpha=0.85)
            for xv, lbl, c_ in [(-2,"Highly sol.","#388E3C"),
                                   (-4,"Low sol.","#F57C00"),
                                   (-6,"Insoluble","#D32F2F")]:
                axes[0].axvline(xv, color=c_, lw=1.8, ls="--", label=lbl)
            axes[0].set_title("ESOL LogS Distribution (Delaney 2004)",
                               fontweight="bold", color="black")
            axes[0].set_xlabel("LogS (estimated)"); axes[0].set_ylabel("Count")
            axes[0].legend(fontsize=8, frameon=False)

            if "Solubility_Cat" in prop_vr.columns:
                vc3 = prop_vr["Solubility_Cat"].value_counts()
                color_map = {"Highly soluble":"#1B5E20","Soluble":"#388E3C",
                              "Low solubility":"#F57C00","Insoluble":"#B71C1C"}
                c3 = [color_map.get(x,"#607D8B") for x in vc3.index]
                axes[1].bar(vc3.index, vc3.values, color=c3, edgecolor="black", linewidth=0.5)
                axes[1].set_title("Solubility Category Breakdown",
                                   fontweight="bold", color="black")
                axes[1].set_xlabel("Category"); axes[1].set_ylabel("Count")
                axes[1].tick_params(axis="x", rotation=20, colors="black")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            dl_btn(fig, "panel6_solubility.png"); plt.close()

        # ── Panel 7: MPO ranking bar chart ───────────────────────
        if mpo_vr is not None and "MPO_Score" in mpo_vr.columns:
            st.markdown("---")
            st.markdown("### 📌 Panel 7 — MPO Score Ranking")
            n_bar = st.slider("Top N for bar chart", 10, min(50, len(mpo_vr)), 30, key="vr_mpo_n")
            top_mpo = mpo_vr.sort_values("MPO_Score", ascending=False).head(n_bar)
            fig, ax = pub_fig((max(10, n_bar*0.35), 5))
            cmap_ = plt.cm.RdYlGn
            colors_ = cmap_(top_mpo["MPO_Score"].values)
            bars = ax.bar(top_mpo["Structure_ID"].astype(str),
                           top_mpo["MPO_Score"],
                           color=colors_, edgecolor="black", linewidth=0.4)
            mean_mpo = mpo_vr["MPO_Score"].mean()
            ax.axhline(mean_mpo, color="steelblue", lw=2, ls="--",
                        label=f"Library mean = {mean_mpo:.3f}")
            ax.set_xlabel("Compound ID", fontsize=10)
            ax.set_ylabel("MPO Score (0–1)", fontsize=10)
            ax.set_title(f"MPO Score Ranking — Top {n_bar} Compounds", fontweight="bold")
            ax.set_xticklabels(top_mpo["Structure_ID"].astype(str), rotation=90, fontsize=7)
            ax.legend(fontsize=9, frameon=True)
            ax.set_ylim(0, 1.05)
            sm_ = plt.cm.ScalarMappable(cmap=cmap_, norm=plt.Normalize(0,1))
            sm_.set_array([]); fig.colorbar(sm_, ax=ax, fraction=0.02, pad=0.01).set_label("MPO Score")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            dl_btn(fig, "panel7_mpo_ranking.png"); plt.close()

        # ── Panel 8: Radar — reference vs top 3 ──────────────────
        if (mpo_vr is not None and "MPO_Score" in mpo_vr.columns):
            d_cols_vr = [c for c in mpo_vr.columns if c.startswith("d_")]
            radar_lbl_vr = [c[2:] for c in d_cols_vr]
            if len(d_cols_vr) >= 2:
                st.markdown("---")
                st.markdown("### 📌 Panel 8 — MPO Radar: Reference vs Top Hits")
                sv_r, sn_r = [], []
                # Reference first
                has_ref_vr = (st.session_state.ref_df is not None)
                ref_lbl_vr = None
                if has_ref_vr:
                    rdf_vr = st.session_state.ref_df
                    src_ref_vr = ("Structure_ID" if "Structure_ID" in rdf_vr.columns
                                  else st.session_state.id_col)
                    ref_lbl_vr = str(rdf_vr[src_ref_vr].values[0]) if src_ref_vr in rdf_vr.columns else None
                    if ref_lbl_vr:
                        ref_row_mpo = mpo_vr[mpo_vr["Structure_ID"].astype(str)==ref_lbl_vr]
                        if len(ref_row_mpo):
                            sv_r.append([ref_row_mpo[dc].values[0] for dc in d_cols_vr])
                            ref_mpo_v = ref_row_mpo["MPO_Score"].values[0]
                            sn_r.append(f"{ref_lbl_vr} (MPO:{ref_mpo_v:.2f})")
                # Top 3 non-reference
                for _, row_ in mpo_vr.sort_values("MPO_Score", ascending=False).head(5).iterrows():
                    if ref_lbl_vr and str(row_["Structure_ID"]) == ref_lbl_vr:
                        continue
                    if len(sv_r) >= 4: break
                    sv_r.append([row_.get(dc,0) for dc in d_cols_vr])
                    sn_r.append(f"{str(row_['Structure_ID'])[:16]} (MPO:{row_['MPO_Score']:.2f})")
                if sv_r:
                    fig_r = radar_fig(radar_lbl_vr, sv_r, sn_r,
                                      title="MPO Radar — Reference vs Top Hits",
                                      ref_idx=0)
                    if fig_r:
                        # Make white for download
                        fig_r.patch.set_facecolor("white")
                        c1r, c2r = st.columns([1.2, 0.8])
                        with c1r:
                            st.pyplot(fig_r, use_container_width=True)
                        with c2r:
                            st.markdown("**MPO Scores:**")
                            for nm_, vv_ in zip(sn_r, sv_r):
                                st.markdown(f"• {nm_}")
                        dl_btn(fig_r, "panel8_radar.png"); plt.close()

        # ── Panel 9: Chemical space (PCA) ─────────────────────────
        if len(mols_vr) >= 4:
            st.markdown("---")
            st.markdown("### 📌 Panel 9 — Chemical Space (PCA of Morgan Fingerprints)")
            if st.button("Generate PCA plot", key="vr_pca_btn"):
                try:
                    from sklearn.decomposition import PCA as _PCA
                    fps_arr = []
                    valid_ids_pca = []
                    for mid_, mol_ in zip(ids_vr, mols_vr):
                        try:
                            fp_ = AllChem.GetMorganFingerprintAsBitVect(mol_, 2, 1024)
                            arr_ = np.zeros((1024,))
                            DataStructs.ConvertToNumpyArray(fp_, arr_)
                            fps_arr.append(arr_)
                            valid_ids_pca.append(str(mid_))
                        except: pass
                    if len(fps_arr) >= 4:
                        coords_ = _PCA(n_components=2).fit_transform(np.array(fps_arr))
                        fig, ax = pub_fig((8, 6))
                        # colour by MPO if available
                        if mpo_vr is not None and "MPO_Score" in mpo_vr.columns:
                            mpo_map = dict(zip(mpo_vr["Structure_ID"].astype(str),
                                               mpo_vr["MPO_Score"]))
                            c_arr = [mpo_map.get(i, 0.5) for i in valid_ids_pca]
                            sc_ = ax.scatter(coords_[:,0], coords_[:,1],
                                              c=c_arr, cmap="RdYlGn", s=60,
                                              edgecolors="black", linewidths=0.4,
                                              vmin=0, vmax=1)
                            fig.colorbar(sc_, ax=ax, fraction=0.025).set_label("MPO Score")
                        else:
                            # colour by cluster
                            if st.session_state.clusters:
                                cl_map = {}
                                for ci_, cl_ in enumerate(st.session_state.clusters):
                                    for j_ in cl_:
                                        if j_ < len(ids_vr):
                                            cl_map[str(ids_vr[j_])] = ci_
                                c_arr2 = [cl_map.get(i, -1) for i in valid_ids_pca]
                                ax.scatter(coords_[:,0], coords_[:,1],
                                           c=c_arr2, cmap="Set1", s=60,
                                           edgecolors="black", linewidths=0.4)
                            else:
                                ax.scatter(coords_[:,0], coords_[:,1],
                                           color="#4472C4", s=60,
                                           edgecolors="black", linewidths=0.4)
                        # Annotate top 5 by MPO
                        if mpo_vr is not None and "MPO_Score" in mpo_vr.columns:
                            top5_ids = mpo_vr.nlargest(5,"MPO_Score")["Structure_ID"].astype(str).tolist()
                            for i_, (xi_, yi_) in enumerate(zip(coords_[:,0], coords_[:,1])):
                                if valid_ids_pca[i_] in top5_ids:
                                    ax.annotate(valid_ids_pca[i_], (xi_, yi_),
                                                textcoords="offset points", xytext=(5,4),
                                                fontsize=8, color="black",
                                                fontweight="bold")
                        ax.set_xlabel("PC1", fontsize=11); ax.set_ylabel("PC2", fontsize=11)
                        ax.set_title("Chemical Space — PCA of Morgan Fingerprints\n"
                                      "(colour = MPO score, ★ = top 5 hits)",
                                      fontweight="bold")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        dl_btn(fig, "panel9_pca.png"); plt.close()
                except ImportError:
                    st.warning("scikit-learn required for PCA: `pip install scikit-learn`")

        # ── Panel 10: Docking vs Caco-2 scatter ──────────────────
        if prop_vr is not None or st.session_state.df is not None:
            src_sc = prop_vr if prop_vr is not None else st.session_state.df
            dock_candidates = [c for c in src_sc.columns
                                if any(k in c.lower() for k in ["dock","docking"])]
            admet_candidates= [c for c in src_sc.columns
                                if any(k in c.lower() for k in ["caco","papp","er","pampa","mdck"])]
            if dock_candidates and admet_candidates:
                st.markdown("---")
                st.markdown("### 📌 Panel 10 — Docking Score vs Permeability Scatter")
                sc1, sc2 = st.columns(2)
                x_ax = sc1.selectbox("X axis (activity)", dock_candidates, key="vr_xax")
                y_ax = sc2.selectbox("Y axis (permeability)", admet_candidates, key="vr_yax")
                if x_ax and y_ax and x_ax in src_sc.columns and y_ax in src_sc.columns:
                    id_col_sc = ("Structure_ID" if "Structure_ID" in src_sc.columns
                                  else st.session_state.id_col if st.session_state.id_col in src_sc.columns
                                  else src_sc.columns[0])
                    plot_df_ = src_sc[[x_ax, y_ax, id_col_sc]].dropna().copy()
                    plot_df_["id"] = plot_df_[id_col_sc].astype(str)
                    fig, ax = pub_fig((8, 6))
                    if mpo_vr is not None and "MPO_Score" in mpo_vr.columns:
                        mpo_map2 = dict(zip(mpo_vr["Structure_ID"].astype(str), mpo_vr["MPO_Score"]))
                        c_mpo = [mpo_map2.get(i, 0.5) for i in plot_df_["id"]]
                        sc__ = ax.scatter(plot_df_[x_ax], plot_df_[y_ax],
                                           c=c_mpo, cmap="RdYlGn", s=70,
                                           edgecolors="black", linewidths=0.5,
                                           vmin=0, vmax=1, zorder=3)
                        fig.colorbar(sc__, ax=ax, fraction=0.025).set_label("MPO Score")
                    else:
                        ax.scatter(plot_df_[x_ax], plot_df_[y_ax],
                                   color="#4472C4", s=70, edgecolors="black",
                                   linewidths=0.5, zorder=3)
                    # Annotate top 5
                    if mpo_vr is not None:
                        top5_ = mpo_vr.nlargest(5,"MPO_Score")["Structure_ID"].astype(str).tolist()
                        for _, row_ in plot_df_.iterrows():
                            if str(row_["id"]) in top5_:
                                ax.annotate(row_["id"], (row_[x_ax], row_[y_ax]),
                                            textcoords="offset points", xytext=(5,4),
                                            fontsize=8, fontweight="bold", color="black")
                    ax.set_xlabel(x_ax, fontsize=11); ax.set_ylabel(y_ax, fontsize=11)
                    ax.set_title(f"{x_ax} vs {y_ax}\n(colour = MPO score, labelled = top 5)",
                                  fontweight="bold")
                    ax.grid(True, color="lightgrey", linewidth=0.5, alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    dl_btn(fig, "panel10_scatter.png"); plt.close()

        # ── Panel 11: Correlation heatmap ─────────────────────────
        if prop_vr is not None:
            num_pr = [c for c in prop_vr.select_dtypes(include=[np.number]).columns
                      if prop_vr[c].notna().sum() > 2][:14]
            if len(num_pr) >= 3:
                st.markdown("---")
                st.markdown("### 📌 Panel 11 — Descriptor Correlation Heatmap")
                corr_ = prop_vr[num_pr].corr()
                sz_ = max(7, len(num_pr))
                fig, ax = plt.subplots(figsize=(sz_, sz_-1), facecolor="white")
                ax.set_facecolor("white")
                im_ = ax.imshow(corr_.values, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(len(num_pr)))
                ax.set_xticklabels(num_pr, rotation=45, ha="right",
                                    fontsize=8, color="black")
                ax.set_yticks(range(len(num_pr)))
                ax.set_yticklabels(num_pr, fontsize=8, color="black")
                for i_ in range(len(num_pr)):
                    for j_ in range(len(num_pr)):
                        v_ = corr_.values[i_,j_]
                        ax.text(j_, i_, f"{v_:.2f}", ha="center", va="center",
                                 fontsize=7,
                                 color="white" if abs(v_)>0.65 else "black")
                ax.set_title("Descriptor Correlation Matrix", fontweight="bold",
                              fontsize=12, color="black")
                fig.colorbar(im_, ax=ax, fraction=0.025, pad=0.02)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                dl_btn(fig, "panel11_correlation.png"); plt.close()

        # ── Panel 12: SA Score vs SCScore synthesis map ───────────
        if prop_vr is not None and "SA_Score" in prop_vr.columns and "SCScore" in prop_vr.columns:
            st.markdown("---")
            st.markdown("### 📌 Panel 12 — Synthesis Complexity Map (SA Score vs SCScore)")
            fig, ax = pub_fig((8, 6))
            if mpo_vr is not None and "MPO_Score" in mpo_vr.columns:
                mpo_map3 = dict(zip(mpo_vr["Structure_ID"].astype(str), mpo_vr["MPO_Score"]))
                id_col_pr = st.session_state.id_col
                c_sa = [mpo_map3.get(str(x), 0.5)
                         for x in prop_vr.get(id_col_pr, prop_vr.get("Structure_ID",""))]
                sc_sa = ax.scatter(prop_vr["SCScore"], prop_vr["SA_Score"],
                                    c=c_sa, cmap="RdYlGn", s=65,
                                    edgecolors="black", linewidths=0.4,
                                    vmin=0, vmax=1, zorder=3)
                fig.colorbar(sc_sa, ax=ax, fraction=0.025).set_label("MPO Score")
            else:
                ax.scatter(prop_vr["SCScore"], prop_vr["SA_Score"],
                           color="#4472C4", s=65, edgecolors="black",
                           linewidths=0.4, zorder=3)
            ax.axvline(3, color="orange", lw=1.8, ls="--", label="SCScore=3")
            ax.axhline(4, color="red",    lw=1.8, ls="--", label="SA Score=4")
            ax.invert_yaxis()
            ax.set_xlabel("SCScore (1→easy, 5→complex)", fontsize=11)
            ax.set_ylabel("SA Score (1→easy, 10→hard)",  fontsize=11)
            ax.set_title("Synthesis Complexity Map\nBottom-left = easiest to synthesise",
                          fontweight="bold")
            ax.legend(fontsize=9, frameon=True)
            ax.grid(True, color="lightgrey", linewidth=0.5, alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            dl_btn(fig, "panel12_synthesis_map.png"); plt.close()

        # ── Download all panels as zip ────────────────────────────
        st.markdown("---")
        st.info("💡 Each panel above has its own ⬇ Download PNG button (300 dpi, white background, publication-ready).")


st.markdown('<hr class="d">', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#37474f;font-size:.77rem'>"
    "ChemSelect Pro v2 · Akshay Hegde's Markush Pipeline · "
    "RDKit · Pfizer CNS-MPO · SA Score · SCScore (MIT) · PAINS/Brenk · AI-powered"
    "</div>", unsafe_allow_html=True
)
