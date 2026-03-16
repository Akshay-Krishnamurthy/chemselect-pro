# ⚗️ ChemSelect Pro

**A Streamlit-based medicinal chemistry platform for compound library analysis, MPO scoring, scaffold clustering, and AI-driven synthesis recommendations.**

Built for the drug discovery project. Upload your CSV, configure parameters through the UI, and get publication-quality results in minutes.

---

## 🖥️ Live Demo

> Upload `sample_data/sample_library.csv` to try the tool instantly.  
> Reference compound: pick **Structure_ID = 7** (Cpd-07_BenzImid, highest MPO).

---

## ✨ Features

| Step | Feature | Description |
|------|---------|-------------|
| 1 | **Upload & Preview** | CSV/Excel upload, auto-detects SMILES & ID columns, reference compound picker, 2D structure viewer |
| 2 | **Mode & Parameters** | Basic (use existing columns) or Advanced (21 calculated properties) |
| 3 | **Patent Helper** | Butina scaffold clustering → R-group decomposition → SciFinder Markush labels |
| 4 | **Property Calculator** | SA Score, SCScore, QED, Lipinski, Veber, PAINS, Brenk, halogen alerts, ESOL, pKa, Fsp3, and more |
| 5 | **MPO Scoring** | Simple (notebook-style) or Advanced (custom desirability thresholds) — radar, bar chart, parallel coords |
| 6 | **Final Selection** | Top-N synthesis cards with 2D structure, MPO score, go/no-go badge, alert flags |
| 7 | **AI Report** | Gemini 2.5 Flash / Claude / GPT-4o — 8 report types including FTO strategy |
| 8 | **Visual Report** | 12 publication-quality panels (300 dpi PNG), all white background |

---

## 📦 Installation

### Option A — Conda (recommended)

```bash
# 1. Create environment
conda create -n chemselect python=3.10 -y

# 2. Activate
conda activate chemselect

# 3. RDKit must be installed via conda-forge
conda install -c conda-forge rdkit -y

# 4. Remaining packages
pip install -r requirements.txt
```

### Option B — pip only (if conda unavailable)

```bash
pip install -r requirements.txt
```

> ⚠️ RDKit via pip may have issues on some platforms. conda-forge is strongly recommended.

---

## 🚀 Run

```bash
streamlit run chemselect_app.py
```

Opens at **http://localhost:8501**

---

## 📁 Repository Structure

```
chemselect-pro/
├── chemselect_app.py          # Main Streamlit app (single file)
├── requirements.txt           # Python dependencies
├── sample_data/
│   └── sample_library.csv     # 20 demo compounds (kinase-like scaffolds)
├── docs/
│   └── MPO_Analysis.docx      # Tool walkthrough with screenshots
└── README.md
```

---

## 📊 Sample Data

`sample_data/sample_library.csv` contains **20 fictional kinase-like compounds** across 4 scaffolds (Pyrimidine, Benzimidazole, Thienopyrimidine, Quinazoline) with realistic ADMET columns matching the HCIE dataset format:

| Column | Description |
|--------|-------------|
| `Structure_ID` | Unique compound ID |
| `Title` | Scaffold label |
| `SMILES` | 2D structure |
| `docking score` | Glide/GOLD docking score (more negative = better) |
| `CH42_Caco2_ER` | Caco-2 efflux ratio (0–1, higher = better permeability) |
| `CH42_Caco2_Papp_bin` | Caco-2 Papp binary class (0/1) |
| `CH42_Caco2_Papp_reg` | Caco-2 Papp regression value (10⁻⁶ cm/s) |
| `CH42_MDCK-MDR1_Papp_reg` | MDCK-MDR1 permeability (10⁻⁶ cm/s) |
| `CH42_PAMPA_Pe_reg` | PAMPA permeability (10⁻⁶ cm/s) |

---

## ⚙️ Calculated Properties (Advanced Mode)

### Synthetic Complexity
- **SA Score** — Ertl & Schuffenhauer 2009 (1=easy, 10=hard; target ≤4)
- **SCScore** — Coley/MIT 2018 (1=cheap, 5=complex; target ≤3)

### Drug-Likeness Rules
- **QED** — Bickerton 2012 (0→1; target >0.6)
- **Lipinski Ro5** — classic oral drug-likeness filter
- **Veber Rules** — RotBonds ≤10, TPSA ≤140 (oral bioavailability)
- **Egan Egg** — LogP ≤5.88, TPSA ≤131.6 (passive absorption)
- **Muegge Filter** — broader than Lipinski, covers natural-product space
- **Rule of 3** — fragment-like screening filter

### Structural Alerts
- **PAINS A/B/C** — 480 pan-assay interference substructures
- **Brenk Alerts** — 105 toxicophore SMARTS patterns
- **Halogen Alerts** — ≥5 F (excessive fluorination), iodine, poly-Cl/Br
- **Reactive Groups** — nitro, aldehyde, Michael acceptor, epoxide, acyl halide, etc.

### ADMET Estimates
- **ESOL LogS** — Delaney 2004 aqueous solubility estimate
- **pKa Proxy** — basic nitrogen count + ionisation range estimate
- **CNS/BBB Score** — 5-parameter blood-brain barrier score (0–5)
- **Fsp3** — fraction sp3 carbons (Lovering 2009; >0.42 preferred)

### Efficiency Metrics
- **Ligand Efficiency (LE)** — 1.37 × pIC50 / N_heavy
- **Lipophilic Efficiency (LipE)** — pIC50 − cLogP
- **Tanimoto Similarity** — Morgan FP vs reference compound

---

## 🤖 AI Report (Step 7)

Requires an API key (entered in the sidebar — never stored):

| Provider | Model | Notes |
|----------|-------|-------|
| **Gemini** | gemini-2.5-flash | Free tier via Google AI Studio |
| **Claude** | claude-sonnet-4-5 | Anthropic console |
| **OpenAI** | gpt-4o-mini | OpenAI platform |

**Report types available:**
1. Full Candidate Selection Report
2. MPO Analysis Deep-Dive
3. Synthetic Accessibility & Cost
4. PAINS & Structural Alert Analysis
5. Reference Compound Comparison
6. Patent Novelty & SciFinder R-Group FTO Strategy
7. Top 5 Synthesis Recommendations
8. Go/No-Go Decision Matrix

---

## 📈 Visual Report Panels (Step 8)

All panels: **white background, 300 dpi, individual PNG download**.

1. Library Overview (2D structure grid)
2. Scaffold Cluster Sizes
3. Property Distributions (with literature threshold lines)
4. Structural Alert Summary
5. Halogen & Fluorine Profile
6. ESOL Solubility Distribution
7. MPO Score Ranking (RdYlGn bar chart)
8. MPO Radar — Reference vs Top Hits
9. Chemical Space (PCA of Morgan fingerprints)
10. Docking Score vs Permeability Scatter
11. Descriptor Correlation Heatmap
12. Synthesis Complexity Map (SA Score vs SCScore)

---

## 📝 Citation / Acknowledgements

- RDKit: [rdkit.org](https://www.rdkit.org)
- SA Score: Ertl & Schuffenhauer, *J. Cheminform.* 2009
- SCScore: Coley et al., *J. Chem. Inf. Model.* 2018
- PAINS: Baell & Holloway, *J. Med. Chem.* 2010
- Brenk: Brenk et al., *ChemMedChem* 2008
- ESOL: Delaney, *J. Chem. Inf. Comput. Sci.* 2004
- Pfizer CNS-MPO: Wager et al., *ACS Chem. Neurosci.* 2010

---

## 📄 License

For academic and research use. SMILES and property values in sample data are fictional and for demonstration purposes only.
