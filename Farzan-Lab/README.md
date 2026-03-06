# IM-Biotype Toolbox — Tutorial Package

This package contains two self-contained tutorials demonstrating the IM-Biotype toolbox for multimodal biotype discovery using regularized Canonical Correlation Analysis (rCCA) and clustering, following methodology established in the psychiatric biotyping literature.

## Contents

```
ForOBICfA/
├── README.md                     ← this file
├── environment.yml               ← conda environment specification
├── setup.py                      ← package installer
├── tutorials/
│   ├── rcca_tutorial.ipynb       ← Tutorial 1: EEG rCCA + test-train evaluation
│   └── pond_meg_rcca_tutorial.ipynb  ← Tutorial 2: POND MEG rCCA + clustering
├── matlab_adaptation/
│   ├── rcca_m_files/             ← Tutorial 1 data (CAN-BIND/EMBARC)
│   └── POND_DATA_SCRIPTS/        ← Tutorial 2 data (POND)
├── lib/imbiotype/                ← core IM-Biotype library
└── src/cca-zoo/                  ← vendored CCA algorithm library
```

## Setup

```bash
# 1. Create conda environment
conda env create --file environment.yml
conda activate imbiotype

# 2. Install the imbiotype package
pip install -e .

# 3. Launch Jupyter and open a tutorial
cd tutorials
jupyter notebook
```

## Tutorial 1: EEG rCCA with Test-Train Evaluation

**Notebook:** `tutorials/rcca_tutorial.ipynb`

Applies a regularized CCA biotyping pipeline to CAN-BIND/EMBARC data (386 subjects, 56 EEG delta-power channels, 16 QIDS depression items). Covers:

- Study creation and dataset import
- Covariate residualization (Age, Sex, Site/Study)
- Z-score standardization
- Regularized CCA with nested cross-validated hyperparameter optimization
- Site-aware permutation testing with Bonferroni correction
- K-means clustering (k=2, 3, 4)
- **Test-train evaluation framework**: assesses generalizability of discovered biotypes to unseen data via repeated nested cross-validation with inside-fold preprocessing to prevent data leakage

**Data:**
- `matlab_adaptation/rcca_m_files/CCA_demo_clinical.csv` — clinical demographics + 16 QIDS items
- `matlab_adaptation/rcca_m_files/data_bio_with_headers.csv` — 56 EEG delta-power channels

## Tutorial 2: POND MEG rCCA and Clustering

**Notebook:** `tutorials/pond_meg_rcca_tutorial.ipynb`

Pairs POND MEG sensor power (149 channels) with CBCL syndrome total scores (8 clinical scales) across four diagnostic groups (ASD, ADHD, OCD, typically-developing). Covers:

- Inner-join data merging across datasets with different subject pools
- Missing-data handling
- Z-score preprocessing (no demographics available)
- Configurable frequency band selection (delta, theta, alpha, beta, gamma)
- K-means clustering on CCA canonical variates with silhouette-based model selection (k=2, 3, 4)
- Side-by-side scatter plots comparing cluster assignments against primary diagnosis
- Optional: full four-stage MEG preprocessing pipeline from raw CTF data

**Data:**
- `matlab_adaptation/POND_DATA_SCRIPTS/MEG_sensor_delta.csv` — 221 subjects, 149 MEG channels
- `matlab_adaptation/POND_DATA_SCRIPTS/clinical_CBCL_6-18_syndrome_total_scores.csv` — 572 subjects, 8 syndrome scales

## Quick Test Mode

Both tutorials include a `QUICK_TEST_MODE = True` toggle for fast execution (minutes). Set to `False` for full replication with production settings (1000 CV repeats, 1000 permutations).

## Requirements

- Python 3.11
- Key dependencies: numpy, scipy, pandas, scikit-learn, matplotlib, PyQt5
- Full dependency list in `environment.yml`
