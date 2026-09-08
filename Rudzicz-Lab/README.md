# Rudzicz Lab — CfA Methods Tutorial

**Project:** Continuous monitoring for depression relapse using longitudinal actigraphy and speech.

**Lab:** Rudzicz Lab (Dalhousie University / Vector Institute)

**Date:** September 2026

People with remitted major depression remain at risk of relapse. MADRS is collected at clinic visits, while wearable activity and speech can be collected more frequently and may provide signals of change between clinical assessments. We analyzed both streams in the CAN-BIND / CBN-WELL cohort (Brain-CODE). The primary predictive model uses actigraphy; speech is treated as a separate quality-control and exploratory marker pipeline.

This folder provides a compact, reproducible tutorial version of the analysis. The annotated notebooks run entirely on synthetic data in `test_data/`; no CAN-BIND participant identifiers or Brain-CODE data are included. The synthetic actigraphy example contains 10 invented participants and 80 visits.

## Motivation

Visit-based MADRS is sparse. Daily sleep/activity summaries and session-level speech can be aligned to clinical visits to support monitoring between assessments. The actigraphy model learns a stable embedding region and scores how far a new visit lies from that region. The speech pipeline is QC-first and applies task-specific quality criteria before marker analysis.

## Data

| Stream | Study source | What the analysis used |
|---|---|---|
| Actigraphy + MADRS | CAN-BIND / CBN-WELL (Brain-CODE) | Minute-level wear, six daily features, MADRS totals, relapse fields `CNSR` and `ADT` |
| Voice | CBN-WELL (Brain-CODE) | Task recordings and SRI quality metrics |

Access to the real cohort is through Brain-CODE and the study PIs. Content in this tutorial follows the CC BY 4.0 license used by the CfA Methods Tutorials repository.

`test_data/` contains invented IDs (`SYN_001` … `SYN_010`), dates, scores, SRI-shaped QC rows, and synthetic transcripts. Table structures mirror those used by the analysis; values are not real. The actigraphy tables are sized so the stable-manifold KNN can use **k = 20** without clipping.

## Relapse definition

In the executed analysis, relapse status is taken from the study relapse file, with relapsers identified by `CNSR == 0`. The first relapse date is `ADT` when available; otherwise, the first visit with MADRS ≥ 22 is used as a fallback. This tutorial follows those analysis labels rather than re-implementing the clinical relapse-definition procedure.

## Pipeline overview

### Actigraphy

1. Keep participants with usable wear time and MADRS visits.
2. Cut relapser timelines at first relapse so no post-relapse days are used.
3. Use six daily features:
   - intradaily variability
   - total sleep time
   - sleep efficiency
   - mean activity
   - sleep onset latency
   - number of awakenings
4. For each MADRS visit, summarize the 7 / 14 / 28-day windows strictly before the visit. For the six features, compute mean, SD, and velocity across the three windows (54 values), together with 18 availability columns, for 72 model inputs. Exclude a visit if any window has <40% availability.
5. Train a small MLP (`72 → 256 → 64`) with contrastive learning: stable visits from the same participant are pulled together, while prodromal visits are pushed away from stable anchors. Same-participant stable–prodrome pairs receive 2× weight.
6. Compute risk from the mean distance to the 20 nearest stable embeddings, normalized by the baseline distance within the stable neighbourhood. The tutorial plots continuous risk trajectories. Any thresholds shown are illustrative only.

### Speech

1. Apply task-specific QC using SRI metrics. Sustained-vowel QC does **not** use `SADSPEECHEXISTS`; it uses RMS, SNR, clipping, and duration.
2. In the original analysis, acoustic features were computed only for recordings that passed task-specific QC. This tutorial demonstrates the QC logic using synthetic SRI-shaped tables rather than loading audio.
3. Use Whisper transcripts from valid free-speech tasks for linguistic-marker analysis. **Negation ratio** counts explicit negation forms (`not`, `n't`, `never`, `couldn't`, etc.) over tokens. **Negative emotion** is a separate lexicon-based feature; a negator can suppress the next emotion-word count without changing the negation ratio.

## Repository layout

```text
Rudzicz-Lab/
├── README.md
├── requirements.txt
├── 01_actigraphy_relapse_model.ipynb
├── 02_speech_qc_and_markers.ipynb
└── test_data/
    ├── sample_actigraphy.csv
    ├── sample_visits.csv
    ├── sample_qc.csv
    └── sample_transcripts.csv
```

| File | Role |
|---|---|
| `01_actigraphy_relapse_model.ipynb` | Relapse zones, six features, 7/14/28-day aggregation, contrastive encoder, KNN with k = 20, and risk trajectories |
| `02_speech_qc_and_markers.ipynb` | Task-specific QC, including the fact that SAD is not a vowel criterion, followed by negation-ratio and negative-emotion marker analysis |
| `test_data/` | Synthetic demonstrative tables only |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

From this folder, open and run the notebooks in order:

1. `01_actigraphy_relapse_model.ipynb`
2. `02_speech_qc_and_markers.ipynb`

PyTorch is required for the encoder cell in notebook 01 and runs on **CPU** for this tutorial; no GPU environment is required. The speech notebook does not call Whisper and does not load audio. It uses the synthetic QC table and synthetic transcripts included in this repository.

## Privacy

No Brain-CODE exports, audio recordings, or real participant identifiers are included in this repository. All files in `test_data/` are synthetic and were created solely to demonstrate the analysis workflow.

## License

CC BY 4.0, consistent with the rest of the CfA Methods Tutorials repository.
