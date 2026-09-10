# Dr. Rudzicz Lab — CfA Methods Tutorial

**Authors:** Elahe Rahimi and Dr. Frank Rudzicz  
**Lab:** Dr. Rudzicz Lab (Dalhousie University / Vector Institute)  
**Contact:** Elahe Rahimi — [erahimi@dal.ca]  
**Date:** September 2026

People with remitted major depression remain at risk of relapse. MADRS is collected at clinic visits, while wearable activity and speech can be collected more frequently and may provide signals of change between clinical assessments. We analyzed both streams in the CAN-BIND / CBN-WELL cohort (Brain-CODE). The primary predictive model uses actigraphy; speech is treated as a separate quality-control, transcription, and exploratory linguistic-marker pipeline.

This folder provides a compact, reproducible tutorial version of the analysis. The annotated notebooks run entirely on synthetic data in `test_data/`; no CAN-BIND participant identifiers or Brain-CODE data are included. The synthetic actigraphy example contains 10 invented participants and 80 visits.

## Motivation

Visit-based MADRS is sparse. Daily sleep/activity summaries and repeated speech recordings can provide information between clinical assessments. The actigraphy model learns a stable embedding region and scores how far a new visit lies from that region. The speech analysis follows a separate path: task-specific quality control, transcription of usable free-speech recordings, and exploratory linguistic-marker analysis.

## Data

| Stream | Study source | What the analysis used |
|---|---|---|
| Actigraphy + MADRS | CAN-BIND / CBN-WELL (Brain-CODE) | Minute-level wear, six daily features, MADRS totals, relapse fields `CNSR` and `ADT` |
| Voice | CBN-WELL (Brain-CODE) | Sustained-vowel and speech-task recordings, SRI quality metrics, and transcripts of usable free-speech recordings |

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

1. Focus on two recording streams used in the speech analysis:
   - **sustained vowel** (`/a/` for at least 5 seconds)
   - **free speech** describing physical condition, mental condition, or a happy event (at least 30 seconds)

   Read speech and automatic speech (counting / alphabet) were also collected and are included in the QC logic.

2. Apply task-specific QC using SRI metrics:
   - sustained vowel: RMS, SNR, clipping, and duration; `SADSPEECHEXISTS` is **not** used as a vowel criterion
   - free speech: speech activity detected, duration ≥30 s, SNR ≥10 dB, clipping ≤10%, and single speaker
   - read / counting tasks: speech activity detected, duration 20–120 s, SNR ≥10 dB, clipping ≤10%, and single speaker

3. In the original analysis, approximately **9000** task recordings were processed and about **58%** were usable after QC. About **3000 recordings** were transcribed with Whisper. This tutorial does not redistribute audio and does not call Whisper; it uses synthetic SRI-shaped QC rows and invented transcripts.

4. Eight exploratory psychological and linguistic markers were examined, including first-person language, emotion, and absolutist language. The clearest preliminary linguistic signal closer to relapse was **negation ratio**: explicit negation forms such as `not`, `n't`, `never`, `cannot`, `couldn't`, `wouldn't`, and `shouldn't`, divided by the number of tokens.

5. **Negative emotion** is shown only as a secondary lexicon example. If a negator immediately precedes an emotion word, that emotion hit can be skipped without changing the negation-ratio calculation.

The speech analysis is exploratory and is not a validated relapse classifier. The synthetic example demonstrates the QC and marker calculations; it does not reproduce the cohort-level time-to-relapse result.

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
| `02_speech_qc_and_markers.ipynb` | Speech-task QC, synthetic transcript filtering, and exploratory linguistic markers with negation ratio as the main speech signal |
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
