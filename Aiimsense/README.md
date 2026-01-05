# Conformal Prediction and EFGR-Based Post-Processing for MRI and EMI Super-Resolution

## 1. Conformal Prediction

This project implements a **Conformal Prediction (CP) framework** to enhance clinically meaningful translation of MRI and EMI super-resolution. It addresses a key limitation of existing deep learning approaches: **lack of reliable uncertainty quantification**.

Key features:
- Produces **pixel-wise and region-wise prediction intervals** with formal finite-sample coverage guarantees.
- Generates **statistically valid uncertainty maps** that help clinicians distinguish reliable super-resolved anatomical details from uncertain regions (e.g., tissue boundaries, low-SNR areas, or pathological structures).
- Supports **safer clinical decision-making**, improves interpretability in disorder detection and disease assessment, and mitigates over-confident hallucinated features in downstream diagnostic tasks and clinical trials.
- **Validated** on multiple brain MRI datasets (normal anatomy and pathological cases).
- **Model-agnostic**, post hoc, and compatible with state-of-the-art super-resolution networks without retraining.

Next steps for clinical translation:
- Prospective validation with clinical MRI data.
- Evaluate the impact of uncertainty-aware super-resolution on radiologist confidence and diagnostic consistency.
- Integrate into real-world electromagnetic imaging (EMI) workflows.

---

## 2. EFGR-Based Post-Processing Methods

The project also develops **EFGR-based post-processing methods** to improve the reliability and interpretability of advanced medical imaging algorithms.

Key contributions:
- Enhances **reconstruction fidelity** while explicitly addressing **uncertainty and robustness**.
- Reduces the risk of misleading or overconfident image enhancements that could affect diagnosis, disease monitoring, or treatment planning.
- Applicable to **clinical decision support**, clinical trials relying on quantitative imaging biomarkers, and downstream tasks related to clinical super-resolution.
- Demonstrated **technical feasibility and performance gains** on representative datasets.

Next steps for translation:
- Validate on **real-world clinical data**.
- Collaborate with clinical partners to assess **diagnostic impact**.
- Integrate into existing imaging pipelines as a **lightweight, deployable module** with potential for industry adoption and future commercialization.
- Use EFGR-based post-processing methods to **address weaknesses of state-of-the-art super-resolution networks** especially for EMI headscanner.

---

Comments: the source of these data are from  [BrainTumorMRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), [IXI](https://brain-development.org/ixi-dataset/), and [NINS](https://figshare.com/articles/dataset/Brain_MRI_Dataset/14778750?file=28399209) datasets. The licensing for these data are open sources.

## Summary

This repository presents a comprehensive framework combining **Conformal Prediction** and **EFGR-based post-processing** to enhance MRI super-resolution, providing **quantifiable uncertainty, robust reconstruction**, and pathways for clinical deployment and commercialization.


## Contents
This project has the following contents:

- “README.md” summary of project

- “Conformal Prediction.pdf” report
- “Data-CP” for Conformal Prediction data 
- “Source code-Matlab” folder for Matlab source code
- “Source code-Python” folder Python source code.
- “README-CP.md” for running codes
- “Error Feedback Guided Refinement.pdf” report
- “Data-EFGR” for Error Feedback Guided Refinement data including low- and high-resolution images
- “EFGR-Analysis” file for Matlab source code
- “README-EFGR.md” for running codes
- “Data-HL” for synthetic high- and low-resolution data

