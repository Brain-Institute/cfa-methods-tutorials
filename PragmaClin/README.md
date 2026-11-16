# Handwriting and Spiral Analysis

This repository contains two Python notebooks for processing scanned handwriting samples and spiral drawings. The notebooks use image-processing techniques to extract measurable characteristics from writing and drawing patterns and generate plots for research and exploratory analysis.

The project was developed by **PragmaClin Research Inc.** as part of its work on objective, quantitative assessment of neurological movement symptoms.

---

## Table of contents

- [About PragmaClin](#about-pragmaclin)
- [Repository contents](#repository-contents)
- [Image-processing concepts used](#image-processing-concepts-used)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Image input and measurement considerations](#image-input-and-measurement-considerations)
- [Intended use and limitations](#intended-use-and-limitations)
- [License and intellectual property](#license-and-intellectual-property)

---

## About PragmaClin
<img width="220"  alt="PragmaClin-Logo 2026" src="https://github.com/user-attachments/assets/0d7b412f-5c2f-4549-9c9a-ef0827af55a4" />

### Making the invisible measurable

[**PragmaClin Research Inc.**](https://www.pragmaclin.com) is a clinical research technology company headquartered in St. John's, Newfoundland and Labrador, with an office in Portland, Maine. Its work is aligned with United Nations Sustainable Development Goals 3 (Good Health and Well-being) and 10 (Reduced Inequalities).

### The problem

Neurological care can be subjective, inconsistent, and inaccessible. Movement symptoms are often scored by visual observation, and two clinicians observing the same patient may reach different conclusions. Many patients also have limited access to specialist assessment.

### The solution: PRIMS®

**PRIMS®** is a contactless clinical movement-assessment platform. It uses depth-sensing cameras and machine learning to turn standardized movement examinations into objective, quantitative information that clinicians can review and compare over time.

**Contactless · Fast · Scalable · Standardized**

> **Clinical decision support. Sensor-derived. Not diagnostic.**

### Why PRIMS is different

- **Depth sensing:** 3D spatial data can capture movement characteristics, including the intensity of bradykinesia (slowness of movement), that may not be represented fully by 2D video.
- **Beyond motor symptoms:** The platform is designed to assess areas such as speech, cognition, and mood alongside motor symptoms.
- **No patient-worn hardware:** Assessments are performed in the clinic without requiring patients to wear or maintain sensors.
- **Telehealth potential:** The platform is designed to support assessment in settings without dedicated specialist infrastructure.

### What PRIMS measures

Parkinson's disease is the initial model system. PRIMS captures movement and related characteristics such as:

`Tremor` · `Gait` · `Balance` · `Speech` · `Posture` · `Mobility` · `Facial expression` · `Fall risk`

Work is also expanding into stroke, genetic and rare neurological disorders, and general balance and mobility.

### Who benefits

| Audience | Potential value |
| --- | --- |
| **Hospitals and health systems** | Support triage, reduce assessment bottlenecks, and promote consistent scoring across departments. |
| **Clinics and community care** | Make structured assessments more accessible in settings without a neurologist on site. |
| **Researchers and academia** | Support reproducible, multi-site research and quantitative longitudinal outcome measurement. |
| **Pharma and clinical trials** | Explore quantitative endpoints and reduce inter-rater variability across study sites. |

### Clinical evidence and traction

PragmaClin reports the following company-level milestones:

- 90%+ model agreement with clinicians
- 10,000+ data points collected
- 15 of 16 study sites active across 6 countries and 4 continents
- FDA-agreed protocol for the final commercialization study, now underway
- Intellectual property filed
- 3 clinics deployed, with 10 hospital, health-system, and academic partners in pilots, studies, or letters of intent
- Active dialogue with 6 pharma and industry partners

PragmaClin's published usability work is available in [JMIR Human Factors (2024)](https://pubmed.ncbi.nlm.nih.gov/38787603/).

### Work with PragmaClin

PragmaClin welcomes interest from clinical and research sites, researchers and students, hospitals, health systems, and industry partners.

- **Website:** [www.pragmaclin.com](https://www.pragmaclin.com)
- **Email:** [contact@pragmaclin.com](mailto:contact@pragmaclin.com)

---

## Repository contents

This repository contains two Jupyter notebooks.

### 1. Parkinson's Handwriting Analysis

Processes a scanned handwriting sample and extracts image-based characteristics of the writing.

1. **Image preparation:** Loads the scanned image, converts it to grayscale, and resizes it to a fixed working width.
2. **Ink segmentation:** Uses automatic thresholding to separate dark writing strokes from the paper background.
3. **Skeletonization:** Reduces the writing strokes to thin centre lines, giving a simplified representation of each stroke's path.
4. **Stroke-width measurement:** Uses a distance transform to estimate the local thickness of the original ink around the skeleton. The distance from a skeleton pixel to the nearest background pixel estimates the stroke's half-width.
5. **Line and letter-group detection:** Analyses the spatial distribution and connected components of the writing to identify text lines and groups of connected letters.
6. **Feature calculation:** Calculates characteristics such as stroke-width variation, spacing between letter groups, baseline slope, and baseline wobble.
7. **Visualization:** Generates plots showing detected traces, spacing, and writing slope to help inspect the extracted measurements.

### 2. Automatic Spiral Analysis

Processes a scanned spiral drawing containing both a printed guide spiral and a participant's pen trace.

1. **Image preparation:** Resizes the image to a fixed working width and converts it to grayscale.
2. **Ink segmentation:** Applies Otsu thresholding to separate dark marks from the paper background.
3. **Stroke-width analysis:** Uses a distance transform and skeleton-based measurements to estimate local stroke thickness.
4. **Guide identification:** Uses stroke-width information to distinguish the printed guide, which is generally thicker, from the participant's pen line.
5. **Guide-mask generation:** Builds a mask representing the printed guide.
6. **Guide removal:** Removes the detected guide from the binary image, leaving the participant's pen trace for further processing.
7. **Visualization:** Displays the detected guide mask and the remaining pen ink so the result can be visually checked.

The extracted pen trace can serve as the starting point for additional processing, such as path cleanup, drawing-accuracy analysis, and tremor-related measurements.

> **Note:** Guide removal is an image-processing step and should be visually reviewed, as scan quality and overlapping strokes can affect separation.

---

## Image-processing concepts used

| Technique | Purpose |
| --- | --- |
| **Grayscale conversion** | Simplifies the image to intensity values before segmentation. |
| **Otsu thresholding** | Automatically selects an intensity threshold to separate foreground ink from the background. |
| **Morphological operations** | Clean or modify binary shapes, for example by joining small gaps or removing unwanted marks. |
| **Skeletonization** | Reduces strokes to approximately one-pixel-wide centre lines while preserving their overall paths. |
| **Distance transform** | Measures each foreground pixel's distance to the nearest background pixel; values along a skeleton can estimate stroke thickness. |
| **Connected-component analysis** | Finds groups of connected foreground pixels that may represent letters, marks, or other image regions. |
| **Plot overlays** | Displays intermediate masks, detected lines, and measurements on or beside the source image for visual inspection. |

---

## Requirements

The notebooks use Python and the following libraries:

| Library | Package name |
| --- | --- |
| OpenCV | `opencv-python` |
| NumPy | `numpy` |
| Pandas | `pandas` |
| Matplotlib | `matplotlib` |
| scikit-image | `scikit-image` |
| SciPy | `scipy` |

Install the dependencies with:

```bash
pip install opencv-python numpy pandas matplotlib scikit-image scipy
```

Using a virtual environment is recommended so that notebook dependencies stay isolated from other Python projects.

---

## How to run

1. Clone or download this repository.
2. Create and activate a Python virtual environment (recommended).
3. Install the dependencies listed above.
4. Open the relevant notebook in Jupyter Notebook, JupyterLab, or Visual Studio Code.
5. Update the image path variable (for example, `IMAGE_PATH`) to point to your scanned image.
6. Run the notebook cells in order.
7. Review the generated plots, intermediate image masks, and printed measurements.

---

## Image input and measurement considerations

- **Input:** Both notebooks expect a scanned image file. The image path in each notebook may be set to a local example path; update it before running on another computer.
- **Units:** Images are resized to a fixed working width, so measurements are reported in pixels at the resized resolution. They are not automatically converted to physical units such as millimetres.
- **Consistency:** For meaningful comparisons, use a consistent scanning setup, image orientation, crop, and scale.
- **Quality checks:** Image quality can affect thresholding, skeletonization, connected-component detection, and guide removal. Check the intermediate plots to confirm the writing or pen trace has been segmented as expected.

---

## Intended use and limitations

These notebooks are intended for image processing, research, and exploratory analysis. The extracted measurements are not, by themselves, a diagnosis of Parkinson's disease or any other condition, and should not be used as the sole basis for clinical decisions.

The methods and derived features require appropriate testing and validation before any clinical use. Results may be affected by scan resolution, contrast, paper background, pen type, line overlap, image rotation, and other acquisition differences.

> **PRIMS® is an investigational clinical decision-support tool and is not intended for diagnosis. It is not currently cleared or approved by the FDA or Health Canada.**

No participant data is included in this repository. Do not commit identifiable, confidential, or otherwise sensitive participant information.

---

## License and intellectual property

All content within this repository is protected under the Creative Commons Attribution 4.0 International License (CC BY 4.0). For more details, please refer to the full text of the Creative Commons Attribution 4.0 International License available at [Creative Commons](https://creativecommons.org/licenses/by/4.0/deed.en).

---

<sub>© PragmaClin Research Inc. · [www.pragmaclin.com](https://www.pragmaclin.com)</sub>
