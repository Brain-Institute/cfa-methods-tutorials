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

[**PragmaClin Research Inc.**](https://www.pragmaclin.com) is a Canadian clinical research technology company headquartered in St. John's, Newfoundland and Labrador, with an office in Portland, Maine. PragmaClin is developing tools to support more objective and accessible neurological assessments, and working with clinical and academic partners to explore how computer vision, machine learning, and digital technologies can support the measurement of neurological symptoms.

One of the company's primary areas of development is PRIMS® (PragmaClin Real-time Intelligent Medical System), a contactless assessment platform designed to capture and quantify movement characteristics using depth-sensing cameras and machine learning. Initially focused on Parkinson's disease, PRIMS is being developed to support more consistent, quantitative assessments of motor symptoms and their progression over time.

PragmaClin's published usability work is available in [JMIR Human Factors (2024)](https://pubmed.ncbi.nlm.nih.gov/38787603/).

### Research and image processing

This repository showcases exploratory image-processing methods for extracting quantitative characteristics from handwriting and spiral drawings. These tasks are relevant to neurological research because changes in handwriting, drawing patterns, and fine motor control can provide measurable information about movement characteristics.

The notebooks demonstrate how computer vision techniques can be applied to scanned images to identify and quantify features such as:

- **Stroke characteristics:** Width, thickness variation, and continuity of writing or drawing strokes.
- **Spatial organization:** Letter-group spacing, line positioning, and baseline characteristics.
- **Drawing patterns:** Separation of a participant's pen trace from a printed guide spiral.
- **Movement-related features:** Image-derived characteristics that may support further investigation of fine motor control and tremor.

The methods presented here are intended for research and technical exploration. They illustrate some of the image-processing approaches that can contribute to quantitative neurological assessment, alongside other sensing modalities and assessment methods being explored by PragmaClin.

### PRIMS and clinical research

PRIMS uses depth-sensing technology rather than scanned paper images to capture movement during standardized clinical assessments. Although the methods in this repository are not components of the PRIMS platform, they reflect a broader research interest in applying computational methods to make neurological measurements more objective, reproducible, and accessible.

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
