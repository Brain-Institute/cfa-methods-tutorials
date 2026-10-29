# Handwriting and Spiral Analysis

This repository contains two Python notebooks that process handwriting and spiral drawings from scanned images. They extract measurements from the lines and create plots to help study writing patterns and fine motor control.

## Notebooks

### 1. Parkinson's Handwriting Analysis

This notebook processes a scanned handwriting sample.

It:
- Converts the image to grayscale and resizes it to a fixed width.
- Separates ink from the paper using automatic thresholding.
- Creates a thin centre line (skeleton) for each stroke.
- Measures stroke width using a distance transform.
- Detects text lines and groups of connected letters.
- Calculates features such as stroke-width variation, spacing between letter groups, baseline slope, and baseline wobble.
- Displays plots showing the traces, spacing, and writing slope.

### 2. Automatic Spiral Analysis

This notebook processes a scanned spiral drawing that contains a printed guide and a patient's pen line.

It:
- Resizes the image to a fixed width and converts it to grayscale.
- Separates ink from the paper using Otsu thresholding.
- Measures stroke widths to help distinguish the thicker printed guide from the thinner pen line.
- Builds a guide mask and removes the guide from the image.
- Displays plots showing the detected guide and the remaining pen ink.

The extracted pen line can be used as a starting point for further cleaning and analysis of drawing accuracy and tremor.

## Requirements

The notebooks use Python and the following libraries:

- OpenCV (`opencv-python`)
- NumPy
- Pandas
- Matplotlib
- scikit-image
- SciPy

Install the dependencies with:

```bash
pip install opencv-python numpy pandas matplotlib scikit-image scipy
```

## How to Run

1. Clone or download this repository.
2. Install the required libraries.
3. Open a notebook in Jupyter Notebook, JupyterLab, or Visual Studio Code.
4. Update `IMAGE_PATH` in the notebook to point to your own scanned image.
5. Run the notebook cells in order and review the plots and printed measurements.

## Image Input

Both notebooks expect a scanned image file. The image path in each notebook is currently set to a local example path, so change it before running the notebook on another computer.

The notebooks resize images to a fixed working width. Measurements are reported in pixels at that resized resolution, so scans should be captured at a similar scale when comparing results.

## Notes

These notebooks are for image processing and research exploration. The extracted measurements are not, by themselves, a diagnosis of Parkinson's disease. The methods and features need to be tested and validated before being used for clinical decisions.
