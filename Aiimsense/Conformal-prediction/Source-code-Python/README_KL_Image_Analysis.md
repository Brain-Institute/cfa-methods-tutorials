# KL Image Intensity Distribution Analysis

This repository contains a script designed to analyze image collections by comparing their intensity distributions using **Symmetric Kullback–Leibler (KL) Divergence**.



---

## 1. What is this code used for?
The script evaluates the similarity between images based on their pixel intensity probability density functions (PDFs). Specifically, the code:

* Reads grayscale or RGB images from a designated folder.
* Converts all images to grayscale (if necessary).
* Normalizes images to the range [0, 1].
* Saves the normalized high-resolution (HR) images.
* Computes pairwise symmetric KL divergence between image intensity PDFs.
* Produces a KL divergence matrix visualization (heatmap).

### Understanding the Results
* Lower KL Divergence: Indicates more similar intensity distributions.
* Higher KL Divergence: Indicates stronger distributional differences.

---

## 2. How to Run the Code (PyCharm / Python)

### Step 1: Folder Structure
Ensure your project directory is organized as follows:

project_folder/
│── kl_image_divergence.py
│── calib/
│    ├── image1.jpg
│    ├── image2.jpg
│    └── ...

Note: The 'calib/' folder must contain input images (JPG format by default).

### Step 2: Install Required Packages
Open your terminal and run the following command to install dependencies:

pip install numpy matplotlib scikit-image

### Step 3: Execute the Script
Run the analysis using:

python kl_image_divergence.py

---

## 3. Outputs
The script generates three primary outputs:

1. Normalized Images: Saved in 'Saved_HR_Images/' as HR_0001.png, HR_0002.png, etc.
2. Terminal Output: Displays the number of images found and the mean symmetric KL divergence.
3. Visualization: A heatmap showing pairwise symmetric KL divergence between all images in the dataset.



---

## 4. Technical Details
* Histogram Bins: Set to 320.
* Metric Symmetry: Symmetric KL divergence is used to ensure consistency.
* Numerical Stability: Small epsilon values are added to avoid log(0) errors.

---

## 5. Typical Applications in Research
This method is commonly utilized for:

* Medical Imaging: Quantifying domain shift across different MRI scanners.
* Dataset Analysis: Identifying outlier images or verifying calibration/test set homogeneity.
* Image Processing: Pre-analysis for super-resolution or reconstruction pipelines.
* Reliability: Supporting uncertainty-aware or conformal prediction pipelines for high-level publications.