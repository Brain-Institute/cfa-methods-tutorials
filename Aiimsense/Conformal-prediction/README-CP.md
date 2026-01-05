# MRI_SR_Conformal_Pipeline_Corrected.py Guide

This guide provides instructions for running the MRI Super-Resolution (SR) Conformal Prediction pipeline.

## 1. Folder Structure
Ensure the following structure exists in your project root:

```text
project_root/
├── train/        # Training images (JPG)
├── calib/        # Calibration images (JPG)
├── test/         # Test images (JPG)
├── my_SRnet.pth  # Optional: Pretrained model (if trainModel=False)
└── MRI_SR_Conformal_Pipeline_Corrected.py
```

> **Note:** All images should be grayscale or RGB JPGs; RGB files are automatically converted to grayscale. Images are resized to 320x320 during preprocessing.

---

## 2. Prerequisites
Install the required Python packages using pip:

```bash
pip install numpy torch scikit-image scipy matplotlib
```
* **GPU Support:** Automatic if a GPU is available.

---

## 3. Running the Script
1. Open **PyCharm** and set the project root to the script's folder.
2. Open `MRI_SR_Conformal_Pipeline_Corrected.py`.
3. Execute the script via the UI or terminal:
   `python MRI_SR_Conformal_Pipeline_Corrected.py`

---

## 4. Model Configuration
The script behavior is controlled by the `trainModel` variable:

* **Load Pretrained (Default):** Set `trainModel = False`. Ensure `my_SRnet.pth` exists.
* **Train from Scratch:** Set `trainModel = True`. 
    * Settings: 200 epochs, batch size 8, Adam optimizer, and MSE loss.

---

## 5. Outputs & Visualization
* **Saved Images:** `test-SR-result.png` (SR output) and `test-GT.png` (ground truth).
* **Metrics:** PSNR and SSIM values are printed to the console.
* **Uncertainty:** Pixel-level uncertainty maps and overlays are displayed for the first 4 test images.

---

## 6. Parameters
| Parameter | Description |
| :--- | :--- |
| `alpha_list` | Significance levels for conformal prediction. |
| `patchSize` | Size of patches for scores (must divide image size). |
| `scale` | Super-resolution scaling factor. |
| `weights` | Weights for k-space, MS-SSIM, and edge-based scores. |

---

## 7. References
* [PyTorch](https://pytorch.org/)
* [scikit-image](https://scikit-image.org/)
* MATLAB to Python translation guide for SR networks