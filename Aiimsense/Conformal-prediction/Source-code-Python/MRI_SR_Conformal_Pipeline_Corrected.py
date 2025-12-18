
"""
MRI_SR_Conformal_Pipeline_Corrected.py

Domain-aware conformal prediction for MRI super-resolution.
Python (PyTorch + NumPy + scikit-image) translation of the provided MATLAB code.
All logical parts are preserved:
- Data loading
- SR network definition
- Optional training or loading
- Image-, pixel-, and patch-level split conformal calibration
- Test-time prediction intervals (NO HR used)
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import io, color
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.fft import fft2, fftshift
from scipy.ndimage import sobel
import matplotlib.pyplot as plt

# ---------------- 0. Reproducibility ----------------
np.random.seed(0)
torch.manual_seed(0)

# ---------------- 1. Load and prepare data ----------------
def preprocess_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float64) / 255.0
    img = resize(img, (320, 320), anti_aliasing=True)
    return img

def read_images(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return [preprocess_image(f) for f in files]

base_dir = os.getcwd()
trainFolder = os.path.join(base_dir, "train")
calibFolder = os.path.join(base_dir, "calib")
testFolder  = os.path.join(base_dir, "test")

print("Loading images...")
trainImgs = read_images(trainFolder)
calibImgs = read_images(calibFolder)
testImgs  = read_images(testFolder)

trainSet = np.stack(trainImgs, axis=0)
calibSet = np.stack(calibImgs, axis=0)
testSet  = np.stack(testImgs, axis=0)

N_train, H, W = trainSet.shape
N_calib = calibSet.shape[0]
N_test  = testSet.shape[0]

print(f"Loaded train={N_train}, calib={N_calib}, test={N_test} images ({H}x{W})")

# ---------------- 2. Configuration ----------------
alpha_list = [0.1, 0.2, 0.05]
patchSize = 16
weights = [1, 1, 1]
scale = 4

# ---------------- 3. Define SR network ----------------
class SimpleSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

SRnet = SimpleSRNet()

# ---------------- 4. Train (or load) SR model ----------------
trainModel = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRnet = SRnet.to(device)

if trainModel:
    optimizer = optim.Adam(SRnet.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    XTrain = []
    YTrain = []
    for i in range(N_train):
        hr = trainSet[i]
        lr_small = resize(hr, (H//scale, W//scale), anti_aliasing=True)
        lr_up = resize(lr_small, (H, W), anti_aliasing=True)
        XTrain.append(lr_up)
        YTrain.append(hr)

    XTrain = torch.tensor(np.array(XTrain)[:, None, :, :], dtype=torch.float32).to(device)
    YTrain = torch.tensor(np.array(YTrain)[:, None, :, :], dtype=torch.float32).to(device)

    for epoch in range(200):
        optimizer.zero_grad()
        out = SRnet(XTrain)
        loss = criterion(out, YTrain)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.6f}")

    torch.save(SRnet.state_dict(), "my_SRnet.pth")
else:
    SRnet.load_state_dict(torch.load("my_SRnet.pth", map_location=device))

SRnet.eval()

# ---------------- 5. Calibration ----------------
def frequency_grid(h, w):
    y, x = np.meshgrid(np.linspace(-1,1,h), np.linspace(-1,1,w), indexing='ij')
    return np.sqrt(x**2 + y**2)

def kspace_score_raw(y_pred, y_gt):
    Kp = fft2(y_pred)
    Kg = fft2(y_gt)
    W = fftshift(frequency_grid(*y_pred.shape))
    num = np.linalg.norm(W * np.abs(Kp - Kg))
    denom = np.linalg.norm(W * np.abs(Kg)) + 1e-12
    return num / denom

def msssim_score_raw(y_pred, y_gt):
    sim = ssim(y_gt, y_pred, data_range=1.0)
    return 1 - sim

def edge_weighted_residual(y_pred, y_gt):
    res = np.abs(y_pred - y_gt)
    gx = sobel(y_gt, axis=0)
    gy = sobel(y_gt, axis=1)
    edges = np.sqrt(gx**2 + gy**2)
    edges /= edges.max() + 1e-12
    w = 1 + edges
    return np.mean(w * res), res

k_order = lambda n, a: int(np.ceil((n + 1) * (1 - a))) - 1

s_kspace = np.zeros(N_calib)
s_msssim = np.zeros(N_calib)
s_edge = np.zeros(N_calib)
residual_maps_cal = np.zeros((N_calib, H, W))

print("Computing calibration scores...")
for i in range(N_calib):
    hr = calibSet[i]
    lr_small = resize(hr, (H//scale, W//scale), anti_aliasing=True)
    x = resize(lr_small, (H, W), anti_aliasing=True)

    with torch.no_grad():
        yhat = SRnet(torch.tensor(x[None,None,:,:], dtype=torch.float32).to(device)).cpu().numpy()[0,0]

    s_kspace[i] = kspace_score_raw(yhat, hr)
    s_msssim[i] = msssim_score_raw(yhat, hr)
    s_edge[i], res_map = edge_weighted_residual(yhat, hr)
    residual_maps_cal[i] = res_map

# Normalize scores
def mad(x):
    return np.median(np.abs(x - np.median(x))) + 1e-12

s_k_n = (s_kspace - np.median(s_kspace)) / mad(s_kspace)
s_m_n = (s_msssim - np.median(s_msssim)) / mad(s_msssim)
s_e_n = (s_edge   - np.median(s_edge))   / mad(s_edge)

scores_calib_image = weights[0]*s_k_n + weights[1]*s_m_n + weights[2]*s_e_n

q_image = {}
for a in alpha_list:
    s_sorted = np.sort(scores_calib_image)
    q_image[a] = s_sorted[k_order(N_calib, a)]

# Pixel-level
q_pixel_map = np.zeros((len(alpha_list), H, W))
res_sorted = np.sort(residual_maps_cal, axis=0)
for ia, a in enumerate(alpha_list):
    q_pixel_map[ia] = res_sorted[k_order(N_calib, a)]

# Patch-level
nPy, nPx = H//patchSize, W//patchSize
res_patch = np.zeros((N_calib, nPy, nPx))
for i in range(N_calib):
    rm = residual_maps_cal[i]
    for py in range(nPy):
        for px in range(nPx):
            block = rm[py*patchSize:(py+1)*patchSize,
                       px*patchSize:(px+1)*patchSize]
            res_patch[i,py,px] = block.mean()

q_patch_map = np.zeros((len(alpha_list), nPy, nPx))
res_patch_sorted = np.sort(res_patch, axis=0)
for ia, a in enumerate(alpha_list):
    q_patch_map[ia] = res_patch_sorted[k_order(N_calib, a)]

# ---------------- 6. Test Stage (NO HR USED) ----------------
print("Running test-time prediction intervals (no HR used)...")

for ia, a in enumerate(alpha_list):
    for i in range(N_test):
        x = testSet[i]
        lr_small = resize(x, (H//scale, W//scale), anti_aliasing=True)
        x_up = resize(lr_small, (H, W), anti_aliasing=True)

        with torch.no_grad():
            yhat = SRnet(torch.tensor(x_up[None,None,:,:], dtype=torch.float32).to(device)).cpu().numpy()[0,0]

        io.imsave("test-SR-result.png", np.clip(yhat,0,1))
        io.imsave("test-GT.png", np.clip(x,0,1))

        print("PSNR:", psnr(x, yhat, data_range=1.0),
              "SSIM:", ssim(x, yhat, data_range=1.0))

        q_pix = q_pixel_map[ia]
        upper = yhat + q_pix
        lower = yhat - q_pix

        if i < 4:
            plt.figure()
            plt.imshow(upper - lower, cmap="cool")
            plt.title(f"Pixel-level uncertainty α={a}, test={i}")
            plt.colorbar()
            plt.show()

print("Done.")
