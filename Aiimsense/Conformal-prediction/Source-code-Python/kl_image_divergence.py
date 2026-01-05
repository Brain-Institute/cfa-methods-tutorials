
import os
import numpy as np
from skimage import io, color, img_as_float
import matplotlib.pyplot as plt

def main():
    # --- Load images from a folder ---
    input_folder = 'calib'  # Change if needed
    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    N = len(img_files)

    if N == 0:
        raise RuntimeError(f'No images found in the specified folder: {input_folder}')

    print(f'Found {N} images in {input_folder}')

    # --- Save normalized high-resolution images (optional) ---
    output_folder = 'Saved_HR_Images'
    os.makedirs(output_folder, exist_ok=True)

    print(f'Saving normalized HR images to {output_folder} ...')

    images = []

    for i, fname in enumerate(img_files, start=1):
        img = io.imread(os.path.join(input_folder, fname))

        # Convert to grayscale if needed
        if img.ndim == 3:
            img = color.rgb2gray(img)

        img = img_as_float(img)
        images.append(img)

        out_name = os.path.join(output_folder, f'HR_{i:04d}.png')
        io.imsave(out_name, np.clip(img, 0, 1))

    print(f'Saved {N} HR images.')

    # --- Compute Kullback–Leibler Divergence ---
    print('Computing pairwise KL divergence between image distributions...')

    num_bins = 320
    edges = np.linspace(0, 1, num_bins + 1)
    pdfs = np.zeros((num_bins, N))

    for i, img in enumerate(images):
        hist, _ = np.histogram(img.ravel(), bins=edges, density=True)
        hist = hist / (hist.sum() + np.finfo(float).eps)
        pdfs[:, i] = hist + np.finfo(float).eps  # avoid zeros

    KL_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            p = pdfs[:, i]
            q = pdfs[:, j]
            kl_pq = np.sum(p * np.log(p / q))
            kl_qp = np.sum(q * np.log(q / p))
            kl_sym = 0.5 * (kl_pq + kl_qp)
            KL_matrix[i, j] = kl_sym
            KL_matrix[j, i] = kl_sym

    mean_KL = KL_matrix[KL_matrix > 0].mean()
    print(f'Mean symmetric KL divergence between image distributions: {mean_KL:.6f}')

    # --- Visualization ---
    plt.figure()
    plt.imshow(KL_matrix)
    plt.colorbar()
    plt.title('Pairwise Symmetric KL Divergence Matrix')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.show()


if __name__ == '__main__':
    main()
