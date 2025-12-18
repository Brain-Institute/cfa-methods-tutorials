clc; clear; close all;

%% --- Load images from a folder ---
inputFolder = 'calib'; % <--- Change this to your actual folder name
imgFiles = dir(fullfile(inputFolder, '*.jpg')); % or *.jpg, *.tif, etc.
N = length(imgFiles);

if N == 0
    error('No images found in the specified folder: %s', inputFolder);
end

fprintf('Found %d images in %s\n', N, inputFolder);

%% --- Save normalized high-resolution images (optional) ---
outputFolder = 'Saved_HR_Images';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

fprintf('Saving normalized HR images to %s ...\n', outputFolder);

images = cell(N, 1);
for i = 1:N
    img = imread(fullfile(inputFolder, imgFiles(i).name));
    
    % Convert to grayscale if needed
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    
    img = im2double(img);
    images{i} = img;
    
    fname = fullfile(outputFolder, sprintf('HR_%04d.png', i));
    imwrite(mat2gray(img), fname);
end

fprintf('Saved %d HR images.\n', N);

%% --- Compute Kullback–Leibler Divergence between image intensity distributions ---

fprintf('Computing pairwise KL divergence between image distributions...\n');

numBins = 320;
edges = linspace(0,1,numBins+1);
pdfs = zeros(numBins, N);

for i = 1:N
    h = histcounts(images{i}(:), edges, 'Normalization', 'probability');
    pdfs(:,i) = h(:) + eps; % Avoid zeros
end

KL_matrix = zeros(N);
for i = 1:N
    for j = i+1:N
        p = pdfs(:,i);
        q = pdfs(:,j);
        KL_pq = sum(p .* log(p ./ q));
        KL_qp = sum(q .* log(q ./ p));
        KL_sym = 0.5 * (KL_pq + KL_qp);
        KL_matrix(i,j) = KL_sym;
        KL_matrix(j,i) = KL_sym;
    end
end

mean_KL = mean(KL_matrix(KL_matrix > 0));
fprintf('Mean symmetric KL divergence between image distributions: %.6f\n', mean_KL);

%% --- Visualization ---
figure;
imagesc(KL_matrix);
colorbar;
title('Pairwise Symmetric KL Divergence Matrix');
xlabel('Image Index');
ylabel('Image Index');
