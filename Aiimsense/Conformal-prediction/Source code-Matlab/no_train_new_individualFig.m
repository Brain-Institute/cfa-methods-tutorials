% MRI_SR_Conformal_Pipeline_Corrected.m
% Domain-aware conformal prediction for MRI super-resolution
% Implements image-level, pixel-level, and patch-level split-conformal
% with normalized k-space + MS-SSIM + edge-weighted residual nonconformity.
%
% Test stage does NOT use HR images; only applies calibrated quantiles.

clc; clear; close all; rng(0);

%% ---------------- 1. Load and prepare data ----------------
trainFolder = fullfile(pwd, 'train');
calibFolder = fullfile(pwd, 'calib');
testFolder  = fullfile(pwd, 'test');

readImages = @(folderPath) ...
    cellfun(@(f) preprocess_image(fullfile(folderPath,f)), ...
            natsortfiles({dir(fullfile(folderPath,'*.jpg')).name}), ...
            'UniformOutput', false);

fprintf('Loading images...\n');
trainImgs = readImages(trainFolder);
calibImgs = readImages(calibFolder);
testImgs  = readImages(testFolder);

trainSet = permute(cat(3, trainImgs{:}), [3 1 2]);
calibSet = permute(cat(3, calibImgs{:}), [3 1 2]);
testSet  = permute(cat(3, testImgs{:}), [3 1 2]);

H = size(trainSet,2);
W = size(trainSet,3);
N_train = size(trainSet,1);
N_calib = size(calibSet,1);
N_test  = size(testSet,1);

fprintf('Loaded train=%d, calib=%d, test=%d images (%dx%d)\n', ...
        N_train, N_calib, N_test, H, W);

%% ---------------- 2. Configuration ----------------
alpha_list = [0.1,0.2, 0.05];
patchSize = 16;
weights = [1, 1, 1];
scale = 4; % SR scale factor

%% ---------------- 3. Define SR network ----------------
layers = [
    imageInputLayer([H W 1],'Normalization','none','Name','input')
    convolution2dLayer(3,32,'Padding','same','Name','conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(3,32,'Padding','same','Name','conv2')
    reluLayer('Name','relu2')
    convolution2dLayer(3,1,'Padding','same','Name','conv3')
    regressionLayer('Name','output')
];
lgraph = layerGraph(layers);

%% ---------------- 4. Train (or load) SR model ----------------
trainModel = false;
if trainModel
    numTrain = N_train;
    XTrain = zeros(H, W, 1, numTrain);
    YTrain = zeros(H, W, 1, numTrain);

    for i = 1:numTrain
        hr = squeeze(trainSet(i,:,:));
        lr_small = imresize(hr, 1/scale, 'bicubic');
        lr_up = imresize(lr_small, [H W], 'bicubic');
        XTrain(:,:,1,i) = lr_up;
        YTrain(:,:,1,i) = hr;
    end

    options = trainingOptions('adam', ...
        'MaxEpochs',200, ...
        'MiniBatchSize',8, ...
        'InitialLearnRate',1e-5, ...
        'Shuffle','every-epoch', ...
        'Verbose',true, ...
        'Plots','training-progress');

    SRnet = trainNetwork(XTrain, YTrain, lgraph, options);
else
    load('my_SRnet.mat','SRnet');
end

%% ---------------- 5. Calibration -----------------
numCalib = N_calib;
k_order = @(n,alpha) ceil((n + 1) * (1 - alpha));

s_kspace = zeros(numCalib,1);
s_msssim = zeros(numCalib,1);
s_edge = zeros(numCalib,1);
residual_maps_cal = zeros(H,W,numCalib);

fprintf('Computing calibration scores...\n');
for i = 1:numCalib
    hr = squeeze(calibSet(i,:,:));
    lr_small = imresize(hr, 1/scale, 'bicubic');
    x = imresize(lr_small, [H W], 'bicubic');
    yhat = squeeze(predict(SRnet, reshape(x,[H W 1 1])));

    s_kspace(i) = kspace_score_raw(yhat, hr);
    s_msssim(i) = msssim_score_raw(yhat, hr);
    [s_edge(i), res_map] = edge_weighted_residual(yhat, hr);
    residual_maps_cal(:,:,i) = abs(yhat - hr);
end

% --- Normalize and combine scores ---
med_k = median(s_kspace); mad_k = mad(s_kspace,1)+eps;
med_m = median(s_msssim); mad_m = mad(s_msssim,1)+eps;
med_e = median(s_edge);   mad_e = mad(s_edge,1)+eps;

s_k_n = (s_kspace - med_k) ./ mad_k;
s_m_n = (s_msssim - med_m) ./ mad_m;
s_e_n = (s_edge   - med_e) ./ mad_e;

scores_calib_image =...
    weights(1)*s_k_n + weights(2)*s_m_n + weights(3)*s_e_n;

% --- Image-level quantiles ---
q_image = containers.Map('KeyType','double','ValueType','double');
for a = alpha_list
    kidx = k_order(numCalib, a);
    s_sorted = sort(scores_calib_image,'ascend');
    q_image(a) = s_sorted(kidx);
end

% --- Pixel-level quantiles ---
fprintf('Computing per-pixel thresholds...\n');
res_sorted = sort(residual_maps_cal,3,'ascend');
q_pixel_map = zeros(H,W,numel(alpha_list));
for ia = 1:numel(alpha_list)
    a = alpha_list(ia);
    kidx = k_order(numCalib, a);
    q_pixel_map(:,:,ia) = res_sorted(:,:,kidx);
end

% --- Patch-level quantiles ---
if mod(H,patchSize)~=0 || mod(W,patchSize)~=0
    error('patchSize must divide H and W exactly.');
end
nPy = H/patchSize; nPx = W/patchSize;
res_patch_cal = zeros(nPy, nPx, numCalib);
for i = 1:numCalib
    rm = residual_maps_cal(:,:,i);
    for py = 1:nPy
        for px = 1:nPx
            y0 = (py-1)*patchSize+1; x0 = (px-1)*patchSize+1;
            block = rm(y0:y0+patchSize-1, x0:x0+patchSize-1);
            res_patch_cal(py,px,i) = mean(block(:));
        end
    end
end
res_patch_sorted = sort(res_patch_cal,3,'ascend');
q_patch_map = zeros(nPy,nPx,numel(alpha_list));
for ia = 1:numel(alpha_list)
    a = alpha_list(ia);
    kidx = k_order(numCalib, a);
    q_patch_map(:,:,ia) = res_patch_sorted(:,:,kidx);
end

%% ---------------- 6. Test Stage (NO HR USED) ---------------------
fprintf('Running test-time prediction intervals (no HR used)...\n');
numTest = N_test;

for ia = 1:numel(alpha_list)
    a = alpha_list(ia);
    q_img = q_image(a);
    q_pix = q_pixel_map(:,:,ia);
    q_pat = q_patch_map(:,:,ia);

    for i = 1:numTest
        x = squeeze(testSet(i,:,:)); % test LR image
        lr_small = imresize(x, 1/scale, 'bicubic');
        x_up = imresize(lr_small, [H W], 'bicubic');
        yhat = squeeze(predict(SRnet, reshape(x_up,[H W 1 1])));
        imwrite(yhat,'test-SR result.png');
        imwrite(x,'test-GT.png');
        test_psnr(i) = psnr(double(yhat), x)
        test_ssim(i) = ssim(double(yhat), x)



        % --- Prediction intervals ---
        upper_pix = yhat + q_pix;
        lower_pix = yhat - q_pix;

        upper_img = yhat + q_img;
        lower_img = yhat - q_img;

        q_patch_up = imresize(q_pat, [H W], 'nearest');
        upper_patch = yhat + q_patch_up;
        lower_patch = yhat - q_patch_up;

        % --- Optional adaptive scaling by gradient magnitude ---
        grad_mag = imgradient(yhat);
        complexity = rescale(grad_mag, 0.8, 1.2);
        upper_pix_adapt = yhat + q_pix .* complexity;
        lower_pix_adapt = yhat - q_pix .* complexity;

        % --- Visualization ---
        % % % % % if i <= 2
        % % % % %     figure('Name',sprintf('Conformal Bounds (α=%.2f, Test=%d)', a, i));
        % % % % %     subplot(2,3,1); imshow(yhat-x,[]); title('residual GT and Prediction');
        % % % % %     subplot(2,3,2); imshow(upper_pix - lower_pix,[]); title('Pixel-level Uncertainty');
        % % % % %     subplot(2,3,3); imshowpair(lower_pix, upper_pix); title('CI - confidence interval bounds');
        % % % % % 
        % % % % %     % subplot(2,3,4); imshow(upper_img - lower_img,[]); title('Image-level Range');
        % % % % %     % subplot(2,3,5); imshow(upper_patch - lower_patch,[]); title('Patch-level Range');
        % % % % %     subplot(2,3,6); imshow(upper_pix_adapt - lower_pix_adapt,[]); title('Adaptive (grad-weighted)');
        % % % % % end
        %%
    
if i <= 4

    %% 1. Residual map (|Prediction − Ground Truth|)
    figure('Name', sprintf('Residual Map | α=%.2f, Test=%d',...
        a, i));
    imagesc(abs(yhat - x));
    axis image off;
    colormap('cool'); colorbar;
    title('Residual Map (abs: Prediction − Ground Truth)');

    %% 2. Pixel-level uncertainty (CI width)
    ci_width = upper_pix - lower_pix;
    figure('Name', ...
        sprintf('Pixel-level Uncertainty | α=%.2f, Test=%d', ...
        a, i));
    imagesc(ci_width);
    axis image off;
    colormap('cool'); colorbar;
    title('Pixel-level CI Width (Uncertainty Map)');

    %% 3. Predicted SR vs Ground Truth (side-by-side)
    % figure('Name', sprintf('SR vs GT | α=%.2f, Test=%d', a, i));
    % imshowpair(yhat, x, 'montage');
    % title('Predicted SR (Left) vs Ground Truth (Right)');

    %% 4. Uncertainty overlay on SR prediction
    overlay_unc = imfuse(yhat, mat2gray(ci_width), ...
                         'falsecolor', ...
                         'Scaling', 'joint', ...
                         'ColorChannels', [1 2 0]);
    figure('Name', sprintf('SR + Uncertainty Overlay | α=%.2f, Test=%d', a, i));
    imshow(overlay_unc);
    title('SR Prediction with Uncertainty Overlay');
    colorbar;

%     figure('Name', ...
%         sprintf('SR with Uncertainty Overlay | α=%.2f, Test=%d',...
%         a, i));
% 
% imshow(yhat, []); hold on;
% 
% h = imagesc(ci_width);
% colormap(gca, 'hot');
% set(h, 'AlphaData', mat2gray(ci_width) * 0.6);  % transparency
% axis image off;
% 
% cb = colorbar;
% cb.Label.String = 'Prediction Interval Width';
% 
% title('Super-Resolved Image with Pixel-wise Uncertainty Overlay');


    %% 5. Adaptive (gradient-weighted) uncertainty
    % ci_width_adapt = upper_pix_adapt - lower_pix_adapt;
    % figure('Name', sprintf('Adaptive Uncertainty | α=%.2f, Test=%d', a, i));
    % imagesc(ci_width_adapt);
    % axis image off;
    % colormap('parula'); colorbar;
    % title('Adaptive (Gradient-weighted) Uncertainty');

end


    

        %%
    end
end

% fprintf('Conformal prediction intervals computed (no HR used in test stage).\n');

%% ---------------- 7. Helper functions -----------------
function s_k = kspace_score_raw(y_pred, y_gt)
    Kp = fft2(y_pred);
    Kg = fft2(y_gt);
    W = fftshift(abs(frequency_grid(size(y_pred))));
    diffMag = abs(Kp - Kg);
    num = norm(W .* diffMag,'fro');
    denom = norm(W .* abs(Kg),'fro') + eps;
    s_k = num / denom;
end

function F = frequency_grid(sz)
    [X,Y] = meshgrid(linspace(-1,1,sz(2)), linspace(-1,1,sz(1)));
    F = sqrt(X.^2 + Y.^2);
end

function s_m = msssim_score_raw(y_pred, y_gt)
    try
        sim = multissim(y_pred,y_gt);
    catch
        sim = 1 - mean(abs(y_pred(:)-y_gt(:)));
        sim = max(min(sim,1),0);
    end
    s_m = 1 - sim;
    s_m = min(max(s_m,0),1);
end

function [s_e, res_map] = edge_weighted_residual(y_pred, y_gt)
    res_map = abs(y_pred - y_gt);
    edges = imgradient(y_gt);
    edges = edges / (max(edges(:))+eps);
    w = 1 + edges;
    weighted = w .* res_map;
    s_e = mean(weighted(:));
end

function I = preprocess_image(path)
    I = imread(path);
    if size(I,3) == 3
        I = rgb2gray(I);
    end
    I = im2double(I);
    I = imresize(I, [320 320]);
end

function A = natsortfiles(A)
    [~,idx] = sort_nat(A);
    A = A(idx);
end

function [sorted,idx] = sort_nat(c)
    [~,idx] = sort(lower(regexprep(c,'\d+','${num2str(str2double($0)+1e6)}')));
    sorted = c(idx);
end
