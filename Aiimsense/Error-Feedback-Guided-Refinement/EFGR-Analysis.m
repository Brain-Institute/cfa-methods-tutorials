clc; clear; close all;

%% Load images
% my_orig  = imread('myoriginal.png','png');
% my_EDSR  = imread('EDSRoutput.png','png');
% my_LRUP  = imread('LRimage_sameSizeAsOrig.png','png');
% my_LRdown = imread('LRimage_small.png','png');
% my_bic = imresize(my_LRdown,[size(my_EDSR,1) size(my_EDSR,2)],"triangle");



my_orig  = imread('brain_test_01.png','png');
my_EDSR  = imread('brain_test_01_LR_1_4_bicubic_x4_SR.png','png');
my_LRUP  = imread('brain_test_01_HR_X4_bicubic.png','png');
my_LRdown = imread('brain_test_01_LR_1_4_bicubic.png','png');

% -------- Ensure all images are 3-channel --------
if size(my_orig,3) ~= 3
    my_orig = cat(3, my_orig, my_orig, my_orig);
end

if size(my_EDSR,3) ~= 3
    my_EDSR = cat(3, my_EDSR, my_EDSR, my_EDSR);
end

if size(my_LRUP,3) ~= 3
    my_LRUP = cat(3, my_LRUP, my_LRUP, my_LRUP);
end

if size(my_LRdown,3) ~= 3
    my_LRdown = cat(3, my_LRdown, my_LRdown, my_LRdown);
end
% ------------------------------------------------

my_bic = my_LRUP;

%% Assign to variables
LR = im2double(my_LRdown);    % Low-resolution (input)
SR = im2double(my_EDSR);      % EDSR output (SR result)
HR = im2double(my_orig);      % Ground truth

%%

my_psnrrrr = psnr(im2double(my_bic),HR)
%% -----------------------------
% Frequency–Adaptive Error Feedback Guided Refinement (FA–EFGR)
% with separate evaluation of spatial-only and frequency-only cases
% -----------------------------

% Step 1: Downscale SR to LR size (simulate degradation)
SR_down = imresize(SR, [size(LR,1), size(LR,2)], ...
    'bicubic');

% Step 2: Compute reconstruction error at LR scale
err_LR = SR_down - LR;

% Step 3: Upsample error to HR size
err_up = imresize(err_LR, [size(SR,1), size(SR,2)], ...
    'bicubic');

% Step 4: Spatial correction map α_s(x,y)
graySR = rgb2gray(SR);                 % convert to gray for edge map
[Gx, Gy] = gradient(graySR);
grad_mag = sqrt(Gx.^2 + Gy.^2);        % gradient magnitude

grad_norm = grad_mag ./ (max(grad_mag(:)) + eps);

alpha_max = 2;%0.25;                      % stronger correction in smooth regions
alpha_min = 0.05;                      % weaker correction near edges
alpha_spatial = ...
    alpha_max - (alpha_max - alpha_min) * grad_norm;

% Step 5: Frequency-domain weighting map β_f(x,y)
lap = imfilter(graySR, fspecial('laplacian', 0.9), ...
    'replicate');
freq_energy = abs(lap);
freq_norm = freq_energy ./ (max(freq_energy(:)) + eps);

beta_max = 1.0;                        % full correction at low frequencies
beta_min = 0.1;                        % less correction at high frequencies
beta_freq = ...
    beta_max - (beta_max - beta_min) * ...
    freq_norm;

% Step 6: Combine spatial and frequency maps
% alpha_combined = alpha_spatial .* beta_freq;

% If RGB, expand maps to 3 channels
% if size(SR,3) == 3
%     alpha_spatial  = repmat(alpha_spatial, [1, 1, 3]);
%     beta_freq      = repmat(beta_freq, [1, 1, 3]);
%     alpha_combined = repmat(alpha_combined, [1, 1, 3]);
% end

%% -----------------------------
% Step 7: Apply post-processing refinements
% -----------------------------
I_AEFGR  = SR - alpha_spatial  .* err_up;   % Spatial-only
I_FEFGR  = SR - beta_freq      .* err_up;   % Frequency-only
% I_FAEFGR = SR - alpha_combined .* err_up;   % Combined spatial + frequency

%% -----------------------------
% Step 8: Evaluate PSNR and SSIM
% -----------------------------
psnr_SR      = psnr(SR, HR);
psnr_AEFGR   = psnr(I_AEFGR, HR);
psnr_FEFGR   = psnr(I_FEFGR, HR);
% psnr_FAEFGR  = psnr(I_FAEFGR, HR);

ssim_SR      = ssim(SR, HR);
ssim_AEFGR   = ssim(I_AEFGR, HR);
ssim_FEFGR   = ssim(I_FEFGR, HR);
% ssim_FAEFGR  = ssim(I_FAEFGR, HR);

fprintf('\n==== Evaluation Results ====\n');
fprintf('PSNR (EDSR)     = %.4f dB\n', psnr_SR);
fprintf('PSNR (A-EFGR)   = %.4f dB\n', psnr_AEFGR);
fprintf('PSNR (F-EFGR)   = %.4f dB\n', psnr_FEFGR);
% fprintf('PSNR (FA-EFGR)  = %.4f dB\n', psnr_FAEFGR);
fprintf('SSIM (EDSR)     = %.4f\n',   ssim_SR);
fprintf('SSIM (A-EFGR)   = %.4f\n',   ssim_AEFGR);
fprintf('SSIM (F-EFGR)   = %.4f\n',   ssim_FEFGR);
% fprintf('SSIM (FA-EFGR)  = %.4f\n',   ssim_FAEFGR);

%% -----------------------------
% Step 9: Visualization
% -----------------------------
figure('Name','Comparison of Refinement Methods','NumberTitle','off');
subplot(2,3,1); imshow(SR);          title('EDSR Output');
subplot(2,3,2); imshow(I_AEFGR);     title('S–EFGR (Spatial Only)');
subplot(2,3,3); imshow(I_FEFGR);     title('F–EFGR (Frequency Only)');
% subplot(2,3,4); imshow(I_FAEFGR);    title('FA–EFGR (Combined)');
subplot(2,3,5); imshow(alpha_spatial(:,:,1),[]); title('Spatial α Map');
subplot(2,3,6); imshow(beta_freq(:,:,1),[]);     title('Frequency β Map');
colormap jet; colorbar;
