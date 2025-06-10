clc;
clear;
% 获取当前文件的根目录
rootDir = fileparts(mfilename('fullpath'));

% 将根目录及其所有子目录添加到路径
addpath(genpath(rootDir));
img = double(imread('airplane.bmp'));
img = img/255;
missing_ratio = 0.9;
sz = size(img);
mask = genMask(sz, missing_ratio);
tObs = mask .* img;
i = 1;
opt2.iter1 = 100;
opt2.init = 'rand';
opt2.initScale = 0.5;
opt2.epsilon = -1.0;
opt2.trun = 1e-4;
opt2.isPrune = true;
opt2.isELBO = false;
opt2.pruneMethod = 'absolute';
opt2.tol = 1e-3;

for n = 1:2
    theta = 1e1;
    for ii = 1 : sz(n)
        for jj = 1 : sz(n)
            Lu(ii,jj) = exp(-(ii-jj)^2/theta^2);
        end
    end
    Ku{n} = Lu;
end
Ku{3} = eye(sz(3));
RInit = [10,10,10];
tic;
[model] = VKBTR(tObs, mask, RInit, Ku, opt2);
Timelist(i,1) = toc;
X_VKBTR = coreten2tr(model.G);
RSElist(i,1) = perfscore(X_VKBTR*255, img*255);
PSNRlist(i,1) = PSNR_RGB(X_VKBTR*255,img*255);
SSIMlist(i,1) = ssim_index(rgb2gray(uint8(X_VKBTR*255)),rgb2gray(uint8(img*255)));
