clc;
clear;


load('claire.mat');
img = double(claire(:,:,:,1:30));
img = img/255;
missing_ratio = 0.9; 
sz = size(img);
indice = randperm(sz(4), sz(4)*0.2);
mask1 = ones(sz);
mask1(:,:,:,indice) = 0;
mask2 = genMask(sz, missing_ratio);
X_noise = imnoise(img,'gaussian',0,0.01);
mask = mask1 .* mask2;
tObs = mask .* X_noise;

tObs3 = reshape(tObs,sz(1),sz(2),sz(3)*sz(4));
mask3 = reshape(mask,sz(1),sz(2),sz(3)*sz(4));
i = 1;
opt2.iter1 = 100;
opt2.init = 'rand';
opt2.initScale = 0.5;
opt2.epsilon = -1.0;
opt2.trun = 1e-5;
opt2.isPrune = true;
opt2.pruneMethod = 'absolute';
opt2.tol = 1e-3;

for n = [1,2,4]
    theta = 1e1;
    Lu = zeros(sz(n),sz(n));
    for ii = 1 : sz(n)
        for jj = 1 : sz(n)
            Lu(ii,jj) = exp(-(ii-jj)^2/theta^2);
        end
    end
    Ku{n} = Lu;
end
Ku{3} = eye(sz(3));

RInit = [10,10,10,10];
tic;
[model] = VKBTR(tObs, mask, RInit, Ku, opt2);
Timelist(i,1) = toc;
X_VKBTR = coreten2tr(model.G);
[RSElist(i,1),PSNRlist(i,1), SSIMlist(i,1)] = MSIQA4color(X_VKBTR*255, img*255);

