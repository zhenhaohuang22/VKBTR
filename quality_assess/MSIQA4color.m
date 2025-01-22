function [rse, psnr, ssim] = MSIQA4color(imagery1, imagery2)

for i = 1:size(imagery1,4)
    RSElist(i,1) = perfscore(imagery1(:,:,:,i), imagery2(:,:,:,i));
    PSNRlist(i,1) = PSNR_RGB(imagery1(:,:,:,i), imagery2(:,:,:,i));
    SSIMlist(i,1) = ssim_index(rgb2gray(uint8(imagery1(:,:,:,i))),rgb2gray(uint8(imagery2(:,:,:,i))));
end
rse = mean(RSElist);
psnr = mean(PSNRlist);
ssim = mean(SSIMlist);