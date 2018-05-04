% evaluation - FBPConvNet
% modified from MatconvNet (ver.23)
% 22 June 2017
% contact : Kyong Jin (kyonghwan.jin@gmail.com)

clear
restoredefaultpath
reset(gpuDevice(1))
run ./matconvnet-1.0-beta23/matlab/vl_setupnn

load('./pretrain/net-epoch-151.mat')

% cmode='gpu'; % 'cpu'
cmode='cpu';
if strcmp(cmode,'gpu')
    net = vl_simplenn_move(net, 'gpu') ;
else
    net = vl_simplenn_move(net, 'cpu') ;
end

avg_psnr_m=zeros(1,1);
avg_psnr_rec=zeros(1,1);

magnification = 8;
default_size = 512;
image_size = default_size * magnification;

true_gt = phantom(image_size) * 255.0 - 128.0;
r1000=radon(true_gt, 0:180/(1000 * magnification):180-180/(1000 * magnification));
tic;
gt=iradon(r1000, 0:180/(1000 * magnification):180-180/(1000 * magnification), 'Ram-Lak', 1, image_size);
num = round(1000 * magnification / 7.0);
iradon1000_time = toc;
r143=radon(true_gt, 0:180.0/num:180-180.0/num);
tic;
m=iradon(r143, 0:180.0/num:180-180.0/num, 'Ram-Lak', 1, image_size);
iradon143_time = toc;
if strcmp(cmode,'gpu')
    res=vl_simplenn_fbpconvnet_eval(net,gpuArray((single(m))));
    rec=gather(res(end-1).x)+m;
else
    if image_size == 512
        tic;
        res=vl_simplenn_fbpconvnet_eval(net,((single(m))));
        rec=(res(end-1).x)+m;
        fbp_conv_time = toc;
    else
        tic;
        rec = m;
        fbp_conv_time = toc;
    end
end

snr_m=computeRegressedSNR(m,gt);
tsnr_m=computeRegressedSNR(m,true_gt);
snr_rec=computeRegressedSNR(rec,gt);
tsnr_rec=computeRegressedSNR(rec,true_gt);
snr_gt=computeRegressedSNR(gt,true_gt);
figure(1), 
subplot(141), imagesc(m), colormap(gray), axis equal tight, title({'fbp';num2str(snr_m);num2str(tsnr_m);['time ', num2str(iradon143_time)];})
subplot(142), imagesc(rec),axis equal tight, title({'fbpconvnet';num2str(snr_rec);num2str(tsnr_rec);['time ', num2str(iradon143_time + fbp_conv_time)];})
subplot(143), imagesc(gt),axis equal tight, title({'gt'; 'fbp 1000view'; num2str(snr_gt); ['time ', num2str(iradon1000_time)]})
subplot(144), imagesc(true_gt),axis equal tight, title(['true gt'])
%subplot(131), imshow(m),axis equal tight, title({'fbp';num2str(snr_m)})
%subplot(132), imshow(rec),axis equal tight, title({'fbpconvnet';num2str(snr_rec)})
%subplot(133), imshow(gt),axis equal tight, title(['gt ' num2str(0)])

display(['SNR (FBP) : ' num2str(snr_m)])
display(['SNR (FBPconvNet) : ' num2str(snr_rec)])
