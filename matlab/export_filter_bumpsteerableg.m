clear all
close all

addpath ../scatnet-0.2a
addpath_scatnet;

%% get data and estimate spectral
N=64; %56;
J=6; % 8;
L=8; % 8;
filtopts = struct();
filtopts.J=J;
filtopts.L=L;
filtopts.full2pi=1;

filter_id=1;
filtopts.fcenter=0.425; % om in [0,1], unit 2pi
filtopts.gamma1=1;
[filnew,lpal]=bumpsteerableg_wavelet_filter_bank_2d([N N], filtopts);

%% plot lpal
figure; imagesc(fftshift(lpal)); colormap gray
title(sprintf('Littlewood-Paley: xi0=%g*2pi',filtopts.fcenter))
colorbar

%% compute maps
filid=1;
% figure;
L2  = L*2;

filt_fftpsi = zeros(J,L2,N,N);


for j=1:J
    for q = 1:2*L
        
        fil=filnew.psi.filter{filid};
        
        filt_fftpsi(j,q,:,:) = fil.coefft{1};
        
        filid=filid+1;
        
        
    end
end
assert(length(filnew.psi.filter)==filid-1);

filt_fftphi = filnew.phi.filter.coefft{1};

path = './filters/';
filename = sprintf('bumpsteerableg%d_fft2d_N%d_J%d_L%d.mat',filter_id,N,J,L);

if exist(sprintf('%s/%s',path,filename)) > 0
    error(sprintf('file %s existed, can not export',filename))
end
save(sprintf('%s/%s',path,filename),  'filt_fftpsi', 'filt_fftphi' )
disp(filename)
