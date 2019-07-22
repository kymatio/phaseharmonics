% cd '~/Coding/scatlearn/src/scatnet-0.2'
addpath_scatnet;

x = imread('lena.ppm');
x = mean(x,3)./255;
max(x(:))
min(x(:))
%imagesc(x);
%colormap gray

filt_opt = struct();
filt_opt.J = 3;
filt_opt.min_margin = 0;

Wop = wavelet_factory_2d(size(x),filt_opt);
Sx = scat(x, Wop);
S_mat = format_scat(Sx);

j1 = 0;
j2 = 2;
theta1 = 1;
theta2 = 5;
p = find( Sx{3}.meta.j(1,:) == j1 & ...
    Sx{3}.meta.j(2,:) == j2 & ...
    Sx{3}.meta.theta(1,:) == theta1 & ...
    Sx{3}.meta.theta(2,:) == theta2 );
%imagesc(Sx{3}.signal{p});

image_scat(Sx)

% display_with_layer_order(Sx,1)
% 
% load('/users/data/sixzhang/scatlearn/src/scatwave/scatwave_pkg/test_lena.mat')
% size(sx)
% imagesc(sx(:,:,1))
% 
% load('/users/data/sixzhang/scatlearn/src/scatwave/scatwave_pkg/scatwave_filter_phi_1.mat')

