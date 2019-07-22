%%%%%script to generate figure 4 from the paper 'Invariant Scattering Convolution Networks'
%%%%%feb 2012%%%%%%%
%%%%%%Joan Bruna and Stephane Mallat$$$$$$$$$


%%%%%This figure shows two signals with the same first-order
%%%%%scattering coefficients, but different second order. 
% clear all;

N=512;
foptions.J=2;
foptions.L=8;
soptions.M=1;
[Wop,filters]=wavelet_factory_2d([N N], foptions, soptions);

% compute littlewood palay
littlewood_final = realize_filter(filters.phi.filter).^2;
for k = 1:length(filters.psi.filter)
    littlewood_final = littlewood_final + ...
                    0.5*abs(realize_filter(filters.psi.filter{k})).^2;
end
max(littlewood_final(:))
min(littlewood_final(:))
figure
imagesc(fftshift(littlewood_final)); 
% colormap gray
axis square

if 1

raster=1;
%%%%generate a synthetic triangle
tmp=zeros(4*N);
ix=1:4*N;
iix=ones(4*N,1)*ix;
iiy=iix';
rho=1;
tmp=double(iix < rho*iiy);
apert=pi/50;
angles=angle(iix+i*iiy);
radius=sqrt(iix.^2+iiy.^2);

% tmp=double(angles < pi/4+apert).*(angles > pi/4-apert).*(radius < 4*N);
% tmp=circshift(tmp,[N/2 N/2]);
apert=pi/30;
tmp=double(angles < pi/4+apert).*(angles > pi/4-apert).*(radius < 0.2*4*N);
tmp=circshift(tmp,[round(1.6*N) round(1.6*N)]);

gg=fspecial('gaussian',[9 9],1);
tmp=imfilter(tmp,gg);
[gr1,gr2]=gradient(tmp);
tmp=sqrt(gr1.^2+gr2.^2);

gg=fspecial('gaussian',[9 9],4);
tmp=imfilter(tmp,gg);
test{raster}=tmp(1:4:end,1:4:end);

[psi,phi,lp]=legacy_reshape_filters(filters, size(test{raster}));

test{raster}=ifft2(fft2(test{raster}).*(sqrt(lp{1})));
test{raster}=test{raster}-mean(test{raster}(:));
test{raster}=test{raster}/norm(test{raster}(:));
tempo=randn(size(test{raster}));
tempo=tempo-mean(tempo(:));
tempo=tempo/norm(tempo(:));

imwrite(test{raster},'tri4.png','png')
%imwrite(test{raster}./max(test{raster}(:),'tri3.png','png')

%%%% equalize white noise such that its first-order scattering
%%%% coefficients match the prescribed ones from the first image
[geq,l1f,l1g,l1eq]=equalize_first_order_scattering(test{raster},tempo,psi,phi,lp);

%%%compute scattering transform 
[sc{raster}]=scat(test{raster},Wop);
fprintf('scatt done \n')

raster=raster+1;

test{raster}=geq;
%%%compute scattering transform 
[sc{raster}]=scat(test{raster},Wop);
fprintf('scatt done \n')

end


dirac=zeros(size(test{1}));
dirac(1)=1;
dirac=fftshift(dirac);
[scdirac]=scat(dirac,Wop);

%%%% construct scattering display
copts.renorm_process=0;
copts.l2_renorm=1;
[out{1}]=scat_display(sc{1},scdirac,copts);
%%%% construct scattering display
[out{2}]=scat_display(sc{2},scdirac,copts);

normvalues(1)=max(max(out{1}{1}(:)),max(out{2}{1}(:)));
normvalues(2)=max(max(out{1}{2}(:)),max(out{2}{2}(:)));

%display results
normvalueseff=1.15*normvalues;
test{1}=max(0,test{1});
show_two_spectra_dlux(test{1},out{1}, .5,normvalueseff);
show_two_spectra_dlux(test{2},out{2}, .5,normvalueseff);





