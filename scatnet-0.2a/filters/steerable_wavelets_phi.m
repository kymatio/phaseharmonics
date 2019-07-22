% return real isotropic filter in Fourier domain
function hatfilter = steerable_wavelets_phi(size_filter, J)
    d = length(size_filter);
    assert(d==2);
    
	N = size_filter(1);
	M = size_filter(2);
    [x0, y0] = meshgrid(1:N,1:M);
    assert(M==N);
    % hatfilter = zeros(N,M);
    
    x = x0 - ceil(N/2) - 1;
    y = y0 - ceil(M/2) - 1;

%     [y,x] = meshgrid(1:M,1:N);
%     x=x-(N/2+1);
%     y=y-(M/2+1);
%    modx=fftshift(sqrt(x.^2 + y.^2));

    om1 = x/N*2*pi;
    om2 = y/M*2*pi;
    
    modx=fftshift(sqrt(om1.^2 + om2.^2));   
    hatfilter  = phihat(2^(J-1)*modx);

%     for k=min(modx(:)):max(modx(:))
%         mask = ((modx>=k)&(modx<(k+1)));
%         om = k*2*pi/N;
%         if k>=N/2
%             om=om-2*pi;
%         end
%         hatfilter(mask) = phihat(2^(J-1)*abs(om));
%     end
    %imagesc(fftshift(hatfilter))
end

function phat=phihat(om)
    phat = zeros(size(om));
    sel1 = (om<=pi/4);
    phat(sel1)=1;
    sel2 = (om>pi/4 & om<pi/2); 
    phat(sel2)= cos(pi/2*log2((4*om(sel2))/pi));
   
%     if om <= pi/4 && om >= 0
%         phat=1;
%     elseif om>pi/4 && om<pi/2
%         phat=cos(pi/2*log2((4*om)/pi));
%     else
%         phat=0;
%     end
end