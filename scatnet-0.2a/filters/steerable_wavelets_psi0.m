% return real oriented filter in Fourier domain
function hatfilter = steerable_wavelets_psi0(size_filter)
    d = length(size_filter);
    assert(d==2);
%     extent = 0;
    
	N = size_filter(1);
	M = size_filter(2);
    [x0, y0] = meshgrid(1:N,1:M);
    assert(M==N);
    %hatfilter = zeros(N,M);
%     for ox=extent
%         for oy=extent
    x=x0-(ceil(N/2)+1); % -ox*N;
    y=y0-(ceil(M/2)+1); % -oy*M;
    %modx=fftshift(sqrt(x.^2 + y.^2));
    om1 = x/N*2*pi;
    om2 = y/M*2*pi;
    
    modx=fftshift(sqrt(om1.^2 + om2.^2));
    hatfilter = psihat(modx/2);
%     for k=min(modx(:)):max(modx(:))
%         mask = ((modx>=k)&(modx<(k+1)));
%         om = k*2*pi/N;
%         hatfilter(mask) = psihat(0.5*om); % hatfilter(mask) + psihat(0.5*om);
%     end
%         end
%     end
%     imagesc(fftshift(hatfilter))
end

function phat=psihat(om)
    phat = zeros(size(om));
    sel1 = om>pi/4 & om <= pi;
    phat(sel1) = cos(pi/2*log2((2*om(sel1))/pi));

    %if om>pi/4 && om<=pi
    %    phat=cos(pi/2*log2((2*om)/pi));
    %else
    %    phat=0;
    %end
end
