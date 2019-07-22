% return analytic oriented filter in Fourier domain
function hatfilter = steerable_wavelets_psi(size_filter, j, q, L)
    d = length(size_filter);
    assert(d==2);
    assert(j>0 && q>0)

	N = size_filter(1);
	M = size_filter(2);
    [x0, y0] = meshgrid(1:N,1:M);
    assert(M==N);
    % hatfilter = zeros(N,M);
    
    x = x0 - ceil(N/2) - 1;
    y = y0 - ceil(M/2) - 1;

    om1 = x/N*2*pi;
    om2 = y/M*2*pi;
    
    modx=fftshift(sqrt(om1.^2 + om2.^2));
    angles=fftshift(atan2(om2,om1));
    
    hatfilter  = psihat(2^(j-1)*modx);
%     K=0:min(N,M);
%     for k=min(modx(:)):max(modx(:))
%         mask = ((modx>=k)&(modx<(k+1)));
%         om = k*2*pi/N;
%         if k>=N/2
%             om=om-2*pi; % [-pi,0)
%         end
%         hatfilter(mask) = psihat(2^(j-1)*om);
%     end
    theta_q = 2*pi*(q-1)/(2*L); % M=L-1
    hatfilter = hatfilter .* angularwin(angles,theta_q,L);
    %imagesc(fftshift(hatfilter))
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

function va=angularwin(angles,theta_q,L)
    mask = abs(angles-theta_q)<pi/2 | abs(angles-theta_q+2*pi)<pi/2;
    va = zeros(size(angles));
    va(mask) = cos(angles(mask)-theta_q).^(L-1); % order N=L-1
end