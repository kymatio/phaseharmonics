% Compute a bank of Steerable wavelet filters in the Fourier domain.
% L is the number of angles along [0,pi)

function [filters,littlewood_final] = bumpsteerableg_wavelet_filter_bank_2d(size_in, options)

    if(nargin<2)
		options = struct;
    end
    
%     white_list = {'L', 'J', 'min_margin', 'precision', 'littlewood', 'filter_format'};
%     check_options_white_list(options, white_list);
    
    % Options
    options = fill_struct(options, 'L',4);
    options = fill_struct(options, 'J',4);
    options = fill_struct(options, 'full2pi', 0);
    options = fill_struct(options, 'fcenter', 0.425); % default central freq (unit=2*pi)
    options = fill_struct(options, 'gamma1', 1); 
    options = fill_struct(options, 'xi2sigma', 0.702*2^(-0.55)*sqrt(2)); 
    options = fill_struct(options, 'precision', 'double');
    options = fill_struct(options, 'littlewood', 1);
    options = fill_struct(options, 'gamma1',1);
    
    L = options.L;
    J = options.J;
    xi0 = options.fcenter*2*pi;
    gamma1 = options.gamma1;
    xi2sigma = options.xi2sigma;
    sigma_phiJ = 1/(xi2sigma*xi0*2^(-J+1));
    
    switch options.precision
        case 'single'
            cast = @single;
        case 'double'
            cast = @double;
        otherwise
            error('precision must be either double or single');
    end
    
    % Size
    size_filter = size_in;
	phi.filter.type = 'fourier_multires';
	
	N = size_filter(1);
	M = size_filter(2);

    % Compute low-pass filters phi
%     scale = 2^(J-1);
    extent=1;
    hatphiJ = real(fft2(gabor_2d_period(N, M, sigma_phiJ, 1, 0, 0, [0,0], 'double', extent)));
	phi.filter = cast(hatphiJ);
	phi.meta.J = J;
    phi.filter = optimize_filter(phi.filter, 1, options);
    
    littlewood_final = hatphiJ.^2;
    
    % Compute band-pass filters psi
    littlewood_psi = zeros(size(littlewood_final));
    
    p = 1;
    if options.full2pi == 1
        L2=2*L;
    else
        L2=L;
    end
	for j = 1:J
        for q = 1:L2
            % Morlet along radius
            hatpsi_jq =  wavelets_psi(size_filter, j, q, L, xi0, gamma1);
            littlewood_psi = littlewood_psi + hatpsi_jq.^2;
            psi.filter{p} = cast(hatpsi_jq);
			psi.meta.j(p) = j;
			psi.meta.theta(p) = q-1;
			p = p + 1;
        end
    end
    %imagesc(fftshift(littlewood_psi))
    littlewood_psi0 = zeros(size(littlewood_final));
	for p = 1:numel(psi.filter)
		psi.filter{p} = optimize_filter(psi.filter{p}, 0, options);
        if psi.meta.j(p)==1
            littlewood_psi0 = littlewood_psi0 + psi.filter{p}.coefft{1}.^2;
        end
        littlewood_final = littlewood_final + psi.filter{p}.coefft{1}.^2;
    end
    
	filters.phi = phi;
	filters.psi = psi;
    
	filters.meta.J = J;
	filters.meta.L = L;
	filters.meta.size_in = size_in;
	filters.meta.size_filter = size_filter;
end

function hatfilter = wavelets_psi(size_filter, j, q, L, xi, gamma1)
    d = length(size_filter);
    assert(d==2);
    assert(j>0 && q>0)
    % use 2pi peridizaton in Fourier domain, aliasing near zero!?
    % xi=1/2 correspondes to the central freq pi?
    % how about the Lttlewood-Palay condition
	N = size_filter(1);
	M = size_filter(2);
    [x0, y0] = meshgrid(1:N,1:M);
    assert(M==N);
    
    Cr = 1.29; % normalization constant along radius
    Ca = sqrt(L*factorial(2*L-2)) / (factorial(L-1)*(2^(L-1))); % normalization constant along angle
    hatfilter = zeros(N,M);
    for etx = -1:1
         for ety = -1:1
            % with anti-alising filter, no need for per.
            x = x0 - ceil(N/2) - 1+ etx*N;
            y = y0 - ceil(M/2) - 1+ ety*M;
            om1 = x/N*2*pi;
            om2 = y/M*2*pi;
            modx=fftshift(sqrt(om1.^2 + om2.^2));
            angles=fftshift(atan2(om2,om1));
            hatfilter0  = hwin(modx*2^(j-1)./xi-1,gamma1)/Cr; % hwin((modx-fcenter_j)/fradius_j,gamma1);
            theta_q = 2*pi*(q-1)/(2*L); % M=L-1
            hatfilter0 = hatfilter0 .* angularwin(angles,theta_q,L)/Ca;
            hatfilter = hatfilter + hatfilter0;
         end
	 end
end

function va=angularwin(angles,theta_q,L)
    mask = abs(angles-theta_q)<pi/2 | abs(angles-theta_q+2*pi)<pi/2;
    va = zeros(size(angles));
    va(mask) = cos(angles(mask)-theta_q).^(L-1); % order N=L-1
end

% win size 2*gamma1
function f=hwin(om,gamma1)
    f=zeros(size(om));
    sel = abs(om)<gamma1;
    f(sel)=exp( 1./gamma1^2 - 1./(gamma1^2-om(sel).^2) );
end