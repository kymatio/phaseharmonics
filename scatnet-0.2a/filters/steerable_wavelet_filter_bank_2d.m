% Compute a bank of Steerable wavelet filters in the Fourier domain.
% L is the number of angles along [0,pi)

function [filters,littlewood_final] = steerable_wavelet_filter_bank_2d(size_in, options)

    if(nargin<2)
		options = struct;
    end
    
%     white_list = {'L', 'J', 'min_margin', 'precision', 'littlewood', 'filter_format'};
%     check_options_white_list(options, white_list);
    
    % Options
    options = fill_struct(options, 'L',4);
    options = fill_struct(options, 'J',3);
    options = fill_struct(options, 'full2pi', 0);
    L = options.L;
    J = options.J;
    options = fill_struct(options, 'precision', 'double');
    options = fill_struct(options, 'littlewood', 1);

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
	
% 	N = size_filter(1);
% 	M = size_filter(2);
%     extent = options.extent;
%     
%     % compute first the |w| function given N and J
%     [hatphi_J,hatpsi_j,hatpsi_0,littlewood]=steerable_radial_filters(N,J);
    
	% Compute low-pass filters phi
	hatphiJ = steerable_wavelets_phi(size_filter, J);
	phi.filter = cast(hatphiJ);
	phi.meta.J = J;
    phi.filter = optimize_filter(phi.filter, 1, options);

    littlewood_final = hatphiJ.^2;
    
    % Compute band-pass filters psi
    littlewood_psi = zeros(size(hatphiJ));
    p = 1;
    if options.full2pi == 1
        L2=2*L;
    else
        L2=L;
    end
	for j = 1:J
        for q = 1:L2
            hatpsi_jq = steerable_wavelets_psi(size_filter, j, q, L);
            littlewood_psi = littlewood_psi + hatpsi_jq.^2;
            psi.filter{p} = cast(hatpsi_jq);
			psi.meta.j(p) = j;
			psi.meta.theta(p) = q;
			p = p + 1;
        end
    end
%     imagesc(fftshift(littlewood_psi))

    % compute the 'Haar' filter
    hatpsi0 = steerable_wavelets_psi0(size_filter);
%     figure; imagesc(fftshift(hatpsi0))
    littlewood_psi = littlewood_psi + hatpsi0.^2;
%     figure; imagesc(fftshift(littlewood_psi))
    
    K = max(littlewood_psi(:));
	for p = 1:numel(psi.filter)
		psi.filter{p} = psi.filter{p}/sqrt(K);
		psi.filter{p} = optimize_filter(psi.filter{p}, 0, options);
        littlewood_final = littlewood_final + psi.filter{p}.coefft{1}.^2;
    end
% 	hatpsi0 = hatpsi0/sqrt(K);
    
    littlewood_final = littlewood_final + hatpsi0.^2;
    %imagesc(fftshift(littlewood_final));
    
	filters.phi = phi;
	filters.psi = psi;
    filters.psi0 = hatpsi0;

	filters.meta.J = J;
	filters.meta.L = L;
	filters.meta.size_in = size_in;
	filters.meta.size_filter = size_filter;
end