% GABOR_2D computes the 2-D elliptic Gabor wavelet given a set of 
% parameters
%
% Usage
%    gab = GABOR_2D(N, M, sigma0, slant, xi, theta, offset, precision)
%
% Input
%    N (numeric): width of the filter
%    M (numeric): height of the filter
%    sigma0 (numeric): standard deviation of the envelope
%    slant (numeric): excentricity of the elliptic envelope
%            (the smaller slant, the larger angular resolution)
%    xi (numeric):  the frequency peak
%    theta (numeric): orientation in radians of the filter
%    offset (numeric): 2-D vector reprensting the offset location.
%    Optional
%    precision (string): precision of the computation. Optional
% 
% Output
%    gab(numeric) : N-by-M matrix representing the gabor filter in spatial
%    domain
%
% Description
%    Compute a Gabor wavelet. When used with xi = 0, and slant = 1, this 
%    implements a gaussian
%
%    Gabor wavelets have a non-negligeable DC component which is
%    removed for scattering computation. To avoid this problem, one can use
%    MORLET_2D_NODC.
%
% See also
%    MORLET_2D_NODC, MORLET_2D_PYRAMID

function gab = gabor_2d_period(N, M, sigma0, slant, xi, theta, offset, precision, extent)
	
	if ~exist('offset','var')
		offset = [0,0];
	end
	if ~exist('precision', 'var')
		precision = 'double';
	end
	
	[x0 , y0] = meshgrid(1:M,1:N);
	gabc = zeros(size(x0));

	for ex=-extent:extent
	for ey=-extent:extent

	x = x0 - ceil(M/2) - 1 + ex*M;
	y = y0 - ceil(N/2) - 1 + ey*N;
	x = x - offset(1);
	y = y - offset(2);
	Rth = rotation_matrix_2d(theta);
	A = Rth \ [1/sigma0^2, 0 ; 0 slant^2/sigma0^2] * Rth ;
	s = x.* ( A(1,1)*x + A(1,2)*y) + y.*(A(2,1)*x + A(2,2)*y ) ;
	% Normalization
	gabc = gabc + exp( - s/2 + 1i*(x*xi*cos(theta) + y*xi*sin(theta)));

	end
	end

	gab = 1/(2*pi*sigma0*sigma0/slant)*fftshift(gabc);
	
	if (strcmp(precision, 'single'))
		gab = single(gab);
	end
	
end
