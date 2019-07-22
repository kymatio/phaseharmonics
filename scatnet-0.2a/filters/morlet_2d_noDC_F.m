% MORLET_2D_NODC_F computes the 2-D elliptic Morlet filter given a set of 
%    parameters in Fourier domain
%
% Usage
%    gab = MORLET_2D_NODC_F(N, M, sigma, slant, xi, theta, offset)
%
% Input
%    N (numeric): Width of the filter.
%    M (numeric): Height of the filter.
%    sigma (numeric): Standard deviation of the envelope.
%    slant (numeric): Eccentricity of the elliptic envelope.
%       (the smaller slant, the larger angular resolution).
%    xi (numeric): The frequency peak.
%    theta (numeric): Orientation in radians of the filter.
%    offset (numeric, optional): 2-D vector reprensting the offset location 
%       (default [0 0]).
% 
% Output
%    gab (numeric): N-by-M matrix representing the gabor filter in Fourier
%       domain.
%
% Description
%    Compute a Morlet wavelet in Fourier domain. 
%
%    Morlet wavelets have a 0 DC component.
%
% See also
%    GABOR_2D, MORLET_2D_NODC

function gab = morlet_2d_noDC_F(N, M, sigma, slant, xi, theta, offset)
	if ~exist('offset','var')
		offset = [0, 0];
	end
	[x , y] = meshgrid(1:M, 1:N);
	
	x = x - 1 - offset(1); x = (x/M)*(2*pi);
	y = y - 1 - offset(2); y = (y/N)*(2*pi);
	
	Rth = rotation_matrix_2d(theta);
	A = Rth\ [1/sigma^2, 0 ; 0 slant^2/sigma^2] * Rth ;
    invA = inv(A);
    
    gaussian_envelope = gabor2(x,y,invA,0,0);
	oscilating_part = gabor2(x,y,invA,xi,theta);
    K = oscilating_part(1,1)/gaussian_envelope(1,1);
	gab = oscilating_part - K.*gaussian_envelope;
end

function g=gabor2(x0,y0,A,xi,theta)
    extent = 4; % extent of periodization - the higher, the better
    g=zeros(size(x0));
    for k1 = -extent:1+extent
        for k2 = -extent:1+extent
            x=x0-k1*2*pi;y=y0-k2*2*pi;
            x = x - xi*cos(theta);
            y = y - xi*sin(theta);
            s = x.* ( A(1,1)*x + A(1,2)*y) + y.*(A(2,1)*x + A(2,2)*y ) ;
            gaussian_envelope = exp( - s/2);
            g=g+gaussian_envelope;
        end
    end
end
