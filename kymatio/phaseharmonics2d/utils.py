import scipy.fftpack
import warnings

def compute_padding(M, N, J):
    """
         Precomputes the future padded size.

         Parameters
         ----------
         M, N : int
             input size

         Returns
         -------
         M, N : int
             padded size
    """
    M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J
    return M_padded, N_padded

def fft2(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return scipy.fftpack.fft2(x)

from .backend import fft

def ifft2_c2r(x):
    return fft(x,'C2R')/(x.size(-2)*x.size(-3))

def fft2_c2c(x):
    return fft(x,'C2C') # *(x.size(-2)*x.size(-3))

def ifft2_c2c(x):
    return fft(x,'C2C',inverse=True)/(x.size(-2)*x.size(-3))

def periodic_dis(i1, i2, per):
    if i2 > i1:
        return min(i2-i1, i1-i2+per)
    else:
        return min(i1-i2, i2-i1+per)

def periodic_signed_dis(i1, i2, per):
    if i2 < i1:
        return i2 - i1 + per
    else:
        return i2 - i1

