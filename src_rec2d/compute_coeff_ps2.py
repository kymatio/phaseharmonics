# TEST ON GPU
import os,sys
import numpy as np
import torch

#---- create image without/with marks----#

size=256
# Parameters for transforms
J = 8
L = 8
M = size
N = size
delta_j = int(sys.argv[1])
delta_l = L/2
delta_k = 1
if delta_j > 0:
    nb_chunks = 1
else:
    nb_chunks = 0

# model
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_scaleinter \
    import PhkScaleInter2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_pershift \
    import PHkPerShift2d

chunk_id = 0
wph_op = PhkScaleInter2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id) # , filname='simoncelli')
nb1,nb2 = wph_op.compute_ncoeff()

dn1=0
dn2=0
wph_op_ = PHkPerShift2d(M, N, J, L, dn1, dn2, delta_l, J, chunk_id) # , devid, filname='simoncelli')
nb1_,nb2_ = wph_op_.compute_ncoeff()
print('model si coeff = ',nb1+nb2)
print('model ps coeff = ',nb1_+nb2_)
print('total model coeff = ', nb1+nb2+nb1_+nb2_)
