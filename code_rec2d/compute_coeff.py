# TEST ON GPU
import os,sys
#FOLOUT = sys.argv[1] # store the result in output folder
#os.system('mkdir -p ' + FOLOUT)

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

from time import time
import gc

#---- create image without/with marks----#

size=256

# Parameters for transforms
J = 8
L = 8
M = size
N = size # im.shape[-2], im.shape[-1]
delta_j = int(sys.argv[1])
delta_l = L/2
delta_k = 1
nb_chunks = 1

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_simplephase \
    import PhaseHarmonics2d

chunk_id = 0
wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
nb1,nb2 = wph_op.compute_ncoeff()
print(nb1,nb2,nb1+nb2)

