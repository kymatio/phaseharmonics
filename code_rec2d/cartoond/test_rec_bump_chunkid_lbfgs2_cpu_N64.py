# TEST pytorch LBFGS on GPU

from time import time
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.optim as optim

#---- create image ----#
size=64
data = sio.loadmat('./data/demo_toy7d_N' + str(size) + '.mat')
im = data['imgs']

#plt.imshow(im)
#plt.show()

im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)

# Parameters
J = 6
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L/2
delta_k = 1
nb_chunks = 10

# Model
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_simplephase \
    import PhaseHarmonics2d

Sims = []
factr = 1e3
wph_ops = dict()
for chunk_id in range(nb_chunks+1):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    #wph_op = wph_op.cuda()
    wph_ops[chunk_id] = wph_op
    Sim_ = wph_op(im)*factr # (nb,nc,nb_channels,1,1,2)
    Sims.append(Sim_)

# Objective
def obj_fun(x,chunk_id):
    global wph_ops
    wph_op = wph_ops[chunk_id]
    p = wph_op(x)*factr
    diff = p-Sims[chunk_id]
    loss = torch.mul(diff,diff).mean()
    return loss

def obj_func(x):
    loss = 0
    if x.grad is not None:
        x.grad.data.zero_()
    for chunk_id in range(nb_chunks+1):
        loss_t = obj_fun(x,chunk_id)
        loss_t.backward() # accumulate grad into x.grad
        loss = loss + loss_t
    return loss

# Timer
time0 = time()

# Init
torch.manual_seed(999)
x = torch.Tensor(1, 1, N, N).normal_(std=0.01)+0.5
x = x.requires_grad_(True)

# Optimize
optimizer = optim.LBFGS({x}, max_iter=100, line_search_fn='strong_wolfe', tolerance_grad = 1e-14, tolerance_change = 1e-14, history_size = 100)

def closure():
    optimizer.zero_grad()
    loss = obj_func(x)
    global time0
    count = optimizer.state[optimizer._params[0]]['n_iter']
    if count%10 == 0:
        print('n_iter:', count, 'loss:', loss.item(), 'using time (sec):' , time()-time0)
        time0 = time()
    return loss

optimizer.step(closure)

# output
#im_opt = np.reshape(x.detach().cpu().numpy(), (size,size))
#plt.imshow(im_opt)
#plt.show()
