# TEST ON GPU
import os,sys
FOLOUT = sys.argv[1] # store the result in output folder
os.system('mkdir -p ' + FOLOUT)

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import grad # Variable, grad

from time import time

#---- create image without/with marks----#

size=256

# --- Dirac example---#
data = sio.loadmat('./data/demo_boat1_N' + str(size) + '.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms
J = 8
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = int(sys.argv[2])
delta_l = L/2
delta_k = 1
delta_n = int(sys.argv[3])
nb_chunks = int(sys.argv[4])
nGPU = 2 # int(sys.argv[5])
nb_restarts = 10 # 20
factr = 1e5 # 6
factr2 = factr**2

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_scaleinter \
    import PhkScaleInter2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_pershift \
    import PHkPerShift2d

wph_streams = []
for devid in range(nGPU):
    with torch.cuda.device(devid):
        s = torch.cuda.Stream() 
        wph_streams.append(s)

Sims = []
wph_ops = []
opid = 0
nCov = 0
for dn1 in range(-delta_n,delta_n+1):
    for dn2 in range(-delta_n,delta_n+1):
        if dn1**2+dn2**2 <= delta_n**2:
            for chunk_id in range(J):
                devid = opid % nGPU
                if dn1==0 and dn2==0:
                    wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, delta_l, J, chunk_id, devid)
                else:
                    wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, 0, J, chunk_id, devid) 
                wph_op = wph_op.cuda()
                wph_ops.append(wph_op)
                assert(wph_ops[opid]==wph_op)
                opid += 1
                im_ = im.to(devid)
                with torch.cuda.device(devid):
                    torch.cuda.stream(wph_streams[devid])
                    Sim_ = wph_op(im_) # *factr # (nb,nc,nb_channels,1,1,2)
                    nCov += Sim_.shape[2]
                    Sims.append(Sim_)

torch.cuda.synchronize()

for chunk_id in range(nb_chunks+1):
    devid = opid % nGPU
    wph_op = PhkScaleInter2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, devid)
    wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    assert(wph_ops[opid]==wph_op)
    opid += 1
    im_ = im.to(devid)
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
        Sim_ = wph_op(im_)  # *factr # (nb,nc,nb_channels,1,1,2)
        nCov += Sim_.shape[2]
        Sims.append(Sim_)

torch.cuda.synchronize()

print('total ops is', len(wph_ops))
print('total cov is', nCov)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
#---- Trying scipy L-BFGS ----#

def obj_fun(x,opid):
    #if x.grad is not None:
    #    x.grad.data.zero_()
    global wph_ops
    wph_op = wph_ops[opid]
    p = wph_op(x) # *factr
    diff = p-Sims[opid]
    loss = factr2*torch.mul(diff,diff).sum()/nCov
    return loss

grad_err = im.to(0)

def grad_obj_fun(x_gpus): # x_cpu):
    loss_a = []
    grad_err_a = []
    #x_a = []
    #for opid in range(len(wph_ops)):
        #devid = opid % nGPU
        #x_t = x_cpu.to(devid).requires_grad_(True)
    #    x_a.append(x_t)

    for opid in range(len(wph_ops)):
        devid = opid % nGPU
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            x_t = x_gpus[devid] # x_a[opid]
            loss_ = obj_fun(x_t,opid)
            loss_a.append(loss_)
            grad_err_, = grad([loss_],[x_t], retain_graph=False)
            grad_err_a.append(grad_err_)

    torch.cuda.synchronize()
    
    loss = 0
    global grad_err
    grad_err[:] = 0
    for opid in range(len(wph_ops)):
        grad_err = grad_err + grad_err_a[opid].to(0)
        loss = loss + loss_a[opid].to(0)
        
    return loss, grad_err

count = 0
from time import time
time0 = time()
def fun_and_grad_conv(x):
    x_float = torch.reshape(torch.tensor(x,dtype=torch.float),(1,1,size,size))
    x_gpus = []
    for devid in range(nGPU):
        x_gpus.append(x_float.to(devid)) # 
        x_gpus[devid].requires_grad_(True)
    
    #x_gpu = x_float.cuda()
    loss, grad_err = grad_obj_fun(x_gpus)
    global count
    global time0
    count += 1
    if count%10 == 1:
        print(count, loss, 'using time (sec):' , time()-time0)
        time0 = time()
    return loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)

def callback_print(x):
    return

#seed = 2018
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

x = torch.Tensor(1, 1, N, N).normal_(std=0.01)+0.5
x0 = x.reshape(size**2).numpy()
x0 = np.asarray(x0, dtype=np.float64)

for start in range(nb_restarts):
    if start==0:
        x_opt = x0
    res = opt.minimize(fun_and_grad_conv, x_opt, method='L-BFGS-B', jac=True, tol=None,
                       callback=callback_print,
                       options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 20})
    final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
    print('OPT fini avec:', final_loss,niter,msg)

    im_opt = np.reshape(x_opt, (size,size))
    tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    ret = dict()
    ret['tensor_opt'] = tensor_opt
    ret['normalized_loss'] = final_loss/(factr2) # **2)
    torch.save(ret, FOLOUT + '/' + 'test_rec_bump_chunkid_lbfgs_gpu_N256_ps2par' + '_dn' + str(delta_n) + '_dj' + str(delta_j) + '.pt')
