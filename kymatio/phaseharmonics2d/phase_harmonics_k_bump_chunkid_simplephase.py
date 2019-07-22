# implement basic phase harmonics
# based on John's code to check correctness
# Case: phase_harmonic_cor in representation_complex
#      do not create new Sout for each forward

# Init with SubInit

__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
#import torch.nn.functional as F
from .backend import cdgmm, Modulus, fft, \
    Pad, SubInitSpatialMeanC, PhaseHarmonics2, mulcu
from .filter_bank import filter_bank
from .utils import fft2_c2c, ifft2_c2c, periodic_dis

class PhaseHarmonics2d(object):
    def __init__(self, M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, devid=0):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.dj = delta_j # max scale interactions
        self.dl = delta_l # max angular interactions
        self.dk = delta_k #
        self.nb_chunks = nb_chunks # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        self.devid = devid # gpu id
        assert( self.chunk_id <= self.nb_chunks ) # chunk_id = 0..nb_chunks-1, are the wph cov
        if self.dl > self.L:
            raise (ValueError('delta_l must be <= L'))

        self.pre_pad = False # no padding
        self.cache = False # cache filter bank
        self.build()

    def build(self):
        check_for_nan = False # True
        #self.meta = None
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.phase_harmonics = PhaseHarmonics2.apply

        self.M_padded, self.N_padded = self.M, self.N

        self.filters_tensor()
        if self.chunk_id < self.nb_chunks:
            self.idx_wph = self.compute_idx()
            self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
            self.subinitmean1 = SubInitSpatialMeanC()
            self.subinitmean2 = SubInitSpatialMeanC()
        else:
            self.subinitmeanJ = SubInitSpatialMeanC()

    def filters_tensor(self):
        J = self.J
        L = self.L
        L2 = L*2

        assert(self.M == self.N)
        matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N' + str(self.N) + '_J' + str(self.J) + '_L' + str(self.L) + '.mat')

        fftphi = matfilters['filt_fftphi'].astype(np.complex_)
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

        self.hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
        self.hatphi = torch.FloatTensor(hatphi) # (M,N,2)

        #print('filter shapes')
        #print(self.hatpsi.shape)
        #print(self.hatphi.shape)

    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        #print('la1 shape',self.idx_wph['la1'].shape)

        nb_cov = len(self.idx_wph['la1'])
        print('nb cov is', nb_cov)
        max_chunk = nb_cov // nb_chunks
        nb_cov_chunk = np.zeros(nb_chunks,dtype=np.int32)
        for idxc in range(nb_chunks):
            if idxc < nb_chunks-1:
                nb_cov_chunk[idxc] = int(max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - max_chunk*(nb_chunks-1))
                assert(nb_cov_chunk[idxc] > 0)

        this_wph = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            if idxc == chunk_id:
                this_wph['la1'] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['la2'] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['k1'] = self.idx_wph['k1'][:,offset:offset+nb_cov_chunk[idxc],:,:]
                this_wph['k2'] = self.idx_wph['k2'][:,offset:offset+nb_cov_chunk[idxc],:,:]
            offset = offset + nb_cov_chunk[idxc]

        return this_wph

    def compute_ncoeff(self):
        # return number of mean (nb1) and cov (nb2) of all idx
        L = self.L
        L2 = L*2
        J = self.J
        dj = self.dj
        dl = self.dl
        dk = self.dk
        
        hit_nb1 = dict() # hash table
        hit_nb2 = dict() # value counts either real or complex numbers
        
        # j1=j2, k1=1, k2=0 or 1
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 1
                j2 = j1
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k2 = 0
                        hit_nb1[(j1,k1,ell1)]=0
                        hit_nb1[(j2,k2,ell2)]=1
                        hit_nb2[(j1,k1,ell1,j2,k2,ell2)] = 2
                        k2 = 1
                        hit_nb1[(j1,k1,ell1)]=0
                        hit_nb1[(j2,k2,ell2)]=0
                        hit_nb2[(j1,k1,ell1,j2,k2,ell2)] = 2
                        
        # k1 = 0
        # k2 = 0
        # j1 = j2
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 0
                hit_nb1[(j1,k1,ell1)]=1 # BUG!, move it down 4 lines
                j2 = j1
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k2 = 0
                        hit_nb1[(j2,k2,ell2)]=1
                        hit_nb2[(j1,k1,ell1,j2,k2,ell2)] = 1

        # k1 = 0
        # k2 = 0,1,2
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 0
                hit_nb1[(j1,k1,ell1)]=1
                for j2 in range(j1+1,min(j1+dj+1,J)):
                    for ell2 in range(L2):
                        if periodic_dis(ell1, ell2, L2) <= dl:
                            for k2 in range(3):
                                if k2==0:
                                    hit_nb1[(j2,k2,ell2)]=1
                                    hit_nb2[(j1,k1,ell1,j2,k2,ell2)]=1
                                elif k2==1:
                                    hit_nb1[(j2,k2,ell2)]=0
                                    hit_nb2[(j1,k1,ell1,j2,k2,ell2)]=2
                                else:
                                    hit_nb1[(j2,k2,ell2)]=2
                                    hit_nb2[(j1,k1,ell1,j2,k2,ell2)]=2

        # k1 = 1
        # k2 = 2^(j2-j1)±dk
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 1
                hit_nb1[(j1,k1,ell1)]=0
                for j2 in range(j1+1,min(j1+dj+1,J)):
                     for ell2 in range(L2):
                         if periodic_dis(ell1, ell2, L2) <= dl:
                             for k2 in range(max(0,2**(j2-j1)-dk),2**(j2-j1)+dk+1):
                                if k2==0:
                                    hit_nb1[(j2,k2,ell2)]=1
                                    hit_nb2[(j1,k1,ell1,j2,k2,ell2)]=1
                                elif k2==1:
                                    hit_nb1[(j2,k2,ell2)]=0
                                    hit_nb2[(j1,k1,ell1,j2,k2,ell2)]=2
                                else:
                                    hit_nb1[(j2,k2,ell2)]=2
                                    hit_nb2[(j1,k1,ell1,j2,k2,ell2)]=2

        #print('hit nb1 values',list(hit_nb1.values()))
        nb1 = np.array(list(hit_nb1.values()), dtype=int).sum() + 1 # plus last phiJ channel
        nb2 = np.array(list(hit_nb2.values()), dtype=int).sum() + 1 # plus last phiJ channel

        return nb1, nb2
    
    def compute_idx(self):
        L = self.L
        L2 = L*2
        J = self.J
        dj = self.dj
        dl = self.dl
        dk = self.dk

        idx_la1 = []
        idx_la2 = []
        idx_k1 = []
        idx_k2 = []

        # j1=j2, k1=1, k2=0 or 1
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 1
                j2 = j1
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k2 = 0
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)
                        k2 = 1
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)

        # k1 = 0
        # k2 = 0
        # j1 = j2
        for j1 in range(J):
            for ell1 in range(L2):
                k1=0
                j2 = j1
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k2 = 0
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)


        # k1 = 0
        # k2 = 0,1,2
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 0
                for j2 in range(j1+1,min(j1+dj+1,J)):
                    for ell2 in range(L2):
                        if periodic_dis(ell1, ell2, L2) <= dl:
                            for k2 in range(3):
                                idx_la1.append(L2*j1+ell1)
                                idx_la2.append(L2*j2+ell2)
                                idx_k1.append(k1)
                                idx_k2.append(k2)

        # k1 = 1
        # k2 = 2^(j2-j1)±dk
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 1
                for j2 in range(j1+1,min(j1+dj+1,J)):
                    for ell2 in range(L2):
                        if periodic_dis(ell1, ell2, L2) <= dl:
                            for k2 in range(max(0,2**(j2-j1)-dk),2**(j2-j1)+dk+1):
                                idx_la1.append(L2*j1+ell1)
                                idx_la2.append(L2*j2+ell2)
                                idx_k1.append(k1)
                                idx_k2.append(k2)

        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['k1'] = torch.tensor(idx_k1).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)
        idx_wph['k2'] = torch.tensor(idx_k2).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return idx_wph

    def _type(self, _type, devid=None):
        self.hatpsi = self.hatpsi.type(_type)
        self.hatphi = self.hatphi.type(_type)
        if devid is not None:
            self.hatpsi = self.hatpsi.to(devid)
            self.hatphi = self.hatphi.to(devid)
        #print('in _type',type(self.hatpsi))
        self.pad.padding_module.type(_type)
        return self

    def cuda(self):
        """
            Moves tensors to the GPU
        """
        devid = self.devid
        print('call cuda with devid=', devid)
        assert(devid>=0)
        if self.chunk_id < self.nb_chunks:
            self.this_wph['la1'] = self.this_wph['la1'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['la2'] = self.this_wph['la2'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['k1'] = self.this_wph['k1'].type(torch.cuda.FloatTensor).to(devid)
            self.this_wph['k2'] = self.this_wph['k2'].type(torch.cuda.FloatTensor).to(devid)

        return self._type(torch.cuda.FloatTensor, devid)

    def cpu(self):
        """
            Moves tensors to the CPU
        """
        print('call cpu')
        return self._type(torch.FloatTensor)

    def forward(self, input):
          
        J = self.J
        M = self.M
        N = self.N
        L2 = self.L*2
        dj = self.dj
        dl = self.dl
        pad = self.pad

        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        #print('nbchannels',nb_channels)
        if self.chunk_id < self.nb_chunks:
            nb = hatx_c.shape[0]
            nc = hatx_c.shape[1]
            hatpsi_la = self.hatpsi # (J,L2,M,N,2)
            assert(nb==1 and nc==1) # for submeanC
            nb_channels = self.this_wph['la1'].shape[0]
            Sout = input.new(nb, nc, nb_channels, \
                             1, 1, 2) # (nb,nc,nb_channels,1,1,2)
            for idxb in range(nb):
                for idxc in range(nc):
                    hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
                    #               print('hatpsi_la is cuda?',hatpsi_la.is_cuda)
                    #                print('hatx_bc is cuda?',hatx_bc.is_cuda)
                    hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc) # (J,L2,M,N,2)
                    #print( 'hatxpsi_bc shape', hatxpsi_bc.shape )
                    xpsi_bc = ifft2_c2c(hatxpsi_bc)
                    # reshape to (1,J*L,M,N,2)
                    xpsi_bc = xpsi_bc.view(1,J*L2,M,N,2)

                    # select la1, et la2, P_c = number of |la1| in this chunk
                    xpsi_bc_la1 = torch.index_select(xpsi_bc, 1, self.this_wph['la1']) # (1,P_c,M,N,2)
                    xpsi_bc_la2 = torch.index_select(xpsi_bc, 1, self.this_wph['la2']) # (1,P_c,M,N,2)
                    #print('xpsi la1 shape', xpsi_bc_la1.shape)
                    #print('xpsi la2 shape', xpsi_bc_la2.shape)
                    k1 = self.this_wph['k1']
                    k2 = self.this_wph['k2']
                    xpsi_bc_la1k1 = self.phase_harmonics(xpsi_bc_la1, k1) # (1,P_c,M,N,2)
                    xpsi_bc_la2k2 = self.phase_harmonics(xpsi_bc_la2, -k2) # (1,P_c,M,N,2)
                    # sub spatial mean along M and N
                    xpsi0_bc_la1k1 = self.subinitmean1(xpsi_bc_la1k1) # (1,P_c,M,N,2)
                    xpsi0_bc_la2k2 = self.subinitmean2(xpsi_bc_la2k2) # (1,P_c,M,N,2)
                    # compute mean spatial
                    corr_xpsi_bc = mulcu(xpsi0_bc_la1k1,xpsi0_bc_la2k2) # (1,P_c,M,N,2)
                    corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (1,P_c,1,1,2)
                    Sout[idxb,idxc,:,:,:,:] = corr_bc[0,:,:,:,:]

        else:
            # ADD 1 chennel for spatial phiJ
            # add l2 phiJ to last channel
            hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
            xpsi_c = ifft2_c2c(hatxphi_c)
            # submean from spatial M N
            xpsi0_c = self.subinitmeanJ(xpsi_c)
            xpsi0_mod = self.modulus(xpsi0_c) # (nb,nc,M,N,2)
            xpsi0_mod2 = mulcu(xpsi0_mod,xpsi0_mod) # (nb,nc,M,N,2)
            nb = hatx_c.shape[0]
            nc = hatx_c.shape[1]
            Sout = input.new(nb, nc, 1, \
                             1, 1, 2)
            Sout[:,:,0,:,:,:] = torch.mean(torch.mean(xpsi0_mod2,-2,True),-3,True)

        return Sout
        
    def compute_mean(self,input):
        J = self.J
        M = self.M
        N = self.N
        L2 = self.L*2
        dj = self.dj
        dl = self.dl
        pad = self.pad

        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        #print('nbchannels',nb_channels)
        if self.chunk_id < self.nb_chunks:
            nb = hatx_c.shape[0]
            nc = hatx_c.shape[1]
            hatpsi_la = self.hatpsi # (J,L2,M,N,2)
            assert(nb==1 and nc==1) # for submeanC
            nb_channels = self.this_wph['la1'].shape[0]
            Sout1 = input.new(nb, nc, nb_channels, \
                              1, 1, 2) # (nb,nc,nb_channels,1,1,2)
            Sout2 = input.new(nb, nc, nb_channels, \
                              1, 1, 2) # (nb,nc,nb_channels,1,1,2)
            for idxb in range(nb):
                for idxc in range(nc):
                    hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
                    # print('hatx_bc is cuda?',hatx_bc.is_cuda)
                    hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc) # (J,L2,M,N,2)
                    # print( 'hatxpsi_bc shape', hatxpsi_bc.shape )
                    xpsi_bc = ifft2_c2c(hatxpsi_bc)
                    # reshape to (1,J*L,M,N,2)
                    xpsi_bc = xpsi_bc.view(1,J*L2,M,N,2)

                    # select la1, et la2, P_c = number of |la1| in this chunk
                    xpsi_bc_la1 = torch.index_select(xpsi_bc, 1, self.this_wph['la1']) # (1,P_c,M,N,2)
                    xpsi_bc_la2 = torch.index_select(xpsi_bc, 1, self.this_wph['la2']) # (1,P_c,M,N,2)
                    #print('xpsi la1 shape', xpsi_bc_la1.shape)
                    #print('xpsi la2 shape', xpsi_bc_la2.shape)
                    k1 = self.this_wph['k1']
                    k2 = self.this_wph['k2']
                    xpsi_bc_la1k1 = self.phase_harmonics(xpsi_bc_la1, k1) # (1,P_c,M,N,2)
                    xpsi_bc_la2k2 = self.phase_harmonics(xpsi_bc_la2, -k2) # (1,P_c,M,N,2)
                    mean1_bc = torch.mean(torch.mean(xpsi_bc_la1k1,-2,True),-3,True) # (1,P_c,1,1,2)
                    mean2_bc = torch.mean(torch.mean(xpsi_bc_la2k2,-2,True),-3,True) # (1,P_c,1,1,2)
                    Sout1[idxb,idxc,:,:,:,:] = mean1_bc[0,:,:,:,:]
                    Sout2[idxb,idxc,:,:,:,:] = mean2_bc[0,:,:,:,:]
            Sout = torch.stack((Sout1,Sout2),dim=0)
        else:
            hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
            xpsi_c = ifft2_c2c(hatxphi_c) # (nb,nc,M,N,2)
            Sout = torch.mean(torch.mean(xpsi_c,-2,True),-3,True) # (nb,nc,1,1,2)

        return Sout

    def __call__(self, input):
        return self.forward(input)
