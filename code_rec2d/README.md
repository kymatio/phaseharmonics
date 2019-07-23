PhaseHarmonics: Wavelet phase harmonic transform in PyTorch
======================================

To reproduce Figure 8. in the paper Phase Harmonic Correlations and Convolutional Neural Networks, you need two GPUs to run the code in this folder. 

### Create Bump steerable wavelet filters 
There are 3 major parameters to create filters using the matlab script at ./matlab/export_filter_bumpsteerableg.m
In our expereiemnt, we have chosen N=256 (image size), J=8 (maximum scale), L=8 (number of angles).
Simply "cd ./matlab" and call "matlab -r export_filter_bumpsteerableg" should work.

### Run reconstructions per delta_j for 10 times
Use script run_cartoond_ps2par.sh or run_boat_ps2par.sh

First, you need to switch to the right conda env, for convenience you may use 
. env.sh

Before you run the script, make sure that you have chosen the right parameter delta_j (dj) from 0 to J-1. You may modify that in the script.
Use 2 GPUs can save compuational time, otherwise you should increase the number of chunks (nbchunk) to reduce memory usage on a GPU.

Run ./run_cartoond_ps2par.sh, by default it is with dj=1 (same for ./run_boat_ps2par.sh)

### Outputs
If it works, you should see the similar logs as below, for details of how to plot the results, please check the plot_psnr_nb_boat_Run.pd and plot_psnr_nb_cartoond_Run.pdf files.

Welcome 1 times ./results/boat/Run1
use skcuda backend
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 0  size is  432  among  3456
this la range tensor(0) tensor(15)
shift1= 0 shift2= 0
call cuda with devid= 0
sum of minput tensor(0.1885, device='cuda:0')
sum of minput tensor(0.3771, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 1  size is  432  among  3456
this la range tensor(16) tensor(31)
shift1= 0 shift2= 0
call cuda with devid= 1
sum of minput tensor(0.4268, device='cuda:1')
sum of minput tensor(0.8537, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 2  size is  432  among  3456
this la range tensor(32) tensor(47)
shift1= 0 shift2= 0
call cuda with devid= 0
sum of minput tensor(0.7859, device='cuda:0')
sum of minput tensor(1.5718, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 3  size is  432  among  3456
this la range tensor(48) tensor(63)
shift1= 0 shift2= 0
call cuda with devid= 1
sum of minput tensor(1.2631, device='cuda:1')
sum of minput tensor(2.5261, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 4  size is  432  among  3456
this la range tensor(64) tensor(79)
shift1= 0 shift2= 0
call cuda with devid= 0
sum of minput tensor(1.8269, device='cuda:0')
sum of minput tensor(3.6538, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 5  size is  432  among  3456
this la range tensor(80) tensor(95)
shift1= 0 shift2= 0
call cuda with devid= 1
sum of minput tensor(3.1077, device='cuda:1')
sum of minput tensor(6.2154, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 6  size is  432  among  3456
this la range tensor(96) tensor(111)
shift1= 0 shift2= 0
call cuda with devid= 0
sum of minput tensor(5.1227, device='cuda:0')
sum of minput tensor(10.2453, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 7  size is  432  among  3456
this la range tensor(112) tensor(127)
shift1= 0 shift2= 0
call cuda with devid= 1
sum of minput tensor(4.7882, device='cuda:1')
sum of minput tensor(9.5763, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 0  size is  504  among  6048
this la range tensor(0) tensor(47)
call cuda with devid= 0
sum of minput tensor(0.7819, device='cuda:0')
sum of minput tensor(0.5689, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 1  size is  504  among  6048
this la range tensor(18) tensor(63)
call cuda with devid= 1
sum of minput tensor(1.9289, device='cuda:1')
sum of minput tensor(1.0763, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 2  size is  504  among  6048
this la range tensor(37) tensor(79)
call cuda with devid= 0
sum of minput tensor(3.3876, device='cuda:0')
sum of minput tensor(1.7546, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 3  size is  504  among  6048
this la range tensor(56) tensor(95)
call cuda with devid= 1
sum of minput tensor(5.5759, device='cuda:1')
sum of minput tensor(3.0655, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 4  size is  504  among  6048
this la range tensor(74) tensor(111)
call cuda with devid= 0
sum of minput tensor(9.7442, device='cuda:0')
sum of minput tensor(5.4216, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 5  size is  504  among  6048
this la range tensor(93) tensor(127)
call cuda with devid= 1
sum of minput tensor(16.7464, device='cuda:1')
sum of minput tensor(5.6469, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 6  size is  504  among  6048
this la range tensor(0) tensor(47)
call cuda with devid= 0
sum of minput tensor(-2.0097e-08, device='cuda:0')
sum of minput tensor(0.0063, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 7  size is  504  among  6048
this la range tensor(18) tensor(63)
call cuda with devid= 1
sum of minput tensor(-5.2049e-08, device='cuda:1')
sum of minput tensor(0.0105, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 8  size is  504  among  6048
this la range tensor(37) tensor(79)
call cuda with devid= 0
sum of minput tensor(-9.0911e-08, device='cuda:0')
sum of minput tensor(0.0112, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 9  size is  504  among  6048
this la range tensor(56) tensor(95)
call cuda with devid= 1
sum of minput tensor(-1.4772e-07, device='cuda:1')
sum of minput tensor(0.0576, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 10  size is  504  among  6048
this la range tensor(74) tensor(111)
call cuda with devid= 0
sum of minput tensor(-2.4496e-07, device='cuda:0')
sum of minput tensor(0.0567, device='cuda:0')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
this chunk 11  size is  504  among  6048
this la range tensor(93) tensor(127)
call cuda with devid= 1
sum of minput tensor(-4.6983e-07, device='cuda:1')
sum of minput tensor(0.0084, device='cuda:1')
filter loaded: ./matlab/filters/bumpsteerableg1_fft2d_N256_J8_L8.mat
call cuda with devid= 0
sum of minput tensor(0.1354, device='cuda:0')
total ops is 21
total cov is 9505
1 tensor(19314.0273, device='cuda:0', grad_fn=<AddBackward0>) using time (sec): 2.3269166946411133
11 tensor(62.8728, device='cuda:0', grad_fn=<AddBackward0>) using time (sec): 19.13817024230957
...

