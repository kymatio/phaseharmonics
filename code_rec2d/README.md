A simple demo + How to reproduce the Figure 8 in the paper Phase Harmonic Correlations and Convolutional Neural Networks.

### Create 2d Bump steerable wavelet filters
There are 3 major parameters to create 2d filters using the matlab script at ./matlab/export_filter_bumpsteerableg.m
In our expereiemnt, we have chosen N=256 (image size), J=8 (maximum scale), L=8 (number of angles).
Simply "cd ./matlab" and call "matlab -r export_filter_bumpsteerableg" should work.

### Run a demo on a smaller size image using only 1 GPU
First create the wavelet filters using N=64, J=6, L=8. Then run:
	python cartoond/test_rec_bump_chunkid_lbfgs_gpu_N64.py

If you have an Xorg diaplay, the matplotlib will show you the original image at the beginning, 
and the recontruction image at the end (which should look alike up to a global translation).

### Run reconstructions per delta_j for 10 times using 2 GPUs
Use script run_cartoond_ps2par.sh or run_boat_ps2par.sh

First, you need to switch to the right conda env, for convenience you may run ". env.sh"

Before you run the script, make sure that you have chosen the right parameter delta_j (dj) from 0 to J-1. You may modify that in the script.Use 2 GPUs can save computational time and distribute memory usage. You should increase the number of chunks (nbchunk) to reduce memory usage on GPUs.

Run ./run_cartoond_ps2par.sh, by default it is with dj=1 (same for ./run_boat_ps2par.sh)

### Results
It may take a few hours / days to obtain all the results. For details of how to plot the results, please check the plot_psnr_nb_boat_Run.pd and plot_psnr_nb_cartoond_Run.pdf files. Compute the number of coeffs using compute_coeff_ps2.py; for sanity check, use compute_coeff.py
