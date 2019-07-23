To reproduce the Figure 8 in the paper Phase Harmonic Correlations and Convolutional Neural Networks, you need two GPUs to run the code in this folder. 

### Create 2d Bump steerable wavelet filters
There are 3 major parameters to create 2d filters using the matlab script at ./matlab/export_filter_bumpsteerableg.m
In our expereiemnt, we have chosen N=256 (image size), J=8 (maximum scale), L=8 (number of angles).
Simply "cd ./matlab" and call "matlab -r export_filter_bumpsteerableg" should work.

### Run reconstructions per delta_j for 10 times
Use script run_cartoond_ps2par.sh or run_boat_ps2par.sh

First, you need to switch to the right conda env, for convenience you may use 
. env.sh

Before you run the script, make sure that you have chosen the right parameter delta_j (dj) from 0 to J-1. You may modify that in the script.
Use 2 GPUs can save compuational time, otherwise you should increase the number of chunks (nbchunk) to reduce memory usage on a GPU.

Run ./run_cartoond_ps2par.sh, by default it is with dj=1 (same for ./run_boat_ps2par.sh)

### Results
For details of how to plot the results, please check the plot_psnr_nb_boat_Run.pd and plot_psnr_nb_cartoond_Run.pdf files.

