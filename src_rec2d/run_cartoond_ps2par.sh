CMD=cartoond/test_rec_bump_chunkid_lbfgs_gpu_N256_ps2par.py 
dj=5
nbchunk=16
dn=0
for i in 1 2 3 4 5 6 7 8 9 10 
do
   outpath=../results/cartoond/Run$i
   echo "Welcome $i times" $outpath
   python $CMD $outpath $dj $dn $nbchunk
done
