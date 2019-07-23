CMD=boat/test_rec_bump_chunkid_lbfgs_gpu_N256_ps2par.py 
dj=1
nbchunk=12
dn=0
for i in 1 2 3 4 5 6 7 8 9 10 
do
   outpath=./results/boat/Run$i
   echo "Welcome $i times" $outpath
   python $CMD $outpath $dj $dn $nbchunk
done
