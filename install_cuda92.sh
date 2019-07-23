export PATH="$PATH:~/anaconda3/bin"

conda create --name phaseharmonics python=3.6
source activate phaseharmonics

conda install scipy
conda install pytorch=1.0.0 cuda92 -c pytorch

export CUDA_INC_DIR=/usr/local/cuda-9.2/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64
export PATH=$PATH:/usr/local/cuda-9.2/bin/
pip install scikit-cuda
pip install cupy-cuda92

python setup.py install
