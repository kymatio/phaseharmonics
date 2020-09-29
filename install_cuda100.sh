echo "include conda in your path"

conda create --name phaseharmonics100 python=3.6
source activate phaseharmonics100

conda install scipy
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

CU=/usr/local/cuda-10.0/
export CUDA_INC_DIR=$CU/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CU/lib64
export PATH=$PATH:$CU/bin/
pip install scikit-cuda
pip install cupy-cuda100

python setup.py install
