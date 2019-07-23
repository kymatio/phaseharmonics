PhaseHarmonics: Wavelet phase harmonic transform in PyTorch
======================================

This is an implementation of the wavelet phase harmonic transform based on Kymatio (in the Python programming language). It is suitable for audio and image analysis and modeling.

### Publication
* [1] Stéphane Mallat, Sixin Zhang, Gaspar Rochette. (2019) Phase Harmonic Correlations and Convolutional Neural Networks. [(paper)](https://arxiv.org/abs/1810.12136).

### Installation
For general installation, please follow the instructions at [(kymatio)](https://github.com/kymatio/kymatio). You may also use the script ./install_cuda92.sh to setup GPU-supported anaconda env.

### Reproducing 2d reconstructions in paper [1]. 
The code is tested on Ubuntu 16 + two TITAN Xp GPU + cuda 9.2 + Nvidia Driver Version: 410.66. Please follow the README in the folder code_rec2d. Matlab is needed to generate 2d wavelet filters.
