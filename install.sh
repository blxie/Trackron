#!/bin/bash
conda deactivate
conda remove -n trackron --all

# install Python
echo "****************** Installing env ******************"
conda create -n trackron python=3.8
conda activate trackron
conda info -e


# install pytorch
echo ""
echo ""
echo "****************** Installing pytorch ******************"
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch


# install site-pkgs
echo ""
echo ""
echo "****************** Installing cython-bbox ******************"
echo "****************** Installing requirements.txt ******************"
pip install cython
pip install cython-bbox
pip install -r requirements.txt


echo ""
echo ""
echo "****************** Installing tao ******************"
# git config --system http.sslCAinfo /etc/ssl/certs/ca-certificates.crt
pip install git+https://github.com/TAO-Dataset/tao
# pip install git+http://github.com/TAO-Dataset/tao
