#!/bin/bash

# install pytorch
echo "****************** Installing pytorch ******************"
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch

# install site-pkgs
echo ""
echo ""
echo "****************** Installing cython-bbox ******************"
pip install cython
pip install cython-bbox

echo ""
echo ""
echo "****************** Installing tao ******************"
git config --system http.sslCAinfo /etc/ssl/certs/ca-certificates.crt
pip install git+https://github.com/TAO-Dataset/tao

echo ""
echo ""
echo "****************** Installing requirements.txt ******************"
pip install -r requirements.txt