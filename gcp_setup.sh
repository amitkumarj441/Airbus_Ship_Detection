#!/bin/bash
set -e
set -v

# NVIDIA repo
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
rm ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb

# Install CUDA and system dependencies for Python
sudo apt-get update && sudo apt-get install -y --allow-unauthenticated cuda-9.1 imagemagick unzip make build-essential \
             libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
             libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
             xz-utils tk-dev libturbojpeg && sudo apt-get clean

# Instal CuDNN
curl -O http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-9.0-linux-x64-v7.tgz
tar -xvf ./cudnn-9.0-linux-x64-v7.tgz -C ./
sudo cp -P ./cuda/lib64/* /usr/local/cuda/lib64
sudo cp ./cuda/include/* /usr/local/cuda/include
rm -rf ./cuda
rm cudnn-9.0-linux-x64-v7.tgz

# Install pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# env setup for pyenv and CUDA
export PYENV_ROOT="${HOME}/.pyenv"
echo "export PATH=\"${PYENV_ROOT}/bin:\$PATH\"" >> ~/.profile
echo "eval \"\$(pyenv init -)\"" >> ~/.profile
echo "eval \"\$(pyenv virtualenv-init -)\"" >> ~/.profile
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.profile
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.profile
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.profile
source ~/.profile

# Install Python and project dependencies
pyenv install 3.6.3
pyenv virtualenv 3.6.3 kaggle-airbus-3.6.3
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install -r requirements.txt
