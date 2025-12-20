# Fork of Orion

This is a fork of Orion used to train a Fully Homomorphic Encrypted version of a ResNet20.

This repository was a way to test some code for a Deep Learning project. Our code is mainly in `examples/run_resnet` and `examples/run_perso` and simply make some FHE inference. We were looking for some information about quality of encrypted inference, time of inference and memory usage (which is enormous).

The main repository of the project is here : https://github.com/josselinonduty/fully-homomorphic-encrypted-classifier

## Use for Mesonet

If you intend to use this code on Mesonet (a cluster of GPU for fast calculation) you could simply use the two SLURM file :
- `slurm.sl` will execute `run_resnet` and train a ResNet20 on CIFAR-10
- `slurm_perso.sl` will execute `run_perso` and make some encrypted inference on test image

## General use 

This part is the original README of Orion if you want to run the code on your laptop (be aware that it is advised to have more than 16Gb of RAM for FHE calculation with this code)

We tested our implementation on `Ubuntu 22.04.5 LTS`. First, install the required dependencies:

```
sudo apt update && sudo apt install -y \
    build-essential git wget curl ca-certificates \
    python3 python3-pip python3-venv \
    unzip pkg-config libgmp-dev libssl-dev
```

Install Go (for Lattigo backend):

```
cd /tmp
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
go version # go version go1.22.3 linux/amd64
```

### Install Orion

```
git clone https://github.com/mathis-saunier/orion
cd orion/
pip install -e .
```

### Run the examples!

```
cd examples/
python3 run_lola.py
```
