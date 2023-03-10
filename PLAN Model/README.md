# PLAN Model

## Follow each of these steps to check your configuration

## Env
For specific environments, see the .yaml file

-------------
**ame@ame-linux**

**Nvidia Driver**: 470

**OS**: Ubuntu 20.04.5 LTS ×86_64

**Kernel**: 5.15.0-67-generic

**Uptime**: 2 hours, 56 mins Packages: 1761 (dpkg), 10 (snap)

**Shell**: bash 5.0.17

**Resolution**: 2560×1440

**DE**: GNOME

**WM**: Mutter

**WM** Theme: Adwaita

**Theme**: Yaru [GTK2/3]

**Icons**: Yaru [GTK2/3]

**Terminal**: gnome-terminal

**CPU**: AMD Ryzen 7 5800X (16) @ 3.800G 

**GPU**: NVIDIA GeForce RTX 3080

**Мемогу**: 11287МіВ / 32004MiB

## CUDA
pip install torch==1.7.1

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116         (cuda116)

*All installation packages can be found in the folder Env*

## Check cuda is available.
import torch

print(torch._version_)

print(torch.cuda.is_available())

1.7.1

True

*If the output is used above, it proves that CUDA is available*

## Dataset
https://www.dropbox.com/sh/w3bh1crt6estijo/AAD9p5m5DceM0z63JOzFV7fxa?dl=0

glove word vector: https://drive.google.com/file/d/19t9-sihELSk0Asf-EFAOCb9WfyPgtRMu/view

## Common error reports

error: "can't find model 'en'" it says to use its full name 'en_core_web_sm' but then i get the error: "cant find model 'en_core_web_sm' it doesnt seem to be a python package or a valid path to a data directory"

This is an issue where higher versions of Spacy are not compatible with previous older versions

pip uninstall spacy

pip install spacy==2.1.2

ImportError: cannot import name 'NestedField' from 'torchtext.data' (/home/ame/anaconda3/envs/PLAN/lib/python3.9/site-packages/torchtext/data/__init__.py)

This is an issue where later versions of TorchText are not compatible with older versions

pip install torchtext==0.8.1
