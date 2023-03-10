# GLAN Model

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
pip install torch

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116         (cuda116)

pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl 

pip install torch_sparse-0.6.16+pt113cu116-cp39-cp39-linux_x86_64.whl

pip install torch_cluster-1.6.0+pt113cu116-cp39-cp39-linux_x86_64.whl

pip install torch_spline_conv-1.2.1+pt113cu116-cp39-cp39-linux_x86_64.whl

pip install torch-geometric==1.7.2

*All installation packages can be found in the folder Env*

## Check cuda is available.
import torch

import torch_geometric

print(torch._version_)

print(torch.cuda.is_available())

print(torch_geometric.__version__)

1.13.1+cu116

True

1.7.2

*If the output is used above, it proves that CUDA is available*

