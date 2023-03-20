# GLAN Model

## Follow each of these steps to check your configuration

## Env
For specific environments, see the .yaml file

-------------


## CUDA
First go to `https://developer.nvidia.com/cuda-11.6.0-download-archive` download CUDA Toolkit.

Enter `nvcc -V` in the terminal to see the CUDA version.

My GLAN environment is `cuda11.6`.
1. Edit the `~/.bashrc` file
2. Change the PATH and LD_LIBRARY_PATH environment variables

`export PATH=/usr/local/cuda-11.6$PATH`

`export LD_LIRBRARY_PATH=/usr/local/cuda-11.6`

3. Save the content to exit, enter `source ~/.bashrc` to make the environment take effect.
4. Insert `nvcc -V`to look up the cuda tooltik.


Need to install the corresponding version of torch+cuda

`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`         (cuda116)

`pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl` 

`pip install torch_sparse-0.6.16+pt113cu116-cp39-cp39-linux_x86_64.whl`

`pip install torch_cluster-1.6.0+pt113cu116-cp39-cp39-linux_x86_64.whl`

`pip install torch_spline_conv-1.2.1+pt113cu116-cp39-cp39-linux_x86_64.whl`

`pip install torch-geometric==1.7.2`

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

