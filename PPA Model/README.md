# PPA Model

## Follow each of these steps to check your configuration

## Dataset
The data set used by the author uses the Baidu network disk download tool, the download speed is slow, and it is not recommended to download directly.

https://drive.google.com/file/d/1wds1SVweGzKOfx-vaw15ouIbSLsOva_v/view?usp=share_link

## Important documents
`MainResponseWAE4Early.py`, run file for `Twitter15` and `Twitter16` dataset

`WeiboMainResponseWAE4Early.py` run file for `Weibo` dataset

`Pheme5MainResponseWAE4Early.py` run file for `PHEME` dataset

## Common error reports
1. typeerror: vampire.forward: return type `none` is not a `typing.dict[str, torch.tensor]`.

   **reason**: pytorch version problem

2. `valueerror("incompatible component merge:\n - '*mpich*'\n - 'mpi_mpich_*'")`
 
   **reason**: For anaconda version issues, it is recommended to use `conda install -c conda-forge <package_name>` one by one

3. from allennlp.modules.scalar_mix import scalarmix modulenotfounderror: no module named 'allennlp'
   
    **reason**: 
   
     [1] The allennlp.models and allnnlp versions must be the same.
  
     [2] It is recommended to install `allennlp.models` directly, so that the same version of `allnnlp` will be installed automatically
  
     [3] `Allennlp` package has strict version matching restrictions, including the following packages: `transformers`, `torch`, `torchvison`, `cach-path`, `huggingface-hub`, etc. 
     
     For example, the following error will appear: ERROR: cached-path 1.1.6 has requirement filelock<3.9,>=3.4, but you'll have filelock 3.0.12 which is incompatible. ERROR: cached-path 1.1.6 has requirement huggingface-hub<0.11.0,>=0.8.1, but you'll have huggingface-hub 0.13.1 which is incompatible. Similar to this, you need to choose the right version to match. 
 
     Details can be found in the `https://github.com/allenai/allennlp-models`

## Env
For specific environments, see the .yaml file

-------------
## CUDA
First go to `https://developer.nvidia.com/cuda-11.1.1-download-archive` download CUDA Toolkit.

Enter `nvcc -V` in the terminal to see the CUDA version.

My environment is `cuda11.1`.
1. Edit the `~/.bashrc` file
2. Change the PATH and LD_LIBRARY_PATH environment variables

`export PATH=/usr/local/cuda-11.1$PATH`

`export LD_LIRBRARY_PATH=/usr/local/cuda-11.1`

3. Save the content to exit, enter `source ~/.bashrc` to make the environment take effect.
4. Insert `nvcc -V`to look up the cuda tooltik.


After finish that:

`pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

*All installation packages can be found in the folder Env*

## Check cuda is available.
`import torch`

`print(torch._version_)`

`print(torch.cuda.is_available())`

`1.10.1+cu111`

`True`

*If the output is used above, it proves that CUDA is available*