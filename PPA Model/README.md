# PPA Model

## Follow each of these steps to check your configuration

## Dataset
The data set used by the author uses the Baidu network disk download tool, the download speed is slow, and it is not recommended to download directly.

https://drive.google.com/file/d/1wds1SVweGzKOfx-vaw15ouIbSLsOva_v/view?usp=share_link

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