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