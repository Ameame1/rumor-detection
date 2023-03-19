# PLAN Model

## Follow each of these steps to check your configuration

## Env
For specific environments, see the .yaml file

-------------


## CUDA
First go to `https://developer.nvidia.com/cuda-11.6.0-download-archive` download CUDA Toolkit.

Enter `nvcc -V` in the terminal to see the CUDA version.

My environment is `cuda11.6`.
1. Edit the `~/.bashrc` file
2. Change the PATH and LD_LIBRARY_PATH environment variables

`export PATH=/usr/local/cuda-11.6$PATH`

`export LD_LIRBRARY_PATH=/usr/local/cuda-11.6`

3. Save the content to exit, enter `source ~/.bashrc` to make the environment take effect.
4. Insert `nvcc -V`to look up the cuda tooltik.
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

1. OSError: [E941] Can't find model 'en'. It looks like you're trying to load a model from a shortcut, which is obsolete as of spaCy v3.0. To load the model, use its full name instead:

This is because the direct use of `pip install spacy`, the spacy version is not match with the author's old version.

Because older versions of spacy are not compatible with many of the latest packages, we use the following solution.

`import spacy`

`nlp = spacy.load('en')`

Replace above with

`import spacy`

`nlp = spacy.load('en_core_web_sm')`

2. error: "can't find model 'en'" it says to use its full name 'en_core_web_sm' but then i get the error: "cant find model 'en_core_web_sm' it doesnt seem to be a python package or a valid path to a data directory"

This is an issue where higher versions of Spacy are not compatible with previous older versions

You need to install en_core_web_sm

`python -m spacy download en_core_web_sm`



3. ImportError: cannot import name 'NestedField' from 'torchtext.data' (/home/ame/anaconda3/envs/PLAN/lib/python3.9/site-packages/torchtext/data/__init__.py)

This is an issue where later versions of TorchText are not compatible with older versions

Because older versions of spacy are not compatible with many of the latest packages, we use the following solution.

`pip install torch==0.9.1`

change `torchtext.data`becomes`torchtext.legacy.data`

After that, maybe new error occur:

userwarning: [w108] the rule-based lemmatizer did not find pos annotation for one or more tokens. check that your pipeline includes components that assign token.pos, typically 'tagger'+'attribute_ruler' or 'morphologizer'.

`nlp = spacy.load("en_core_web_sm", disable = ["parser", "tagger", "ner","lemmatizer"])`









