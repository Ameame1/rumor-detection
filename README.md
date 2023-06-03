# Rumor-Detection Documentation
## Table of Contents

- [GLAN Model](#glan-model)
  - [Env](#env)
  - [CUDA](#cuda)
  - [Dataset](#dataset)
  - [Train Models](#train-models)
  
- [PLAN Model](#plan-model)
  - [Dataset](#dataset-1)
  - [Env](#env-1)
  - [CUDA](#cuda-1)
  - [Check CUDA availability](#check-cuda-availability)
  - [Common error reports](#common-error-reports)
  - [Train Models](#train-models-1)
  - [Save Model](#save-model)
  
- [PPA Model](#ppa-model)
  - [Dataset](#dataset-2)
  - [Env](#env-2)
  - [CUDA](#cuda-2)
  - [Check CUDA availability](#check-cuda-availability-1)
  - [Common error reports](#common-error-reports-1)
  - [Train Model](#train-model)
  
- [Essay](#essay)
  - [GLAN](#glan)
  - [PLAN](#plan)
  - [PPA](#ppa)
  
- [Anaconda Env](#anaconda-env)
  - [GLAN](#glan-1)
  - [PLAN](#plan-1)
  - [PPA](#ppa-1)
  
- [Research Logs](#research-logs)
  - [Log-1](#log-1)
  - [Log-2](#log-2)
  - [Log-3](#log-3)
  - [Log-4](#log-4)
  - [Log-5](#log-5)
  
- [Experiment Results](#experiment-results)
  - [Initial Data](#initial-data)
  - [2-label Data](#2-label-data)
  
- [Author](#author)
  - [Ame Liu](#ame-liu)

## GLAN Model

### Env
Install GLAN model dependencies: 
```bash
pip install -r requirements.txt
```
-------------


### CUDA
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

### Check cuda is available
```bash
python3 check.py
```
*Check the cuda version is correct or not*


### Dataset
`GLAN/Dataset` contains three dataset: `twitter15`, `twitter16` and `weibo`

`.pkl` files are the model files.`dev.pkl`, `train.pkl`, `test.pkl`

`vocab.pkl` used to  the preprocess word vectors.


`replies.rar` file is the replies, can't open that.

`graph.txt` is the graph file, display different tweets' relationship.

### Train Models
[1] Change train dataset
```angular2html
if __name__ == '__main__':
    #task = 'twitter16'
    # task = 'twitter16'
    task = 'twitter16'
    print("task: ", task)

    if task == 'weibo':
        config['num_classes'] = 2
        config['batch_size'] = 64
        config['reg'] = 1e-5
        config['target_names'] = ['NR', 'FR']

    model = GLAN
    train_and_test(model, task)
```
Change the task name.(`twitter15`,`twitter16`,`weibo`)
[2] Run the main file
```bash
python3 run_new.py
```
[3] If you want to retrain the model, open these:
```angular2html
 #train codes:
    nn.fit(X_train_tid, X_train_source, X_train_replies, y_train,
    X_dev_tid, X_dev_source, X_dev_replies, y_dev)
    nn=nn.cuda()


```
If there is no need for retraining, please close the interface and directly use the pre-trained model.

[4] If you want to change the test dataset, such as using the twitter15 model to run the twitter16 dataset:
```angular2html
    #test dataset
    y_pred = nn.predict(X_test_tid_twitter16, X_test_source_twitter16,X_test_replies_twitter16)
    print(classification_report(y_test_twitter16, y_pred, target_names=config_16['target_names'], digits=3)) #y =Wx+b  b is the bias, models decide the W.
```
replace different dataset name you want to use.

#### PS: Because the three dataset contain different number of nodes, the bigger on can test the smaller one.

And the sequence as big to small is **weibo(20493)** > **twitter15(4459)** > **twitter16(3550)**

Therefore, we can use GLAN(weibo trained) to run twitter15 and twitter16 as test datasets.
Using twitter15 to run twitter16 as test dataset.
Twitter16 only can run by itself, no others.





## PLAN Model

### Dataset
https://www.dropbox.com/sh/w3bh1crt6estijo/AAD9p5m5DceM0z63JOzFV7fxa?dl=0

glove english word vector: https://drive.google.com/file/d/19t9-sihELSk0Asf-EFAOCb9WfyPgtRMu/view?usp=sharing

word2vec chinese word vector: 

the glove and word2vec word vectors files will use to the word vectors preprocess.

Download these files and extract to `GLAN/code/dataset` folder.
### Env
Install PLAN model dependencies: 
```bash
pip install -r requirements.txt
```
-------------


### CUDA
First go to `https://developer.nvidia.com/cuda-11.1.1-download-archive` download CUDA Toolkit.

Enter `nvcc -V` in the terminal to see the CUDA version.

My PLAN environment is `cuda11.1`.
1. Edit the `~/.bashrc` file
2. Change the PATH and LD_LIBRARY_PATH environment variables

`export PATH=/usr/local/cuda-11.6$PATH`

`export LD_LIRBRARY_PATH=/usr/local/cuda-11.6`

3. Save the content to exit, enter `source ~/.bashrc` to make the environment take effect.
4. Insert `nvcc -V`to look up the cuda tooltik.

After finish that:

`pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html`


*All installation packages can be found in the folder Env*

### Check cuda is available
```bash
python3 check.py
```
*Check the cuda version is correct or not*

### Common error reports

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

### Train Models
**[1] Config train settings**

Open the `PLAM/codes/config_new.py` file.

**Train settings:**
```angular2html
      # Training
        self.num_epoch = 300
        self.batch_size = 4
        self.batch_size_test = 4
        self.num_classes = 4
```
**Word Embedding settings:**
Decided by your train dataset, English datasets use the `glove/6B.300d.txt` file, Chinese datasets use the `word2vec.txt` file.

```angular2html
self.glove_directory = "/home/ame/rumor/PLAN/codes/data/word2vec/"
        self.glove_file = "word2vec.txt"
        self.vector_path = ""
        self.vocab_path = ""
        self.max_vocab = 20000
        self.emb_dim = 300
        self.num_structure_index = 5
```
Other settings please look the config file, many detail remark have been indicated well.

**[2] Config test settings**

Open the `PLAN/codes/test_new.ipynb`

**Choose model:**
```angular2html
folder = "/home/ame/rumor/PLAN/logs/weibo_full_repair_model/weibo_full_repair/best_model"
```
**Choose test datasets:**
```angular2html
test_file_path = "/home/ame/rumor/PLAN/codes/data/twitter15_16/twitter16/split_data/structure_v2/split_4/train_unique_w_structure_v2_modified.json"
```
**Choose word vector preprocess file:**
```angular2html
# Getting the vocab vectors
#Chinese
vec = vocab.Vectors(name = word2vec_file, cache = word2vec_directory)
#English
#vec = vocab.Vectors(name =glove_file, cache =glove_directory)
```

You can use above three steps to run different PLAN models test different datasets.

**[3] Add new dataset**

You can add new datasets in any language, provided that the json file format of the PLAN model is followed as shown in the following example:
```angular2html
{"id_": 552783745565347840, "label": 0, "tweets": ["Ten killed in shooting at headquarters of French satirical weekly Charlie Hebdo, says French media citing witnesses #c4news", "@Channel4News @GidonShaviv must be that peace loving religion again", "@Channel4News my god what is going on,no doubt as it seems to be satirical mag,members of a certain religion have been offended they kill.", "@Channel4News @theresacfc ffs what's going on??", "@Channel4News I think the majority of people ( including you guys) are sick and tired of Islamic immigration and all the problems it brings", "@Channel4News 1 cleric for each victim, this will stop the hate speech from mosques. the streets will echo with silence by outraged muslims"], "time_delay": [0, 0, 0, 2, 3, 8], "structure": [[4, 2, 2, 2, 2, 2], [3, 4, 2, 2, 2, 2], [3, 3, 4, 2, 2, 2], [3, 3, 3, 4, 2, 2], [3, 3, 3, 3, 4, 2], [3, 3, 3, 3, 3, 4]]}
```
The json file must contain the `id`, `label`, `tweets`.

`time_dealy` and `structure` these two parameters are extra parameters used to improve the accuracy.


[4] Extra python tools package

`Extra tools.ipynb` this file contain many functions for this PLAN model research, it is very helpful when want to do some files process operations.

This is the list functions for this jupyter notebook file:
```angular2html
1. Consolidate all the information in TreeWeibo into one file weibo_timedelay.txt
2. Integrate original data with comment data weibo_id_text.txt + weibo_timedelay.txt
3. Re-integrate into the original data content and message categories label: weibo.txt
4. Convert a txt file to a json file in PLAN data format.
5. Count the number of rows in the newly generated json file.
6. Counting the number of different elements in a txt file.
7. Counting txt file columns (to prevent data confusion).
8. Allocation of datasets at 70% 15% 15%.
9. Count the largest element value and the largest number of elements contained in time_delay[] in all json data.
10. Change the time_dealy[] in the Weibo dataset to a maximum of 100.
11. Instead of changing data greater than 100 to 100, the data in the corresponding tweet[] is deleted and the data in the corresponding tweet[] is deleted.
12. The label parameter in the statistics weibo data.
13. Delete the elements of time_dealy[] from the Weibo dataset.
14. Splitting json data.
15. Look up json files not recognised by the PLAN model.
16. Merge json data.
17. We divided the Weibo dataset into 5 parts and selected one of them for testing, following the ratio of 70% for the training set, 15% for test set 1, and 15% for test set 2.
18. Create weibo dataset, correct label.
19. Statistic twitter15/16 labels
20. Change the twitter15 label.
```

**PS: Since the PLAN model is based on TensorFlow, the training time is very long, but the testing phase does not take very long.**



## PPA Model


### Dataset
The data set used by the author uses the Baidu network disk download tool, the download speed is slow, and it is not recommended to download directly.

https://drive.google.com/file/d/1wds1SVweGzKOfx-vaw15ouIbSLsOva_v/view?usp=share_link

Extract this folder into `PPA/data/`.

### Env

Install PPA model dependencies: 
```bash
pip install -r requirements.txt
```
-------------


### CUDA
First go to `https://developer.nvidia.com/cuda-11.1.1-download-archive` download CUDA Toolkit.

Enter `nvcc -V` in the terminal to see the CUDA version.

My PPA environment is `cuda11.1`.
1. Edit the `~/.bashrc` file
2. Change the PATH and LD_LIBRARY_PATH environment variables

`export PATH=/usr/local/cuda-11.1$PATH`

`export LD_LIRBRARY_PATH=/usr/local/cuda-11.1`

3. Save the content to exit, enter `source ~/.bashrc` to make the environment take effect.
4. Insert `nvcc -V`to look up the cuda tooltik.


After finish that:

`pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

*All installation packages can be found in the folder Env*

### Check cuda is available
```bash
python3 check.py
```
*Check the cuda version is correct or not*

### Common error reports
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


### Train Model
**[1] Config settings:**
Open the `PPA/get_args.py` file.

In this file you can set different experiment parameters.

More details have remarked in this file.

**[2] Choose Dataset**
In `PPA/get_args.py` file:

```angular2html
cur_dataset = "T"
if cur_dataset == "T":
    _args = get_response_twitter_args()
elif cur_dataset == "W":
    _args = get_response_weibo_args()
elif cur_dataset == "P":
    _args = get_response_pheme5_args()
```
T = Twitter

W = Weibo

P = Pheme

**[3] Three main file:**

```bash
python3 MainResponseWAE4Early.py
```
Running file for `Twitter15` and `Twitter16` dataset.
```bash
python3 WeiboMainResponseWAE4Early.py
```
Running file for `Weibo` dataset.
```bash
python3 Pheme5MainResponseWAE4Early.py
```
Running file for `PHEME` dataset.

### Save Model
The initial codes can't save PPA model, so we can add these codes in the three main files.
```angular2html
# save model path
        torch.save({"model" : model.state_dict()}, "./model_Twitter16/e" + str(epoch))

# choose test model
        model.load_state_dict(torch.load("./model_Twitter/e5")['model'])
        model.eval()
```
More details I have uploaded the three new main files which I have edited.
`MainResponseWAE4Early_new.py`, `Pheme5MainResponseWAE4Early_new.py`, `WeiboMainResponseWAE4Early_new.py`.


## Eassy

**GLAN**: https://arxiv.org/abs/1909.04465

**PLAN**: https://arxiv.org/abs/2001.10667

**PPA**: https://dl.acm.org/doi/abs/10.1016/j.neucom.2021.06.062


## Anaconda Env

If anyone can not config settings successfully, you can download my Anaconda Env files.

After that, unzip file and copy it to `Anaconda/Env` folder.

**GLAN**: https://drive.google.com/file/d/1zE63-azXCibmmpYtbTT12T6IkEyJXPhb/view?usp=sharing

**PLAN**: https://drive.google.com/file/d/1T2iczGtxvWPnTG5sKwIZ0-zPxkND0kOD/view?usp=sharing

**PPA**: https://drive.google.com/file/d/1chHpT438zq5PAV4LQhPljMMKq4ruXvya/view?usp=sharing

## Research Logs

_These journals record some of the explorations and ongoing discoveries of my research._


[1] Log-1
https://docs.google.com/document/d/1J3ds9t9AVhWc1-yS0qWNCL7c5mk4htvVDQP0hMmLurc/edit?usp=sharing

[2] Log-2
https://docs.google.com/document/d/1MqqaPN2yyNCqYchef9Rp4jxLqTOMEwMhSI2720HJSgY/edit?usp=sharing

[3] Log-3
https://docs.google.com/document/d/1b7XhMxdSiJBcyi_t8At_zfyyQa8LJ3QFw2ELAqnvcJs/edit?usp=sharing

[4] Log-4
https://docs.google.com/document/d/1HWcbEVxvZapyKo1G2okZOZ35DmxiheiCuO0R2_ItuMA/edit?usp=sharing

[5] Log-5
https://docs.google.com/document/d/1mnCFlGI6EMH_l9v26BviOO7xud4Ih8h_G49-n5o_QOM/edit?usp=sharing

## Experiment Results
**Initial Data**

https://docs.google.com/document/d/1Y4V6_wrUUMPtHO0JWvLlPT340NoeNnr0zs1ol1523zI/edit?usp=sharing

**2-label Data**

https://docs.google.com/document/d/1eMzZLAdSBglXRhuK5An68221Sv2D4jyiKz1xphOI7wQ/edit?usp=sharing

## Author
### Ame Liu
[<img src="Figure/Email.png" alt="image" width="">](mailto:22910358@student.uwa.edu.au)
