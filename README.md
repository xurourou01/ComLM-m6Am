# ComLM-m6Am
Combining multiple pre-trained RNA language models for accurate N6,2'-O-dimethyladenosine site prediction.

This repository contains the PyTorch implementation of ComLM-m6Am.


## setup
### Requirements
python==3.9.25

biopython==1.85

numpy==2.0.2

openpyxl==3.1.5

pandas==2.3.3

scikit-learn==1.6.1

scipy==1.13.1

tqdm==4.67.1

tokenizers==0.22.2

torch==2.8.0+cu128

transformers==4.57.6

wheel==0.45.1

### Create an environment
We highly recommend using a virtual environment for the installation of ComLM-m6Am and its dependencies. A virtual environment can be created and (de)activated as follows by using conda:

```sh
# create
conda create -n ComLM-m6Am_Env python=3.9
# activate
conda activate ComLM-m6Am_Env
```

### Install ComLM-m6Am
After creating and activating the environment, download and install ComLM-m6Am from github:

```sh
git clone https://github.com/CSUBioGroup/ComLM-m6Am.git
cd ComLM-m6Am
pip install -r requirements.txt
```


## Usage
### Data
The training and testing data are in the `dataset` folder, and the different sequence information extracted by the pre-trained language models RNA-FM, SpliceBERT, and BiRNA-BERT is in the `features` folder.

### How to use your own data
First, you need to store your dataset in the `dataset` folder.
Second, you can extract features by changing the RnaLLM in the following command:

```bash
python extract_features_scripts/extract_RnaLLM_features.py
```
* The RnaLLM refers to RNAFM, SpliceBERT, or BiRNABert.

### Training
In the `config.py` file, you can adjust the hyperparameters. 

>In the `config.py`, the meaning of the variables is explained as follows:
>
>> ***RANDOM_SEED*** is the seed for model initialization.
>> 
>> ***BATCH_SIZE*** is the batchsize of training.
>>
>> ***EPOCHS*** is the largest number of training epochs.
>> 
>> ***PATIENCE*** is the parameter corresponding to the early stop method.
>> 
>> ***LEARNING_RATE*** is the learning rate of training.
>> 
>> ***CHECKPOINT_DIR*** is the folder where the model is saved.
>> 
>> ***DEVICE*** is the device you used to build and train the model. It can be "cuda" for gpu if torch.cuda.is_available() else "cpu" for cpu.

If you don't modify the file, the default parameters will be used for training. To start the training, just run the following command:

```bash
python train.py
```

`train.py` performs ten‑fold cross‑validation on the ten sub-training sets, primarily for model selection and hyperparameter tuning, while the models saved in the `checkpoints` folder are used for internal validation.

### Testing
For testing, You can run the following command to get the prediction results:

```bash
python test.py
```

In the testing stage, models are retrained from scratch, with each sub‑training set independently producing one final model. The final performance is evaluated by averaging the predictions from all ten models.



## Citation
Ying Xu #, Qianpei Liu #, Yiming Li, Wenkang Wang, Min Li, and Min Zeng*,"ComLM-m6Am: Combining multiple pre-trained RNA language models for accurate N6,2'-O-dimethyladenosine site prediction"
