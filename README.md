# TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series

## Requirements

 - Install Python>=3.8, PyTorch 1.8.1.
 - Numpy (`numpy`) v1.15.2;
 - Matplotlib (`matplotlib`) v3.0.0;
 - Orange (`Orange`) v3.18.0;
 - Pandas (`pandas`) v1.4.2;
 - Weke (`python-weka-wrapper3`) v0.1.6 for multivariate time series (requires Oracle JDK 8 or OpenJDK 8);
 - PyTorch (`torch`) v1.8.1 with CUDA 11.0;
 - Scikit-learn (`sklearn`) v1.0.2;
 - Scipy (`scipy`) v1.7.3;
 - Huggingface (`transformers`) v4.30.1;
 - Absl-py (`absl-py`) v1.2.0 ;
 - Einops (`einops`) v0.4.1;
 - H5PY (`h5py`) v3.7.0;
 - `keopscore` v2.1
 - `opt-einsum` v3.3.0
 - `pandas` v1.4.2 
 - `pytorch-wavelet` 
 - `PyWavelets` v1.4.1
 - `scikit-image` v0.19.3
 - `statsmodels` v0.13.2
 - `sympy` v1.11.1
 

## Datasets

The datasets manipulated in this code can be downloaded on the following locations:
 - the UCR archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/;
 - the UEA archive: http://www.timeseriesclassification.com/;

## Files

### Core
 - `datasets` data and related methods;
 - `encoders` folder: implements encoder and its building blocks (dilated
   convolutions, causal CNN);
 - `losses` folder: implements the triplet loss in the cases of a training set
   with all time series of the same length, and a training set with time series
   of unequal lengths;
 - `models` folder: implements LLM4TS and its building blocks (encoder + GPT attention + output head);
 - `utils` folder: utils;
 - `main_encoder` file: handles learning for encoders (see usage below);
 - `main_LLM4TS` file: handles learning for LLM4TS. The prerequisite is to have a well trained encoder (see usage below);
 - `optimizers` file: optimizer methods for training models;
 - `options` file: input args;
 - `running` file: methods to train and test models.




## Usage

### Selecting text prototype

Download LLM from huggingface

To select text prototypes from GPT2

`python losses/text_prototype.py --llm_model_dir= path/to/llm/folder/ --prototype_dir path/to/save/prototype/file/ --provide Flase(ramdom) or a text lisr --number_of_prototype 10`


### Training encoder on the UEA archives

To train a model on the EthanolConcentration dataset from the UEA archive with specific gpu:

`python main_encoder.py --data_dir path/to/EthanolConcentration/folder/ --gpu 0`

Adding the `--load_encoder` option allows to load a model from the specified save path.

Setting the `--gpu -1` option to use cpu.

### Training LLM4TS on the UEA archives

To train a model on the EthanolConcentration dataset from the UEA archive with specific gpu:

`python main_LLM4TS.py --output_dir experiments --comment "classification from Scratch" --name EthanolConcentration --records_file Classification_records.xls --data_dir path/to/EthanolConcentration/folder/  --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 50 --lr 0.001 --patch_size 8 --stride 8 --optimizer RAdam --d_model 768 --pos_encoding learnable --task classification
--key_metric accuracy --gpu 0`

Setting the `--gpu -1` option to use cpu.
