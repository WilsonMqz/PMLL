# Preventing Privacy Leakage in Vision-Language Models: A Secure Framework for Large-Scale Image Classification

This is the official repository for our paper "Preventing Privacy Leakage in Vision-Language Models: A Secure Framework for Large-Scale Image Classification"


## Installation

Our code is built upon the official codebase of the [CoOp](https://github.dev/KaiyangZhou/CoOp) paper and has been 
tested in an environment with `python 3.8.8` and `pytorch 13.1.1` compiled with `CUDA 11.1`. 

As a first step, install `dassl` library in your environment by following the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

To further install all other dependencies, please run the following command, after having your environment activated:

```
pip install -r requirements.txt
```

## Datasets

Under `/` first make an empty data folder: 

```
mkdir data
```

Then download and structure your datasets according to the instructions provided in the [CoOp](https://github.dev/KaiyangZhou/CoOp)
official repository. All the `8` datasets should be present in the `data/` directory.
Run the `generate_clip_pseudo_labels.py` and `generate_qwen_pseudo_labels.py` to generate the privacy-masked labels.


## Experiments
Run the following demos:
```
python RC_estimator.py --dataset_name caltech101
```
