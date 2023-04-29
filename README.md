# Calibration Error Estimation Using Fuzzy Binning

## About fuzzy-binning

Estimation of calibration error in neural networks is done using metrics based on crisp-binning of prediction probabilities such as ECE and MCE. These metrics are vulnerable to the leftward-skew in model prediction probabilities. To address this issue, we propose Fuzzy Calibration Error (FCE) that utilizes a fuzzy binning approach. Using FCE reduces the impact of probability skew and provides a tighter estimate when measuring calibration error.

This repository contains code and implementation for FCE, as described in the paper "Calibration Estimation Using Fuzzy Binning". 

## How to use this repository?

- Clone this repository
- Create a conda environment and install all the required python packages
```
conda create -n fce_env python=3.10.4
conda activate fce_env
conda install pytorch=2.0.0 torchvision=0.14.1 cudatoolkit=11.3.1 -c pytorch
pip install -r requirements.txt
```

- Run `finetune.py` to generate ECE and FCE scores.
```
python finetune.py --dataset
                   --size
                   --n_bins
                   --data_dir
                   --result_dir
```



- Run `calibration_analysis.ipynb` to visualize binning differences in ECE and FCE.











