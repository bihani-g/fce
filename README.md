# Calibration Error Estimation Using Fuzzy Binning

## About fuzzy-binning

Estimation of calibration error in neural networks is done using metrics based on crisp-binning of prediction probabilities such as ECE and MCE. These metrics are vulnerable to the leftward-skew in model prediction probabilities. To address this issue, we propose Fuzzy Calibration Error (FCE) that utilizes a fuzzy binning approach. Using FCE reduces the impact of probability skew when measuring calibration error.

<p align="center">
<img src="https://github.com/bihani-g/fce/blob/main/binning.png" width="500">
</p>

This repository contains code and implementation for FCE, as described in the paper "Calibration Estimation Using Fuzzy Binning". 

## How to use this repository?

- Clone this repository
- Create a conda environment and install all the required python packages
```bash
conda create -n fce_env python=3.10.4
conda activate fce_env
conda install pytorch=2.0.0 torchvision=0.14.1 cudatoolkit=11.3.1 -c pytorch
pip install -r requirements.txt
```

- Run `finetune.py` to generate ECE and FCE scores for text classification tasks given in the paper.
```bash
python finetune.py --dataset
                   --size
                   --n_bins
                   --data_dir
                   --result_dir
```


## Demo 
- To visualize binning differences in ECE and FCE, you can also run `calibration_analysis.ipynb` that uses prediction probabilities saved in `demo_data` for analysis.


A few examples comparing fuzzy and crisp binning and the reduced impact of probability skew on FCE calculations.


<p align="center">
<img src="https://github.com/bihani-g/fce/blob/main/distrbn%20_ce_across_bins_5k.pdf">
</p>


<p align="center">
<img src="https://github.com/bihani-g/fce/blob/main/ece_vs_fce_across_bins.png">
</p>
















