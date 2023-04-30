# Calibration Error Estimation Using Fuzzy Binning

## About fuzzy-binning

Estimation of calibration error in neural networks is done using metrics based on crisp-binning of prediction probabilities such as ECE and MCE. These metrics are vulnerable to the leftward-skew in model prediction probabilities. To address this issue, we propose Fuzzy Calibration Error (FCE) that utilizes a fuzzy binning approach. Using FCE reduces the impact of probability skew when measuring calibration error.

<p align="center">
<img src="https://github.com/bihani-g/fce/blob/main/figures/binning.png" width="500">
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

- Generating your own ECE and FCE scores
    - Run `fuzzy_binning.py` to generate ECE and FCE scores for your own predictions.

    ```bash
    python fuzzy_binning.py --predict_probs_pkl 
                          --predicted_labels
                          --labels
                          --bins
    ```
  The script requires 3 files in `.pickle` format. 
    - Softmax prediction probabilities (`predict_probs_pkl`)
    - Predicted labels (`predicted_labels`)
    - Actual labels (`labels`)
    

## Reproducing paper results

  - Run `./paper_demo/run.sh` to reproduce ECE and FCE scores given in the paper.

  - Run `./paper_demo/calibration_analysis.ipynb` to plot binning differences in ECE and FCE as shown in the paper.

A few examples comparing fuzzy and crisp binning and the reduced impact of probability skew on FCE calculations.


<p align="center">
<img src="https://github.com/bihani-g/fce/blob/main/figures/distrbn%20_ce_across_bins_5k.pdf">
</p>


<p align="center">
<img src="https://github.com/bihani-g/fce/blob/main/figures/ece_vs_fce_across_bins.png">
</p>


















