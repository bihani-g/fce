import torch
from datasets import load_dataset
from pytorch_lightning import Trainer, seed_everything
from utils import *
import argparse
import warnings
import pickle
import numpy as np
import os
import torch
import pandas as pd
import skfuzzy
from tqdm import tqdm
from calibration_utils import *
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#-------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description='Data setup')
parser.add_argument('ds_name', type=str,
                    help='Dataset name (news/agnews/imdb)')
parser.add_argument('size', type=int,
                    help='Train data size')
parser.add_argument('bins', type=int,
                    help='Number of bins to calculate calibration error')
parser.add_argument('data_dir', type=str,
                    help='Directory location to save data')
parser.add_argument('result_dir', type=str,
                    help='Directory location to save results')


args = parser.parse_args()

size = args.size
ds_name = args.ds_name
filename = ds_name + str(size) + ".hf"
data_dir = args.data_dir
result_dir = args.result_dir
n_bins = args.bins

#-------------------------------------------------------------------------------------------------------------------#
# load dataset
if ds_name == 'news':
    data = load_dataset("SetFit/20_newsgroups")
elif ds_name == 'agnews':
    data = load_dataset("ag_news")
elif ds_name == 'imdb':
    data = load_dataset("imdb")

# saving dataset variation
create_data_split(data, size, ds_name, data_dir)

# setup input data
seed_everything(42)

dm = finetuning_data(model_name_or_path="bert-base-cased", data_name=ds_name, filename = filename,
                     data_dir=data_dir)
dm.setup("fit")

print("Train: {}".format(len(dm.dataset['train'])))
print("Test: {}".format(len(dm.dataset['test'])))

#-------------------------------------------------------------------------------------------------------------------#
# define model parameters
model = finetuner(model_name_or_path="bert-base-cased", num_labels=dm.num_labels)

# define training hyperparameters
trainer = Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None)

# model training
trainer.fit(model, datamodule=dm)
print("Fine-tuning done!")

# model eval
metrs = trainer.test(model, datamodule=dm)
outputs = trainer.predict(model, datamodule=dm, return_predictions=True)

print(metrs)

soft_preds = [y for x in outputs for y in x[0]]
preds = [y for x in outputs for y in x[1]]
labels = [y for x in outputs for y in x[2]]

dir_path = result_dir
# Check whether the specified path exists or not
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

print("Saving prediction probabilities...")

filename = ds_name + str(size)
preds_file = 'predicted_labels' + filename + '.pickle'
soft_preds_file = 'predict_probs' + filename + '.pickle'
labels_file = 'labels' + filename + '.pickle'

preds_path = os.path.join(dir_path, preds_file)
soft_preds_path = os.path.join(dir_path, soft_preds_file)
labels_path = os.path.join(dir_path, labels_file)

# save
with open(soft_preds_path, 'wb') as handle:
    pickle.dump(soft_preds, handle)

with open(preds_path, 'wb') as handle:
    pickle.dump(preds, handle)

with open(labels_path, 'wb') as handle:
    pickle.dump(labels, handle)
    
print("Saved prediction probabilities!")
#-------------------------------------------------------------------------------------------------------------------#
# analysis of predictions

data_name = ds_name+str(size)

ece_list = []
fce_list = []
uf_list = []
of_list = []
bins_list = []
ece_breakdown = []
fce_breakdown = []

print("Binning probabilities...")

for n_bins in tqdm(range(1, n_bins+1)):
    with open(os.path.join(result_dir,'predict_probs'+data_name+'.pickle'), 'rb') as handle:
        soft_preds = pickle.load(handle)
    with open(os.path.join(result_dir,'predicted_labels'+data_name+'.pickle'), 'rb') as handle:
        preds = pickle.load(handle)
    with open(os.path.join(result_dir,'labels'+data_name+'.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    incorrect = []
    correct = []
    for i in range(len(preds)):
        if preds[i] != labels[i]:
            incorrect.append([torch.softmax(soft_preds[i],0), int(preds[i]), int(labels[i])])
        else:
            correct.append([torch.softmax(soft_preds[i],0), int(preds[i]), int(labels[i])])

    # calculate overconfidence --> expectation of confidence over incorrect predictions
    of = []
    for i in range(len(incorrect)):
        of.append(max(incorrect[i][0]))

    # calculate underconfidence --> expectation of 1-confidence over correct predictions
    uf = []
    for i in range(len(correct)):
        uf.append(1-max(correct[i][0]))


    uf_list.append(np.mean(uf))
    of_list.append(np.mean(of))

    ece_dict = {'soft_preds': np.array([np.array(torch.softmax(x,0)) for x in soft_preds]), 
        'preds': [int(x) for x in preds], 
        'labels': [int(x) for x in labels]}


    ece_vals, ece = expected_calibration_error(np.array(ece_dict['labels']), 
                                                        ece_dict['soft_preds'], num_bins=n_bins)

    bins_list.append(n_bins)
    ece_list.append(ece)
    ece_breakdown.append(ece_vals)
    print("Done!")
    
    fce_vals, fce = fuzzy_calibration_error(np.array(ece_dict['labels']), ece_dict['soft_preds'], n_bins)

    fce_list.append(fce)
    fce_breakdown.append(fce_vals)
    print("Done!")



print("Saving calibration evaluation results...")
data = pd.DataFrame(list(zip(bins_list, uf_list, of_list, 
                             ece_list, ece_breakdown, fce_list, fce_breakdown)), 
               columns =['bins', 'uf', 'of', 'ece', 'ece_breakdown',
                         'fce', 'fce_breakdown'])

data.to_csv(os.path.join(result_dir,'calibration_eval.csv'), index = False)
print("Done!")


