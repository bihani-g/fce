import datasets
import csv
import torch
from datasets import load_dataset
from pytorch_lightning import Trainer, seed_everything
from utils import *
import argparse
import csv
import warnings
import gc
from os.path import exists
import pickle


warnings.filterwarnings("ignore")

##----add argparser
# parser = argparse.ArgumentParser(description='Data setup')
# parser.add_argument('ds_name', type = str,
#                     help='name of the dataset (news/agnews/imdb)')
# parser.add_argument('size', type=int,
#                     help='number of entries in training data (50/100/500/1000/2000/5000/10000)')

# args = parser.parse_args()

# size = args.size
# ds_name = args.ds_name
# data_dir = ds_name+str(size)+".hf"

# size = 50
# ds_name = 'news'
# data_dir = ds_name+str(size)+".hf"


# load hf dataset
if ds_name == 'news':
    data = load_dataset("SetFit/20_newsgroups")
elif ds_name == 'agnews':
    data = load_dataset("ag_news")
elif ds_name == 'imdb':
    data = load_dataset("imdb")


# saving dataset variation
create_data_split(data, size, ds_name)
    
#setup input data
seed_everything(42)

dm = finetuning_data(model_name_or_path="bert-base-cased", data_name = ds_name, data_dir=data_dir, data_path = "./conf_data", )
dm.setup("fit")

print("Train: {}".format(len(dm.dataset['train'])))
print("Test: {}".format(len(dm.dataset['test'])))


#define model parameters
model = finetuner(model_name_or_path="bert-base-cased", num_labels = dm.num_labels)

#define training hyperparameters
trainer = Trainer(
    max_epochs=1,
    accelerator = "auto",
    devices=1 if torch.cuda.is_available() else None)

#model training
trainer.fit(model, datamodule=dm)
print("Training done!")

#model eval
metrs = trainer.test(model, datamodule = dm)
print("Testing done!")
outputs = trainer.predict(model, datamodule = dm, return_predictions=True)

print(metrs)

print("Fine-tuning done!")


soft_preds = [y for x in outputs for y in x[0]]
preds = [y for x in outputs for y in x[1]]
labels = [y for x in outputs for y in x[2]]



dir_path = "./demo_results"
# Check whether the specified path exists or not
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

print(" saving confidence data")
filename = ds_name+str(size)
preds_file = 'preds'+filename+'.pickle'
soft_preds_file = 'soft_preds'+filename+'.pickle'
labels_file = 'labels'+filename+'.pickle'

preds_path = os.path.join(dir_path, preds_file)
soft_preds_path = os.path.join(dir_path, soft_preds_file)
labels_path = os.path.join(dir_path, labels_file)


# save
with open(soft_preds_path, 'wb') as handle:
    pickle.dump(soft_preds, handle)

print("soft preds saved!")

with open(preds_path, 'wb') as handle:
    pickle.dump(preds, handle)

print("preds saved!")

with open(labels_path, 'wb') as handle:
    pickle.dump(labels, handle)
    
print("labels saved!")