#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Network on Danish Fungi 2020 Dataset

# In[1]:
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from src.core import models, metrics, training, data, loss_functions
from src.utils import nb_setup
from src.dev import experiments as exp

DATA_DIR = 'data/danish_fungi_dataset/'
TRAIN_SET_DIR = 'train_resized'

SEED = 42

nb_setup.init()
nb_setup.set_random_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# In[2]:

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='efficientnet_b0')
parser.add_argument('--no_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--total_batch_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='mini')
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--weight', type=str, default='none')

parser.add_argument('--gamma', type=float, default=None)
parser.add_argument('--beta', type=float, default=None)
args, _ = parser.parse_known_args()

loss_kwg = {}
if args.gamma is not None:
    loss_kwg['gamma'] = float(args.gamma)
weight_kwg = {}
if args.beta is not None:
    weight_kwg['beta'] = float(args.beta)


# create training 
config = exp.create_config(
    data='df2020',
    path=DATA_DIR,
    model=args.model,
    loss=args.loss,
    opt='sgd',
    no_epochs=args.no_epochs,
    batch_size=args.batch_size,
    total_batch_size=args.total_batch_size,
    learning_rate=0.01,
    weight=args.weight,
    dataset=args.dataset,
    scheduler='reduce_lr_on_plateau',
    **loss_kwg,
    **weight_kwg,
    # note=''
)

# include configuration from model
_model_config = models.get_model(config.model, pretrained=False).pretrained_config
config.update(_model_config)

# save config file
config.save(DATA_DIR + config.specs_name)

# create loss, optimizer and scheduler functions
loss_fn = loss_functions.LOSSES[config.loss]
weight_fn = loss_functions.WEIGHTING[config.weight]
opt_fn = training.OPTIMIZERS[config.opt]
sched_fn = training.SCHEDULERS[config.scheduler]

DATASETS = {
    'full': ('DF20-train_metadata_PROD.csv', 'DF20-public_test_metadata_PROD.csv'),
    'mini': ('DF20M-train_metadata_PROD.csv', 'DF20M-public_test_metadata_PROD.csv')
}

print(config)


# ## Load the Data

# In[3]:


# load metadata
train_df = pd.read_csv(DATA_DIR + DATASETS[config.dataset][0])
valid_df = pd.read_csv(DATA_DIR + DATASETS[config.dataset][1])

classes = np.unique(train_df['scientificName'])
no_classes = len(classes)
assert no_classes == len(np.unique(valid_df['scientificName']))
print(f'No classes: {no_classes}')
print(f'Train set length: {len(train_df):,d}')
print(f'Validation set length: {len(valid_df):,d}')


# In[4]:


# create transforms
train_tfms, valid_tfms = data.get_transforms(
    size=config.input_size, mean=config.image_mean,
    std=config.image_std)

# create data loaders
trainloader = data.get_dataloader(
    train_df, img_path_col='image_path', label_col='scientificName',
    path=DATA_DIR + TRAIN_SET_DIR, transforms=train_tfms, labels=classes,
    batch_size=config.batch_size, shuffle=True, num_workers=4)
validloader = data.get_dataloader(
    valid_df, img_path_col='image_path', label_col='scientificName',
    path=DATA_DIR + TRAIN_SET_DIR, transforms=valid_tfms, labels=classes,
    batch_size=config.batch_size, shuffle=False, num_workers=4)

trainloader.dataset.show_items()


# In[ ]:





# ## Train the Model

# In[5]:


# create model
model = models.get_model(config.model, no_classes, pretrained=True)
assert np.all([param.requires_grad for param in model.parameters()])

# create loss
freq = train_df['scientificName'].value_counts()[trainloader.dataset.labels].values
weights = weight_fn(freq, **weight_kwg)
criterion = loss_fn(weight=torch.Tensor(weights).to(device) if weights is not None else None,
                    **loss_kwg)

# create trainer
trainer = training.Trainer(
    model,
    trainloader,
    criterion,
    opt_fn,
    sched_fn,
    validloader=validloader,
    accumulation_steps=config.total_batch_size // config.batch_size,
    path=DATA_DIR,
    model_filename=config.model_name,
    history_filename=config.history_file,
    device=device)


# In[6]:


# train model
trainer.train(no_epochs=config.no_epochs, lr=config.learning_rate)


# In[7]:


# find learning rate
# lr_finder = trainer.lr_find()


# In[ ]:




