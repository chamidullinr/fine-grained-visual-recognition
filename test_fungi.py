#!/usr/bin/env python
# coding: utf-8

# # Test Network on Danish Fungi 2020 Dataset

# In[1]:
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from src.core import models, training, data, metrics
from src.utils import nb_setup, io

DATA_DIR = 'data/danish_fungi_dataset/'
TRAIN_SET_DIR = 'train_resized'

nb_setup.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--dataset', type=str, default='mini')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--output', type=str, required=True)
args, _ = parser.parse_known_args()

MODEL_ARCH = args.model
MODEL_NAME = args.checkpoint
BATCH_SIZE = args.batch_size
print(f'Creating predictions of {MODEL_ARCH} (checkpoint={MODEL_NAME}) on dataset {args.dataset}.')


# ## Load the Data

# In[2]:


# load metadata
if args.dataset == 'full':
    valid_df = pd.read_csv(DATA_DIR + 'DF20-public_test_metadata_PROD.csv')
else:
    valid_df = pd.read_csv(DATA_DIR + 'DF20M-public_test_metadata_PROD.csv')


classes = np.unique(valid_df['scientificName'])
no_classes = len(classes)
print(f'No classes: {no_classes}')
print(f'Test set length: {len(valid_df):,d}')


# ## Create Network and Dataloader

# In[3]:


# create fine-tuned network
model = models.get_model(MODEL_ARCH, no_classes, pretrained=False)
training.load_model(model, MODEL_NAME, path=DATA_DIR + 'models')
assert np.all([param.requires_grad for param in model.parameters()])

model_config = model.pretrained_config

# create transforms
_, valid_tfms = data.get_transforms(
    size=model_config['input_size'], mean=model_config['image_mean'],
    std=model_config['image_std'])

# create data loaders
validloader = data.get_dataloader(
    valid_df, img_path_col='image_path', label_col='scientificName',
    path=DATA_DIR + TRAIN_SET_DIR, transforms=valid_tfms, labels=classes,
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

validloader.dataset.show_items()


# ## Create Predictions

# In[4]:


# create predictions
pred, targ, _ = training.predict(model, validloader)

np.save(f'{args.output}_{MODEL_ARCH}_pred.npy', pred)
np.save(f'{args.output}_targ.npy', targ)