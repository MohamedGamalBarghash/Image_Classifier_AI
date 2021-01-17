import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from collections import OrderedDict

import importing_data as imd
from functions import train_model, check_validation_set, parser_fun_train, create_checkpoint

import json
import argparse

def main ():
    args = parser_fun_train()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    #training the model
    optimizer, model = train_model(args.arch, imd.train_dl, imd.valid_dl, args.epochs, 40, args.device, args.hidden_units, args.lr)
    create_checkpoint(model, '', args.arch, imd.training_dataset.class_to_idx, optimizer, args.epochs)
    
if __name__ == '__main__': main()