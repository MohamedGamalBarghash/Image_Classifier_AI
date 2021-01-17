import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from collections import OrderedDict

import importing_data as imd
from functions import train_model, check_validation_set, parser_fun_train

import json
import argparse

def main ():
    args = parser_fun_train()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    if args.arch == 'vgg19':
        model = models.vgg19(pretrained = True)      
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained = True)
    else:
        print("Please try for vgg19 or densenet121")
        
    torch.set_grad_enabled(True);
    for param in model.parameters():
        param.requires_grad = False
    
    #training the model
    optimizer = train_model(model, imd.train_dl, imd.valid_dl, args.epochs, 40, args.device, args.hidden_units, args.lr)

    model.class_to_idx = imd.training_dataset.class_to_idx

    checkpoint = {'transfer_model': 'vgg19',
                  'input_size': 25088,
                  'output_size': 102,
                  'features': model.features,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'idx_to_class': {v: k for k, v in imd.training_dataset.class_to_idx.items()}
                 }

    torch.save(checkpoint, "trained_model.pth")
    
if __name__ == '__main__': main()