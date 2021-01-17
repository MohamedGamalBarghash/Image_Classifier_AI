import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from collections import OrderedDict

import importing_data as imd
from functions import load_model, check_validation_set, predict, parser_fun_test

import json
import argparse

def main ():
    args = parser_fun_test()
    
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
    
    # load the model
    model = load_model("trained_model.pth")
    
    # prediction
    probabilities = predict(args.image_path, model, args.top_k, 'gpu')
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    
    i=0
    while i < args.top_k:
        print("{} with a probability of {:.5f}".format(labels[i], probability[i]))
        i += 1
    print("Predictiion is done !")
    
if __name__ == '__main__': main()