import argparse
import torch
import numpy as np
from torch import nn, optim
from torchvision import models, datasets, transforms
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image 
#define the function to use argparse
def parser_fun_train ():
    parser = argparse.ArgumentParser(description="Training Settings")

    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU or not', dest="device")
    parser.add_argument('--arch', type=str, default='vgg19', help='architecture [available: densenet121, vgg19]', dest="arch")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate', dest="lr")
    parser.add_argument('--hidden_units', type=int, default=4096, help='hidden units for fc layer', dest="hidden_units")
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs', dest="epochs")

    args = parser.parse_args()
    return args

def parser_fun_test():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('--img_path', action='store',nargs='?' , type = str 
                    ,default = 'flowers/test/102/image_08004.jpg',
                    help='Enter path to image.', dest='image_path')
    parser.add_argument('--top_k', action='store',
                    type=int, default = 5,
                    help='Enter number of top most likely classes to view, default is 3.')
    args = parser.parse_args()
    return args

# define the function to train the model
def train_model(model, train_dl, valid_dl, epochs, print_every, device, no_of_layers, lr):    
    epochs = epochs
    print_every = print_every
    steps = 0
    
    # define a classifier    
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, int(no_of_layers))),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(int(no_of_layers), 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # define the criterion (loss) and the optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
        
    # change to GPU
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in train_dl:
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                accuracy = check_validation_set(train_dl, 'cuda', model)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      ",Loss: {:.4f}".format(running_loss/print_every),
                      ",Validation Loss {:.4f}".format(loss / len(train_dl)),
                      ",Accuracy: {:.4f}".format(accuracy*100))

                running_loss = 0
    print("DONE TRAINING!")
    return optimizer
   

# define a function to check the values
def check_validation_set(valid_loader,device, model):    
    accuracy = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            
            images, labels = images.to('cuda'), labels.to('cuda')
            
            output = model.forward(images)
            
            ps = torch.exp(output)
            top_p, top_flower = ps.topk(1, dim=1)
            total += labels.size(0)
            equals = top_flower == labels.view(*top_flower.shape)
            accuracy += equals.type(torch.FloatTensor).sum().item()
    return accuracy / total


# define a function to load a checkpoint
def load_model (path):
    state_dict = torch.load(path)
    model = models.vgg19(pretrained = True)
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                               ]))

    model.load_state_dict(state_dict['state_dict'])
    return model

# process the image for classification    
def process_image(image):
    test_image = Image.open(image)
    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = prepoceess_img(test_image)
    return img_tensor

        
# define a function to predict the model
def predict(image_path , model, topk=5 , device = 'gpu'):   
    if device == 'gpu':
        model.to('cuda')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    
    else:
        with torch.no_grad():
            output = model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)