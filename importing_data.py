from torchvision import models, datasets, transforms
import torch

# getting the paths
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# transforming the pics
training_transforms = transforms.Compose([transforms.RandomRotation(25),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(225),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
training_dataset = datasets.ImageFolder(train_dir, transform = training_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform = testing_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_dl = torch.utils.data.DataLoader(training_dataset, batch_size = 64, shuffle = True)
valid_dl = torch.utils.data.DataLoader(validation_dataset, batch_size = 32, shuffle = True)
test_dl = torch.utils.data.DataLoader(testing_dataset, batch_size = 32, shuffle = True)