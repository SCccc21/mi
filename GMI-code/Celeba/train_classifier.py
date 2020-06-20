import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

import random
import time
from classify import *
from utils import *

if __name__ == "__main__":

    file = "./config/classify" + ".json"
    args = load_params(json_file=file)

    train_path = args['dataset']['train_file_path']
    val_path = = args['dataset']['test_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    epochs = args[model_name]['epochs']

    print("---------------------Training [%s]------------------------------" % model_name)

    if model_name.startswith("VGG16"):
        model = VGG16(1000)
    elif model_name.startswith('IR152'):
        model = IR152(1000)
    elif model_name == "FaceNet64":
        model = FaceNet64(1000)


    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    train_set, train_loader = init_dataloader(args, train_path, batch_size)
    test_set, test_loader = init_dataloader(args, test_path, batch_size)
    
    iden_path = '/home/sichen/data/identity_CelebA.txt'

    for e in range(epochs):




    path_V = "attack_model/" + model_name + ".tar"
    torch.save(, path_V)
