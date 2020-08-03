import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import shutil

import random
from classify import *
from utils import *
# check, if file exists, make link

def get_parser():
    parser = ArgumentParser(description='private domain image')
    parser.add_argument('--img_path', type=str, default="/home/sichen/data/img_align_celeba_png", help='dataset name')
    parser.add_argument('--train_root', type=str, default=False, help='training data root')

 
if __name__ == '__main__':
    global args

    file = "./config/classify.json"
    args = load_json(json_file=file)
    file_path = '/home/sichen/mi/GMI-code/Celeba/fid/private_list.txt'
    save_img_dir = "./fid/private_domain"
    os.makedirs(save_img_dir, exist_ok=True)

    private_set, private_loader = init_dataloader(args, file_path, mode="train")

    for i, (imgs, iden) in enumerate(private_loader):
        print("-------------- Process batch {} -----------------".format(i))
        for b in range(imgs.shape[0]):
            save_tensor_images(imgs[b], os.path.join(save_img_dir, "iden_{}.png".format(iden[b]+1)))
            print("save image of iden {}".format(iden[b]+1))