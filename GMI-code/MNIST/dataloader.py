import os, utils, torchvision
import json, PIL, time, random
import torch, math, cv2

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F 
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

class GrayFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        # print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "gan":
                img_name = line.strip()
            else:
                img_name, iden = line.strip().split(' ')
                label_list.append(int(iden))
            name_list.append(img_name)

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('L')
                img_list.append(img)
        return img_list
    
    def get_processor(self):
        proc = []
        if self.args['dataset']['name'] == "mnist":
            re_size = 32
        else:
            re_size = 64
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        if self.mode == "gan":
            return img
        label = self.label_list[index]
    
        return img, label

    def __len__(self):
        return self.num_img

