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



if __name__ == "__main__":

    global args
    file = "./config/classify" + ".json"
    args = load_params(json_file=file)

    model_name = args['dataset']['model_name']
    model_name_T = "VGG16"
    model_name_E = "FaceNet"
    dataset_name = "celeba"
   
    z_dim = 100

    path_G = '/home/sichen/models/celeba_G.tar'
    path_D = '/home/sichen/models/celeba_D.tar'
    path_T = '/home/sichen/models/target_model/VGG16/model_latest.pth'
    path_E = '/home/sichen/models/target_model/FaceNet/model_latest.pth'

    train_path = args['dataset']['train_file_path']
    val_path = args['dataset']['test_file_path']
    # lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']

    ###########################################
    ###########     load model       ##########
    ###########################################
    # no mask
    # G = Generator(z_dim)
    # torch.nn.DataParallel(G).cuda()
    # D = DGWGAN(3)
    # torch.nn.DataParallel(D).cuda()
    # ckp_G = torch.load(path_G)
    # load_my_state_dict(G, ckp_G['state_dict'])
    # ckp_D = torch.load(path_D)
    # load_my_state_dict(D, ckp_D['state_dict'])

    if model_name_T.startswith("VGG16"):
        T = VGG16(1000)
        E = FaceNet(1000)

    # T = torch.nn.DataParallel(T).cuda()
    T= T.cuda()
    ckp_T = torch.load(path_T)
    E = torch.nn.DataParallel(E).cuda()
    ckp_E = torch.load(path_E)

    if 0:
        print("Pre-trained model_latest.pth (ckp_E) state_dict:")
        n = 0
        for k,v in ckp_E['state_dict'].items():
            print ('idx = %d' %n, k, v.shape)
            n += 1
        
        #NOTE: added by CCJ:
        # Print model's state_dict
        print("\n\nModel VGG16 state_dict:")
        n = 0
        for param_tensor in T.state_dict():
            print('idx = ', n, "\t", param_tensor, "\t", T.state_dict()[param_tensor].size())
            n += 1
            if 'module.' in param_tensor:
                tmp_k = param_tensor[len('module.'):]
            else:
                tmp_k = param_tensor
            if tmp_k not in ckp_T['state_dict']:
                print ("not found:", tmp_k)

    T.load_state_dict(ckp_T['state_dict'], strict=False)
    
    E.load_state_dict(ckp_E['state_dict'], strict=False)

    train_set, train_loader = init_dataloader(args, train_path, batch_size, mode="classify")
    print(train_path)
    val_set, val_loader = init_dataloader(args, val_path, batch_size, mode="classify")

    criterion = nn.CrossEntropyLoss().cuda()
    T.eval()
    E.eval()

    

    # print("---------------------Test [%s] accuracy------------------------------" % model_name)
    # # train set
    # for i, (imgs, one_hot, iden) in enumerate(train_loader):
    #     # iden = iden.view(-1).long().cuda()
    #     x = imgs.cuda()
    #     iden = iden.cuda()
    #     img_size = x.size(2)
    #     bs = x.size(0)
    #     # out = T(x)[-1]
    #     out = E(low2high(x))[-1]

    #     eval_iden = torch.argmax(out, dim=1).view(-1)
    #     train_acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
    #     loss = criterion(out, iden)

    #     print("training acc:", train_acc)

    # test set
    total_acc = 0
    for i, (imgs, one_hot, iden) in enumerate(val_loader):
        # iden = iden.view(-1).long().cuda()
        x = imgs.cuda()
        iden = iden.cuda()
        img_size = x.size(2)
        bs = x.size(0)
        # out = T(x)[-1]
        out = E(low2high(x))[-1]

        eval_iden = torch.argmax(out, dim=1).view(-1)
        val_acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
        total_acc += val_acc
        loss = criterion(out, iden)
        print("val acc:", val_acc)

    aver_acc = total_acc / (i+1)