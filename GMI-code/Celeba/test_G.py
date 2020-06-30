import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from discri import DGWGAN
from generator import Generator

    
if __name__ == "__main__":
    global args
    file = "./config/celeba.json"
    args = load_params(json_file=file)

    file_path = args['dataset']['train_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']
    utils.print_params(args["dataset"], args[model_name])

    path_G = '/home/sichen/models/celeba_G.tar'
    path_D = '/home/sichen/models/celeba_D.tar'

    G = Generator(z_dim)
    G = torch.nn.DataParallel(G).cuda()
    D = DGWGAN(3)
    D = torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=False)
    G.eval()
    D.eval()

    if 1:
        print("Pre-trained celeba_D.tar (ckp) state_dict:")
        n = 0
        for k,v in ckp_D['state_dict'].items():
            print ('idx = %d' %n, k, v.shape)
            n += 1
        
        #NOTE: added by CCJ:
        # Print model's state_dict
        print("\n\nModel state_dict:")
        n = 0
        for param_tensor in D.state_dict():
            print('idx = ', n, "\t", param_tensor, "\t", D.state_dict()[param_tensor].size())
            n += 1
            tmp_k = param_tensor
            if tmp_k not in ckp_D['state_dict']:
                print ("not found:", tmp_k)

    
    dataset, dataloader = init_dataloader(args, file_path, batch_size, mode="gan")
    
    for i, imgs in enumerate(dataloader):
        imgs = imgs.cuda()
        bs = imgs.size(0)

        z = torch.randn(64, z_dim).cuda()
        f_imgs = G(z)
        logit_dg = D(f_imgs)
        # calculate g_loss
        g_loss = - logit_dg.mean()
        print(g_loss)
        if i == 10:
            save_tensor_images(f_imgs, "test_result.png", nrow = 8)
            print("image saved.")