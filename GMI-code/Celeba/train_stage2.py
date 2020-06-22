from losses import completion_network_loss, noise_loss
from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import time
import random
import os, logging
import numpy as np
from attack import inversion
from generator import Generator

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



if __name__ == "__main__":

    global args, logger
    logger = get_logger()
    model_name = "VGG16"
    dataset_name = "celeba"

    file = "./config/attack" + ".json"
    args = load_params(json_file=file)
    logger.info(args)
   
    z_dim = 100

    path_G = '/home/sichen/mi/GMI-code/Celeba/result/models_celeba_gan/celeba_G.tar'
    path_D = '/home/sichen/mi/GMI-code/Celeba/result/models_celeba_gan/celeba_D.tar'
    path_T = '/home/sichen/mi/GMI-code/Celeba/target_model/' + model_name + '/model_best.pth'
    path_E = '/home/sichen/mi/GMI-code/Celeba/target_model/backbone_ir50_ms1m_epoch120.pth'

    ###########################################
    ###########     load model       ##########
    ###########################################
    # no mask
    G = Generator(z_dim)
    torch.nn.DataParallel(G).cuda()
    D = DGWGAN(3)
    torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    load_my_state_dict(G, ckp_G['state_dict'])
    ckp_D = torch.load(path_D)
    load_my_state_dict(D, ckp_D['state_dict'])

    if model_name.startswith("VGG16"):
        T = VGG16(1000)
        E = FaceNet(1000)
    elif model_name.startswith('IR152'):
        T = IR152(1000)
    elif model_name == "FaceNet64":
        T = FaceNet64(1000)

    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    load_my_state_dict(T, ckp_T['state_dict'])
    E = torch.nn.DataParallel(E).cuda()
    ckp_E = torch.load(path_E)
    # import pdb; pdb.set_trace()
    load_module_state_dict(E, ckp_E, add="module.feature.")

    # with mask

    ###########################################
    ###########     load iden       ##########
    ###########################################
    batch_size = 64
    file_path = args['dataset']['train_file_path']
    data_set, data_loader = init_dataloader(args, file_path, batch_size, mode="classify")
    
    

    logger.info("=> Begin attacking ...")

    ###########################################
    ############         attack     ###########
    ###########################################
    for idx, (imgs, one_hot, label) in enumerate(data_loader):
        print("--------------------- Attack batch [%s]------------------------------" % idx)
        inversion(G, D, T, E, label, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1)

    
    