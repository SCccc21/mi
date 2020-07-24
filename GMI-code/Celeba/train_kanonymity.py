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
from attack import inversion, inversion_grad_constraint
from kanonymity import inversion_k, inversion_grad_constraint_k
from generator import Generator
from sklearn.model_selection import GridSearchCV


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

def cal_mean(least_seed_need):
    bs = least_seed_need.shape[0]
    fail = 0
    num = 0
    for b in range(bs):
        if least_seed_need[b] == 1001:
            fail += 1
            continue
        num += least_seed_need[b]

    return num / (bs - fail), fail


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'

    global args, logger
    logger = get_logger()
    model_name_T = "VGG16"
    model_name_E = "FaceNet"
    dataset_name = "celeba"
    improved_flag = False

    file = "./config/attack" + ".json"
    args = load_json(json_file=file)
    logger.info(args)
    print("Using improved GAN:", improved_flag)
   
    z_dim = 100

    # path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_0715.tar'
    # path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_0715.tar'
    # path_G = '/home/sichen/models/improvedGAN/improved_celeba_G_0719.tar'
    # path_D = '/home/sichen/models/improvedGAN/improved_celeba_D_0719.tar'
    
    path_G = '/home/sichen/models/GAN/celeba_G.tar'
    path_D = '/home/sichen/models/GAN/celeba_D.tar'
    path_T = '/home/sichen/models/target_model/target_ckp/VGG16_88.26.tar'
    path_E = '/home/sichen/models/target_model/target_ckp/FaceNet_95.88.tar'

    ###########################################
    ###########     load model       ##########
    ###########################################
    # no mask
    G = Generator(z_dim)
    G = torch.nn.DataParallel(G).cuda()
    if improved_flag == True:
        # D = Discriminator(3, 64, 1000)
        D = MinibatchDiscriminator()
    else:
        D = DGWGAN(3)
    
    D = torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=False)

    if model_name_T.startswith("VGG16"):
        T = VGG16(1000)
    elif model_name_T.startswith('IR152'):
        T = IR152(1000)
    elif model_name_T == "FaceNet64":
        T = FaceNet64(1000)

    
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).cuda()
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)

    # with mask

    ###########     load identity    ##########
    # batch_size = 64
    # file_path = args['dataset']['attack_file_path']
    # data_set, data_loader = init_dataloader(args, file_path, batch_size, mode="classify")

    ############         attack     ###########
    logger.info("=> Begin attacking ...")

    '''
    # mask
    for idx, (imgs, one_hot, iden) in enumerate(data_loader):
        print("--------------------- Attack batch [%s]------------------------------" % idx)
        inversion(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1)
        # inversion_grad_constraint(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1)
    '''



    total_acc, total_acc5 = 0, 0
    num_seed = 0
    num_fail = 0
    # no auxilary
    for i in range(1):
        iden = torch.from_numpy(np.arange(60))

        for idx in range(5):
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            least_seed_need = inversion_k(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=1000, iter_times=1500, clip_range=1, improved=improved_flag)
            # least_seed_need = inversion_grad_constraint_k(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, lamda2=15, iter_times=1500, clip_range=1)
            print("least_seed_need of batch %s \n" % idx, least_seed_need)
            mean_seed, fail = cal_mean(least_seed_need)
            num_seed += mean_seed
            num_fail += fail
            print("average number of seeds is:", mean_seed)
            print("number of failure is:", fail)
            iden += 60

    num_seed /= 5   
    num_fail /= 5 

    logger.info("=> Attack finished.")
    print("average number of seeds need is:", num_seed)
    print("average number of failure is:", num_fail)

    