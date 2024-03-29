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

def get_acc(G, D, T, E, lamda, lamda2):
    total_acc, total_acc5 = 0, 0
    # no auxilary
    for i in range(3):
        iden = torch.from_numpy(np.arange(60))

        for idx in range(5):
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            acc, acc5 = inversion_grad_constraint(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=lamda, lamda2=lamda2, iter_times=1500, clip_range=1, improved=False)
            iden = iden + 60
            total_acc += acc
            total_acc5 += acc5

    aver_acc = total_acc / 15
    aver_acc5 = total_acc5 / 15
    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}".format(aver_acc, aver_acc5))
    
    return aver_acc, aver_acc5



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

    
    ################ param search #############

    dict_acc = {}
    dict_acc5 = {}
    best_acc, best_acc5 = 0, 0
    

    # lamda_list = [10, 50, 100, 150, 200, 500] # iden loss
    # lamda1_list = [0.1, 1] # prior loss
    # lamda2_list = [1, 10, 50, 100, 500] # grad loss

    lamda_list = [100] # iden loss
    lamda2_list = [8, 125, 200] # grad loss

    
    for lamda in lamda_list:
        for lamda2 in lamda2_list:

            aver_acc, aver_acc5 = get_acc(G, D, T, E, lamda, lamda2)
            
            params = 'lamda1=' + str(1) + ' lamda=' + str(lamda) + ' lamda2=' + str(lamda2)
            print(params)
            
            dict_acc[params] = aver_acc
            dict_acc5[params] = aver_acc5

            if aver_acc > best_acc:
                best_acc = aver_acc
                best_params = params
            if aver_acc5 > best_acc5:
                best_acc5 = aver_acc5
                best_params_5 = params

    # print(dict_acc)

    filename = open('./gc_search_acc.txt','w')#dict转txt
    for k,v in dict_acc.items():
        filename.write(k+':\t'+str(v))
        filename.write('\n')
    filename.close()

    filename = open('./gc_search_acc5.txt','w')#dict转txt
    for k,v in dict_acc5.items():
        filename.write(k+':\t'+str(v))
        filename.write('\n')
    filename.close()

    print("Best acc: " + str(best_acc) + "\tparams are: " + best_params)
    print("Best acc5: " + str(best_acc5) + "\tparams are: " + best_params_5)

    


    

    