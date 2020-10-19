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
from attack import inversion, inversion_grad_constraint, natural_grad
from dist_attack import dist_inversion
from generator import Generator


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
	os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
	# os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'

	global args, logger
	logger = get_logger()
	model_name_T = "VGG16"
	dataset_name = "celeba"
	improved_flag = False

	print("Using improved GAN:", improved_flag)
   
	z_dim = 100

	# path_G = './Attack/improvedGAN/improved_mb_cifar_G_entropy.tar'
	# path_D = './Attack/improvedGAN/improved_mb_cifar_D_entropy.tar'
	
	path_G = '/home/sichen/models/GAN/cifar_G.tar'
	path_D = '/home/sichen/models/GAN/cifar_D.tar'

	
	# path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_ffhq_entropy.tar'
	# path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_ffhq_entropy.tar'
	

	###########################################
	###########     load model       ##########
	###########################################
	# no mask
	G = GeneratorCIFAR(z_dim)
	G = torch.nn.DataParallel(G).cuda()
	if improved_flag == True:
		D = MinibatchDiscriminator_CIFAR()
	else:
		D = DGWGAN32(3)
	
	D = torch.nn.DataParallel(D).cuda()
	ckp_G = torch.load(path_G)
	G.load_state_dict(ckp_G['state_dict'], strict=False)
	ckp_D = torch.load(path_D)
	D.load_state_dict(ckp_D['state_dict'], strict=False)

	T = VGG16(5)
	path_T = './Attack/target_model/target_ckp/VGG16_92.15.tar'
	path_E = './Attack/target_model/target_ckp/VGG19_91.89.tar'
	T = torch.nn.DataParallel(T).cuda()
	ckp_T = torch.load(path_T)
	T.load_state_dict(ckp_T['state_dict'], strict=False)

	E = VGG19()
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


	aver_acc, aver_acc5, aver_var = 0, 0, 0

	for idx in range(5):
		iden = torch.from_numpy(np.arange(5))
		print("--------------------- Attack batch [%s]------------------------------" % idx)
		acc, acc5, var = inversion(G, D, T, E, iden, itr=idx, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag)
		# acc, acc5, var = dist_inversion(G, D, T, E, iden, itr=idx, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag, num_seeds=5)
		
		aver_acc += acc / 5
		aver_acc5 += acc5 / 5
		aver_var += var / 5

	print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}".format(aver_acc, aver_acc5, aver_var))

	