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
from dist_attack import dist_inversion
from generator import *
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



if __name__ == "__main__":
	# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
	os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'

	global logger
	logger = get_logger()
	improved_flag = False

	
	print("Using improved GAN:", improved_flag)
   
	z_dim = 100
	
	path_G = './Attack/attack_models/cxr_G.tar'
	path_D = './Attack/attack_models/cxr_D.tar'

	# path_G = '/home/sichen/models/improvedGAN/improved_mb_cxr_G.tar'
	# path_D = '/home/sichen/models/improvedGAN/improved_mb_cxr_D.tar'

	path_T = './Attack/target_ckp/cxr_VGG16_target_45.13.tar'
	path_E = './Attack/target_ckp/cxr_VGG19_eval_44.62.tar'

	###########################################
	###########     load model       ##########
	###########################################
	# no mask
	G = GeneratorCXR(z_dim)
	G = torch.nn.DataParallel(G).cuda()
	if improved_flag == True:
		D = MinibatchDiscriminator(1, 64, 7)
	else:
		D = DGWGAN(1, 64)
	
	D = torch.nn.DataParallel(D).cuda()
	ckp_G = torch.load(path_G)
	G.load_state_dict(ckp_G['state_dict'], strict=False)
	ckp_D = torch.load(path_D)
	D.load_state_dict(ckp_D['state_dict'], strict=False)

	T = VGG16(7)
	T = torch.nn.DataParallel(T).cuda()
	ckp_T = torch.load(path_T)
	T.load_state_dict(ckp_T['state_dict'], strict=False)

	E = VGG19(7)
	E = torch.nn.DataParallel(E).cuda()
	ckp_E = torch.load(path_E)
	E.load_state_dict(ckp_E['state_dict'], strict=False)



	############         attack     ###########
	logger.info("=> Begin attacking ...")

	aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
	

	for i in range(5):
		print('-------------------------')
		iden = torch.from_numpy(np.arange(7))
		acc, acc5, var, var5 = inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag)
		# acc, acc5, var, var5 = dist_inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag)
		aver_acc += acc / 5
		aver_acc5 += acc5 / 5
		aver_var += var / 5
		aver_var5 += var5 / 5

	print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc, aver_acc5, aver_var, aver_var5))

