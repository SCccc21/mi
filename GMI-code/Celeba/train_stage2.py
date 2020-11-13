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
	# os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
	os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'

	global args, logger
	logger = get_logger()
	model_name_T = "IR152"
	model_name_E = "FaceNet"
	dataset_name = "celeba"
	improved_flag = True

	file = "./config/attack" + ".json"
	args = load_json(json_file=file)
	logger.info(args)
	print("Using improved GAN:", improved_flag)
   
	z_dim = 100

	# path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_0715.tar'
	# path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_0715.tar'
	
	# path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_entropy2.tar'
	# path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_entropy2.tar'
	
	# path_G = '/home/sichen/models/GAN/celeba_G.tar'
	# path_D = '/home/sichen/models/GAN/celeba_D.tar'
	# path_G = '/home/sichen/models/GAN/celeba_G_scrub.tar'
	# path_D = '/home/sichen/models/GAN/celeba_D_scrub.tar'

	#NOTE: IR152
	path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_IR152_entropy.tar'
	path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_IR152_entropy.tar'
	#NOTE: Facenet64
	# path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_facenet_entropy_55.tar'
	# path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_facenet_entropy_55.tar'

	
	# path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_ffhq_entropy.tar'
	# path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_ffhq_entropy.tar'
	# path_G = '/home/sichen/models/improvedGAN/improved_mb_celeba_G_facenet_entropy_scrub_48.tar'
	# path_D = '/home/sichen/models/improvedGAN/improved_mb_celeba_D_facenet_entropy_scrub_48.tar'
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
		path_T = '/home/sichen/models/target_model/target_ckp/VGG16_88.26.tar'
	elif model_name_T.startswith('IR152'):
		T = IR152(1000)
		path_T = '/home/sichen/models/target_model/target_ckp/IR152_91.16.tar'
	elif model_name_T == "FaceNet64":
		T = FaceNet64(1000)
		path_T = '/home/sichen/models/target_model/target_ckp/FaceNet64_88.50.tar'

	
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


	aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
	# no auxilary
	for i in range(3):
		iden = torch.from_numpy(np.arange(60))

		for idx in range(5):
			print("--------------------- Attack batch [%s]------------------------------" % idx)
			# acc, acc5, var, var5 = inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag)
			acc, acc5, var, var5 = dist_inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag, num_seeds=5)
			iden = iden + 60
			aver_acc += acc / 15
			aver_acc5 += acc5 / 15
			aver_var += var / 15
			aver_var5 += var5 / 15

	print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc, aver_acc5, aver_var, aver_var5))

	