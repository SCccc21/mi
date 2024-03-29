from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def save_center(I):
	feat_path = './feat/center.npy'
	img_folder = '/home/sichen/mi/GMI-code/Celeba/fid/private_domain_new/'
	center = torch.zeros(300, 512)
	idx = 900

	for i in range(300):
		aver_feat = torch.zeros(512)
		x = torch.zeros(12, 3, 64, 64)
		for j in range(12):
			name = img_folder + str(idx+j) + 'th_iden_' + str(i+1) + '.png'
			image = Image.open(name)
			image = TF.to_tensor(image)
			x[j, :, :, :] = torch.unsqueeze(image, 0)
		# import pdb; pdb.set_trace()
		print(i)
		feat, _ = I(low2high(x))
		center[i, :] = torch.mean(feat.detach().cpu(), dim=0)
		idx += 27
		torch.cuda.empty_cache()

	np.save("./feat/center.npy", center.numpy())
	print("Success!")

def save_knn(I):
	feat_path = './feat/feat.npy'
	info_path = "./feat/info.npy"
	img_folder = '/home/sichen/mi/GMI-code/Celeba/fid/private_domain_new/'

	knn = torch.zeros(300*12, 512) # pick 12 images for each of the 300 identities
	info = torch.zeros(300*12)
	idx = 900
	for i in range(300):
		x = torch.zeros(12, 3, 64, 64)
		
		for j in range(12):
			name = img_folder + str(idx+j) + 'th_iden_' + str(i+1) + '.png'
			image = Image.open(name)
			image = TF.to_tensor(image)
			x[j, :, :, :] = torch.unsqueeze(image, 0)
			info[12*i+j] = i
		import pdb; pdb.set_trace()
		print(i)
		feat, _ = I(low2high(x))
		knn[12*i:12*(i+1), :] = feat.detach().cpu()
		idx += 27

	# np.save(feat_path, knn.numpy())
	# np.save(info_path, info.numpy())
	print("Success!")


def get_center_dist(base_path, I):
	listOfFile = os.listdir(base_path)

	img_list = []
	label_list = []
	for entry in listOfFile:
		_, _, label, _ = os.path.splitext(entry)[0].strip().split('_')

		img = Image.open(base_path+entry)
		img = TF.to_tensor(img)
		img_list.append(img)
		label_list.append(int(label)-1)
		

		# if int(label) in [1,2,3,4,5,6,7,8,9,10]:
		# 	img = Image.open(base_path+entry)
		# 	img = TF.to_tensor(img)
		# 	img_list.append(img)
		# 	label_list.append(int(label)-1)

	# import pdb; pdb.set_trace()
	image = torch.stack(img_list, dim=0)
	print(image.shape)
	iden = torch.LongTensor(label_list)
	feat, _ = I(low2high(image))
	dist = calc_center(feat.detach(), iden, path='./feat')
	return dist, feat.detach()

def get_knn_dist(base_path, I):
	listOfFile = os.listdir(base_path)

	img_list = []
	label_list = []
	for entry in listOfFile:
		idx, _, _, label, seed = os.path.splitext(entry)[0].strip().split('_')
		if int(idx) == 0 and int(label) < 100: 
			img = Image.open(base_path+entry)
			img = TF.to_tensor(img)
			img_list.append(img)
			label_list.append(int(label)-1)
	
	image = torch.stack(img_list, dim=0)
	# print(image.shape)
	iden = torch.LongTensor(label_list)
	feat, _ = I(low2high(image))
	dist = calc_knn(feat.detach(), iden, path='./feat')

	return dist, feat.detach()

def get_knn_dist2(base_path, I):
	listOfFile = os.listdir(base_path) # all attack imgs

	img_list = []
	label_list = []
	for entry in listOfFile:
		_, _, label, seed = os.path.splitext(entry)[0].strip().split('_')
		if int(label) < 100: 
		# if 100 <= int(label) < 200: 
		# if 200 <= int(label) < 299: 
			img = Image.open(base_path+entry)
			img = TF.to_tensor(img)
			img_list.append(img)
			label_list.append(int(label)-1)
	
	image = torch.stack(img_list, dim=0)
	# print(image.shape)
	iden = torch.LongTensor(label_list)
	feat, _ = I(low2high(image))
	dist = calc_knn(feat.detach(), iden, path='./feat')

	return dist, feat.detach()

def save_public_knn(I):
	feat_path = './feat/feat_public.npy'
	info_path = "./feat/public_idx.npy"
	img_folder = '/home/sichen/mi/GMI-code/Celeba/fid/private_domain_new/'

	f_public = open("/home/sichen/data/ganset.txt", "r")
	knn = torch.zeros(30000, 512)
	info = torch.zeros(30000)
	idx = 0
	for line in datafile.readlines():
		img_name = line.strip()
		img = Image.open("/home/sichen/data/img_align_celeba_png/" + img_name)
		img = TF.to_tensor(img)
		feat, _ = I(low2high(img))
		knn[idx, :] = feat.detach().cpu()
		import pdb; pdb.set_trace()
		info[idx] = int(os.path.splitext(img_name)[0]) # save image name (index only)
		idx += 1

	np.save(feat_path, knn.numpy())
	np.save(info_path, info.numpy())
	print("Success!")


#NOTE
def knn_neighbor(attack_path, I):
	feat_public = torch.from_numpy(np.load("./feat/feat_public.npy")).float()
	info = torch.from_numpy(np.load("./feat/public_idx.npy")).view(-1).long()

	listOfFile = os.listdir(attack_path) # attack img path
	img_list = []
	for entry in listOfFile:
		idx, _, _, label, seed = os.path.splitext(entry)[0].strip().split('_')
		img = Image.open(attack_path+entry)
		print(entry)
		img = TF.to_tensor(img)
		img_list.append(img)

	print("start to process...")
	image = torch.stack(img_list, dim=0)
	feat, _ = I(low2high(image))
	feat = feat.cpu()
	bs = feat.size(0)
	tot = feat_public.size(0)
	min_knn = torch.zeros(bs)
	for i in range(bs):
		knn = 1e8
		for j in range(tot):
			dist = torch.sum((feat[i, :] - feat_public[j, :]) ** 2)
			if dist < knn:
				knn = dist
				min_knn[i] = info[j]
		print("Nearest neighbor of {} is public image{}".format(i, min_knn[i]))
	


def psnr(img1, img2):
	mse = torch.mean((img1 - img2) ** 2)
	return 20 * torch.log10(255.0 / torch.sqrt(mse))

def acc_attri300(attack_path):
	att_path = './attribute/300attri.txt'
	att_list = open(att_path).readlines() 
	listOfFile = os.listdir(attack_path)
	#load model
	model_path = './attribute/model_checkpoint_nopretrain.pth'
	
	model = MobileNet()
	model = torch.nn.DataParallel(model).to('cuda')
	ckp_E = torch.load(model_path)
	model.load_state_dict(ckp_E['model_state_dict'], strict=False)
	model.eval()

	correct = np.zeros(40)
	cnt = 0
	for entry in listOfFile:
		if entry.endswith('.png'):
			cnt += 1
			# img
			img = Image.open(attack_path+entry)
			img = TF.to_tensor(img)

			# label
			idx, _, _, label, seed = os.path.splitext(entry)[0].strip().split('_')
			iden = int(label) - 1
			data_label = att_list[iden].split()
			data_label = data_label[2:]
			data_label = [int(p) for p in data_label]

			img = img.cuda()
			data_label = data_label.cuda()
			output = model(low2high(img))
			result = output > 0.5
			for i in range(40):
				correct[i] += (result[i] == data_label[i]).item()
			
	acc = correct / cnt
	print("Attribute acc:", 100. * acc)
	return acc

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

	# I = FaceNet(1000)
	# ckp = torch.load('/home/sichen/models/target_model/backbone_ir50_ms1m_epoch120.pth')
	# load_state_dict(I.feature, ckp)
	# I = torch.nn.DataParallel(I).cuda()
	# I.eval()
	
	# print("Loading Backbone Checkpoint ")
	# # I.load_state_dict(ckp['state_dict'], strict=False)
	# # save_center(I)
	# save_knn(I)
	'''
	path = './feat/origin/'
	center_dist, feat_ori = get_knn_dist(path, I)
	print("origin center dist:", center_dist)

	path = './feat/mb/'
	center_dist , feat_mb = get_knn_dist(path, I)
	print("mb center dist:", center_dist)

	# print(feat_ori - feat_mb)

	path = './feat/mb_h/'
	center_dist, _ = get_knn_dist(path, I)
	print("mb+h center dist:", center_dist)

	path = './feat/dist_mb/'
	center_dist, _ = get_knn_dist(path, I)
	print("mb+dist center dist:", center_dist)

	path = './feat/dist_mb_h/'
	center_dist, _ = get_knn_dist(path, I)
	print("mb+dist+h center dist:", center_dist)
	'''

	# path = './fid/fid_origin/'
	# dist, _ = get_knn_dist(path, I) 
	# print("origin knn:", dist)
	
	# path = './fid/fid_dist_entropy/'
	# dist, _ = get_knn_dist(path, I) 
	# print("our knn:", dist)
	acc_attri300('./fid/fid_dist_entropy/')