import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad, Variable
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from styleGAN import StyleDiscriminator, StyleGenerator
import torch.optim as optim


def freeze(net):
	for p in net.parameters():
		p.requires_grad_(False) 

def unfreeze(net):
	for p in net.parameters():
		p.requires_grad_(True)

def R1Penalty(real_img, f):
	# gradient penalty
	reals = Variable(real_img, requires_grad=True).to(real_img.device)
	real_logit = f(reals)
	apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
	undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

	real_logit = apply_loss_scaling(torch.sum(real_logit))
	real_grads = grad(real_logit, reals, grad_outputs=torch.ones(real_logit.size()).to(reals.device), create_graph=True)[0].view(reals.size(0), -1)
	real_grads = undo_loss_scaling(real_grads)
	r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
	return r1_penalty

save_img_dir = "/home/sichen/models/result/imgs_celeba_stylegan"
save_model_dir= "/home/sichen/models/styleGAN"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"

log_path = "./attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "styleGAN.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')

if __name__ == "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
	
	file = "./config/" + dataset_name + ".json"
	args = load_json(json_file=file)

	file_path = args['dataset']['gan_file_path']
	model_name = args['dataset']['model_name']
	lr=0.00001
	batch_size = 4
	z_dim = args[model_name]['z_dim']
	epochs = 200
	n_critic = args[model_name]['n_critic']

	print("---------------------Training [%s]------------------------------" % model_name)
	utils.print_params(args["dataset"], args[model_name])

	dataset, dataloader = init_dataloader(args, file_path, batch_size, mode="gan")

	G = StyleGenerator()
	D = StyleDiscriminator()
	
	G = torch.nn.DataParallel(G).cuda()
	D = torch.nn.DataParallel(D).cuda()

	optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
	optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
	scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
	scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)
	softplus = nn.Softplus()

	step = 0

	manualSeed = 999
	#manualSeed = random.randint(1, 10000) # use if you want new results
	print("Random Seed: ", manualSeed)
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)

	for epoch in range(epochs):
		start = time.time()
		for i, imgs in enumerate(dataloader):
			
			step += 1
			imgs = imgs.cuda()
			bs = imgs.size(0)
			
			freeze(G)
			unfreeze(D)

			z = torch.randn(bs, z_dim).contiguous().cuda()
			f_imgs = G(z)

			r_logit = D(imgs)
			f_logit = D(f_imgs)
			
			d_loss = softplus(f_logit).mean()
			d_loss = d_loss + softplus(-r_logit).mean()

			# gradient penalty
			r1_penalty = R1Penalty(imgs.contiguous().detach(), D)
			d_loss = d_loss + r1_penalty * (10 * 0.5)
			
			optim_D.zero_grad()
			d_loss.backward()
			optim_D.step()

			# train G

			if step % n_critic == 0:
				freeze(D)
				unfreeze(G)
				z = torch.randn(bs, z_dim).cuda()
				f_imgs = G(z)
				f_logit = D(f_imgs)
				# calculate g_loss
				g_loss = softplus(-f_logit).mean()
				
				optim_G.zero_grad()
				g_loss.backward()
				optim_G.step()

		end = time.time()
		interval = end - start
		
		print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))
		if (epoch+1) % 10 == 0:
			z = torch.randn(32, z_dim).cuda()
			fake_image = G(z)
			save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}_cross_style.png".format(epoch)), nrow = 8)
		
		torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "celeba_G_cross_style.tar"))
		torch.save({'state_dict':D.state_dict()}, os.path.join(save_model_dir, "celeba_D_cross_style.tar"))

		scheduler_D.step()
		scheduler_G.step()

