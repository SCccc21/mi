import torch
import numpy as np
import logging
import models, os
from work import *
from torch.autograd import Variable, grad


# [cyp2c9, vkorc1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
target_str = "vkorc1"
data_folder = 'data'
save_path = './checkpoint'

# [reg, vib]
model_name = 'reg'
att_epochs = 100
att_lr = 2e-1
t_val_min = 0
t_val_max = 1 / 8.07
initial = 0.5 / 8.07 

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    # [0, 9, 16]
    global logger
    logger = get_logger()

    logger.info("=> load data ...")
    x, y, featnames = load_iwpc(data_folder)
    y = trans_project(y)

    t, target_cols = extract_target(x, target_str, featnames)

    target_model = models.MLP(input_dim=x.shape[1]).cuda()
    ckpt_name = './checkpoint/model_latest.pth'
    if os.path.isfile(ckpt_name):
        checkpoint = torch.load(ckpt_name)
        target_model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded target model checkpoint '{}'".format(ckpt_name))
    else:
        logger.info("=> no checkpoint found at '{}'".format(ckpt_name))

    logger.info("=> begin attacking ...")
    # target_model.eval()

    # attack
    est_x = torch.from_numpy(trans_norm(x)).float().cuda()
    init = torch.ones_like(est_x[:,target_cols]).float().cuda()
    init = init / (2*8.07) 
    est_x[:, target_cols] = init
    train_label = torch.from_numpy(y).float().cuda().view(-1, 1)
    # x_adv = Variable(est_x, requires_grad=True)
    x_adv = est_x
    print("initial x_adv:", x_adv[:, target_cols])
    x_adv.requires_grad = True
    # import pdb; pdb.set_trace()
    
    Ipp = torch.eye(x_adv.shape[1]).float().cuda()
    lam = 0.1

    for e in range(att_epochs):
        x_adv.requires_grad = True
        target_out = target_model(x_adv)
        loss1 = torch.mean((train_label - target_out) ** 2)
        loss1.backward(retain_graph=True)
        loss2 = 0
        for param in target_model.parameters():
            loss2 += param.grad
            # print(param)
        
        loss = loss1 + lam * torch.sum(loss2.abs())
        # import pdb; pdb.set_trace()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        x_adv.retain_grad()
        loss.backward()
        print("grad:", x_adv.grad[:, target_cols])
        # x_adv.grad.sign_()
        x_adv.detach_()
        x_adv[:, target_cols] = x_adv[:, target_cols] - att_lr*x_adv.grad[:, target_cols]

        # x_adv[:, target_cols] = where(x_adv[:, target_cols] > est_x[:, target_cols]+eps, est_x[:, target_cols]+eps, x_adv[:, target_cols])
        # x_adv[:, target_cols] = where(x_adv[:, target_cols] < est_x[:, target_cols]-eps, est_x[:, target_cols]-eps, x_adv[:, target_cols])
        x_adv[:, target_cols] = torch.clamp(x_adv[:, target_cols], t_val_min, t_val_max)

        # attack acc
        pred_t, attack_acc = get_result(x_adv, t, target_cols)
        print("prediction:", pred_t)
        
        
        print("Epoch:{}\t loss:{}\t Attack Acc:{:.2f} ".format(e, loss, attack_acc))

    logger.info("=> Attack Finished.")
    print("Final Attack Acc:{:.2f}".format(attack_acc))


if __name__ == "__main__":
    main()