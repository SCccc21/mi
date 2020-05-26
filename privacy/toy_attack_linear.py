import torch
import numpy as np
import logging
import models, os
from work import *
from torch.autograd import Variable


# [cyp2c9, vkorc1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
target_str = "vkorc1"
data_folder = 'data'
save_path = './checkpoint'

# [reg, vib]
model_name = 'reg'
att_epochs = 1000
att_lr = 2e-1
eps=0.4
t_val_min=-1
t_val_max=1

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
    target_model.eval()

    # attack
    train_x = torch.from_numpy(trans_norm(x)).float().cuda()
    train_label = torch.from_numpy(y).float().cuda().view(-1, 1)
    # train_label = Variable(train_label)
    train_t = torch.from_numpy(t).float().cuda().view(-1, 1)
    train_t = trans_project_t(train_t)
    t_adv = Variable(train_t, requires_grad=True)
    x_adv = merge_t(t_adv, train_x, target_cols)
    x_adv = Variable(x_adv)

    for e in range(att_epochs):
        tmp1 = torch.inverse(x_adv.t().mm(x_adv))
        tmp2 = x_adv.t().mm(train_label)
        c_bar = tmp1.mm(tmp2)
        h_adv = x_adv.mm(c_bar)
        cost = torch.mean((train_label - h_adv) ** 2)
        target_out = target_model(x_adv)
        # target_out = Variable(target_out)
        true_cost = torch.mean((train_label - target_out) ** 2)
        loss = torch.mean(true_cost - cost).abs()
        # import pdb; pdb.set_trace()

        if t_adv.grad is not None:
            t_adv.grad.data.fill_(0)
        t_adv.retain_grad()
        loss.backward()

        t_adv.grad.sign_()
        t_adv = t_adv - eps*t_adv.grad
        t_adv = where(t_adv > t+eps, t+eps, t_adv)
        t_adv = where(t_adv < t-eps, t-eps, t_adv)
        t_adv = torch.clamp(t_adv, t_val_min, t_val_max)
        x_adv = merge_t(t_adv.data, x_adv, target_cols)

        # attack acc
        num_correct = np.count_nonzero(x_adv == t)
        num_rows = X.shape[0]
        attack_acc = num_correct / num_rows
        
        print("Attack Acc:{:.2f} ".format(attack_acc))

    logger.info("=> Attack Finished.")
    print("Final Attack Acc:{:.2f}".format(attack_acc))


if __name__ == "__main__":
    main()