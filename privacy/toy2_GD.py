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
eps = 1e-2
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

    train_x, test_x, train_y, test_y, train_t, test_t = train_test_split(x, y, t,
                                                        random_state=random_seed, test_size=0.25)

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
    x_adv = torch.from_numpy(trans_norm(train_x)).float().cuda()
    train_label = torch.from_numpy(train_y).float().cuda().view(-1, 1)
    init = torch.ones_like(x_adv[:,target_cols]).float().cuda()
    init = init / (8.07) 
    x_adv[:, target_cols] = init
    print("initial x_adv:", x_adv[:, target_cols])
    x_adv.requires_grad = True
    # import pdb; pdb.set_trace()
    
    Ipp = torch.eye(x_adv.shape[1]).float().cuda()
    lam = 0.01

    for e in range(att_epochs):
        x_adv.requires_grad = True
        tmp1 = torch.inverse(x_adv.t().mm(x_adv) + lam * Ipp)     # Ridge regression estimator, Hoerl 1970
        tmp2 = x_adv.t().mm(train_label)
        c_bar = tmp1.mm(tmp2)
        h_adv = x_adv.mm(c_bar)
        cost = torch.mean((train_label - h_adv) ** 2)
        target_out = target_model(x_adv)
        true_cost = torch.mean((train_label - target_out) ** 2)
        loss = torch.mean(true_cost - cost).abs()
        assert true_cost-cost>=0

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        x_adv.retain_grad()
        loss.backward()
        # x_adv.grad.sign_()
        x_adv.detach_()
        x_adv[:, target_cols] = x_adv[:, target_cols] - eps*x_adv.grad[:, target_cols]

        # x_adv[:, target_cols] = where(x_adv[:, target_cols] > est_x[:, target_cols]+eps, est_x[:, target_cols]+eps, x_adv[:, target_cols])
        # x_adv[:, target_cols] = where(x_adv[:, target_cols] < est_x[:, target_cols]-eps, est_x[:, target_cols]-eps, x_adv[:, target_cols])
        x_adv[:, target_cols] = torch.clamp(x_adv[:, target_cols], t_val_min, t_val_max)

        # attack acc
        pred_t, attack_acc = get_result(x_adv, train_t, target_cols)
        # print("new x_adv:", x_adv[:, target_cols])
        print("prediction:", pred_t)
        import pdb; pdb.set_trace()
        
        print("Epoch:{}\t loss:{}\t Attack Acc:{:.2f} ".format(e, loss, attack_acc * 100))

    logger.info("=> Attack Finished.")
    print("Final Attack Acc:{:.2f}".format(attack_acc  * 100))


if __name__ == "__main__":
    main()