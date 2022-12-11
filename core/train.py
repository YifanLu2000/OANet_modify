import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test import valid
from loss import MatchLoss
from utils import tocuda
from tensorboardX import SummaryWriter

def train_step(step, optimizer, model, match_loss, data):
    model.train()
    # print(data['xs_reg'].shape)
    # exit(0)
    res_logits, res_e_hat, res_motion_hat = model(data,data['xs_reg'][:,:,:,0:2])
    loss = 0
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    loss_reg, loss_reg_self, loss_reg_quary = match_loss.GPR_reg_loss(step, data, res_motion_hat)
    loss += loss_reg
    loss_val_reg = [loss_reg, loss_reg_self, loss_reg_quary]
    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
        if torch.any(torch.isnan(param.grad)):
            print('skip because nan')
            return loss_val, loss_val_reg

    optimizer.step()
    return loss_val, loss_val_reg


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    match_loss = MatchLoss(config)
    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    if not os.path.exists(os.path.join(config.log_base,config.log_suffix,'train','log_file')):
        os.mkdir(os.path.join(config.log_base,config.log_suffix,'train','log_file'))
    writer=SummaryWriter(os.path.join(config.log_base,config.log_suffix,'train','log_file'))
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        # logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)
    else:
        best_fscore = -1
        start_epoch = 0
        # logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan')
        # logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss', 'Reg Loss']*(config.iter_num+1))
        # logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        # logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss', 'Reg Loss'])
    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        # loss_vals, loss_GPR_vals = train_step(step, optimizer, model, match_loss, train_data)
        loss_val, loss_val_reg = train_step(step, optimizer, model, match_loss, train_data)
        # logger_train.append([cur_lr] + loss_vals)
        if step % config.log_intv == 0:
            writer.add_scalar('lr', cur_lr, step)
            writer.add_scalar('Geo Loss', loss_vals[0], step)
            writer.add_scalar('Clasfi Loss', loss_vals[1], step)
            writer.add_scalar('L2 Loss', loss_vals[2], step)
            writer.add_scalar('Reg Loss', loss_val_reg[0], step)
            writer.add_scalar('Reg self Loss', loss_val_reg[1], step)
            writer.add_scalar('Reg quary Loss', loss_val_reg[2], step)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss, reg_loss, reg_self_loss, reg_quary_loss, prec, reca, fscore  = valid(valid_loader, model, step, config)
            # logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])
            writer.add_scalar('lr', cur_lr, step)
            writer.add_scalar('val_ClassifyLoss', cla_loss, step)
            writer.add_scalar('val_Geo_loss', geo_loss, step)
            writer.add_scalar('val_RegressionLoss', l2_loss, step)
            writer.add_scalar('val_GPR_RegressionLoss', reg_loss, step)
            writer.add_scalar('val_GPR_self_RegressionLoss', reg_self_loss, step)
            writer.add_scalar('val_GPR_quary_RegressionLoss', reg_quary_loss, step)
            writer.add_scalar('val_acc', va_res, step)
            writer.add_scalar('val_fscore', fscore, step)   
            if fscore > best_fscore:
                print("Saving best model with va_res = {} and fscore".format(va_res,fscore))
                best_fscore = fscore
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_fscore': best_fscore,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_fscore': best_fscore,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

