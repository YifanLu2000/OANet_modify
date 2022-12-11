import torch
from utils import torch_skew_symmetric
import numpy as np

def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))
    return ys

class MatchLoss(object):
    def __init__(self, config):
        self.loss_essential = config.loss_essential
        self.loss_classif = config.loss_classif
        self.use_fundamental = config.use_fundamental
        self.obj_geod_th = config.obj_geod_th
        self.geo_loss_margin = config.geo_loss_margin
        self.loss_essential_init_iter = config.loss_essential_init_iter

    def GPR_reg_loss(self, global_step, data, res_motion_hat):
        y_in, y_reg_in = data['ys'], data['ys_reg']
        gt_geod_d = y_in[:, :, 0]
        is_pos = (gt_geod_d < self.obj_geod_th).type(res_motion_hat[0].type())
        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        gt_geod_d_reg = y_reg_in[:, :, 0]
        is_pos_reg = (gt_geod_d_reg < self.obj_geod_th).type(res_motion_hat[0].type())
        num_pos_reg = torch.relu(torch.sum(is_pos_reg, dim=1) - 1.0) + 1.0

        # self reg loss
        x1, x2 = data['xs'][:,:,:,0:2], data['xs'][:,:,:,2:]
        motion = torch.squeeze(x2 - x1)  
        loss_reg_self = torch.mean(torch.sum(torch.sum(torch.abs(motion-res_motion_hat[0].transpose(1,2)),-1)*is_pos,-1) / num_pos)
        # quary reg loss
        x1_reg, x2_reg = data['xs_reg'][:,:,:,0:2], data['xs_reg'][:,:,:,2:]
        motion_reg = torch.squeeze(x2_reg - x1_reg)  
        loss_reg_quary = torch.mean(torch.sum(torch.sum(torch.abs(motion_reg-res_motion_hat[1].transpose(1,2)),-1)*is_pos_reg,-1) / num_pos_reg)

        loss_reg = 0
        # Check global_step and add loss
        if self.loss_GPR_reg > 0 :
            loss_reg += self.loss_GPR_reg * loss_reg_self 
        if self.loss_GPR_reg_quary > 0:
            loss_reg += self.loss_GPR_reg_quary * loss_reg_quary

        return [loss_reg, (self.loss_GPR_reg * loss_reg_self).item(), (self.loss_GPR_reg_quary * loss_reg_quary).item()]


    def run(self, global_step, data, logits, e_hat):
        R_in, t_in, y_in, pts_virt = data['Rs'], data['ts'], data['ys'], data['virtPts']

        # Get groundtruth Essential matrix
        e_gt_unnorm = torch.reshape(torch.matmul(
            torch.reshape(torch_skew_symmetric(t_in), (-1, 3, 3)),
            torch.reshape(R_in, (-1, 3, 3))
        ), (-1, 9))

        e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)

        ess_hat = e_hat
        if self.use_fundamental:
            ess_hat = torch.matmul(torch.matmul(data['T2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['T1s'])
            # get essential matrix from fundamental matrix
            ess_hat = torch.matmul(torch.matmul(data['K2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['K1s']).reshape(-1,9)
            ess_hat = ess_hat / torch.norm(ess_hat, dim=1, keepdim=True)


        # Essential/Fundamental matrix loss
        pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:,:,2:]
        geod = batch_episym(pts1_virts, pts2_virts, e_hat)
        essential_loss = torch.min(geod, self.geo_loss_margin*geod.new_ones(geod.shape))
        essential_loss = essential_loss.mean()
        # we do not use the l2 loss, just save the value for convenience 
        L2_loss = torch.mean(torch.min(
            torch.sum(torch.pow(ess_hat - e_gt, 2), dim=1),
            torch.sum(torch.pow(ess_hat + e_gt, 2), dim=1)
        ))


        # Classification loss
        # The groundtruth epi sqr
        gt_geod_d = y_in[:, :, 0]
        is_pos = (gt_geod_d < self.obj_geod_th).type(logits.type())
        is_neg = (gt_geod_d >= self.obj_geod_th).type(logits.type())
        c = is_pos - is_neg
        classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
        # balance
        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
        classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
        classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
        classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)


        precision = torch.mean(
            torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
            torch.sum((logits > 0).type(is_pos.type()) * (is_pos + is_neg), dim=1)
        )
        recall = torch.mean(
            torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
            torch.sum(is_pos, dim=1)
        )

        loss = 0
        # Check global_step and add essential loss
        if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
            loss += self.loss_essential * essential_loss 
        if self.loss_classif > 0:
            loss += self.loss_classif * classif_loss

        return [loss, (self.loss_essential * essential_loss).item(), (self.loss_classif * classif_loss).item(), L2_loss.item(), precision.item(), recall.item()]
