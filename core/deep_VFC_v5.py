import torch
import torch.nn as nn
from loss import batch_episym

class GridPosition(nn.Module):
    def __init__(self, grid_num, use_gpu = True):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.use_gpu = use_gpu

    def forward(self, batch_size):
        grid_center_x = torch.linspace(-1.+2./self.grid_num/2,1.-2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(-1.+1./self.grid_num/2,1.-1./self.grid_num/2,steps=self.grid_num)
        grid_center_y = torch.linspace(1.-2./self.grid_num/2,-1.+2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(1.-1./self.grid_num/2,-1.+1./self.grid_num/2,steps=self.grid_num)
        # BCHW, (b,:,h,w)->(x,y)
        grid_center_position_mat = torch.reshape(
            torch.cartesian_prod(grid_center_x, grid_center_y),
            (1, self.grid_num, self.grid_num, 2)
        ).permute(0,3,2,1)
        # BCN, (b,:,n)->(x,y), left to right then up to down
        grid_center_position_seq = grid_center_position_mat.reshape(1, 2, self.grid_num*self.grid_num)
        return grid_center_position_seq.repeat(batch_size, 1, 1)

class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new

class ResBlock(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.net(x) + x
        return x

class Filter(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.resnet = nn.Sequential(*[ResBlock(channels) for _ in range(3)])
        self.scale =  nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
        )

    def forward(self, x):
        x = self.scale(self.resnet(x))
        return x

class FilterNet(nn.Module):
    def __init__(self, grid_num, channels):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.filter = Filter(channels)

    def forward(self, x):
        # BCN -> BCHW
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num, self.grid_num)
        x = self.filter(x)
        # BCHW -> BCN
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num*self.grid_num)
        return x

class PositionEncoder(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.position_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.position_encoder(x)

class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)

class ConvMatchLayerBlock(nn.Module):
    def __init__(self, channels, head, grid_num):
        nn.Module.__init__(self)
        self.align = AttentionPropagation(channels, head)
        self.filter = FilterNet(grid_num, channels)
        self.dealign = AttentionPropagation(channels, head)
        self.inlier_pre = InlinerPredictor(channels)

    def forward(self, xs, d, grid_pos_embed):
        # xs: B1N4
        grid_d = self.align(grid_pos_embed, d)
        grid_d = self.filter(grid_d)
        d_new = self.dealign(d, grid_d)
        # BCN -> B1N -> BN
        logits = torch.squeeze(self.inlier_pre(d_new - d), 1)
        e_hat = weighted_8points(xs, logits)
        return d_new, logits, e_hat, grid_d

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, config, T = torch.tensor(0.2, dtype=torch.float32, device='cuda')):
        super().__init__()
        self.learn_temperature = config.learn_temperature
        if config.learn_temperature:
            self.T = nn.Parameter(T)
        else:
            self.T = T

    def forward(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K

class Exact_GPR_solver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sigma2 = nn.Parameter(torch.tensor(10, dtype=torch.float32, device='cuda'))
    
    def forward(self, Knn, P, motion):
        P = torch.maximum(P, torch.tensor(1e-6))
        L,_ = torch.linalg.cholesky_ex(Knn + torch.diag_embed(torch.div(self.sigma2,P)))
        C = torch.cholesky_solve(motion, L)
        # C = torch.linalg.solve(
        #         torch.multiply(P,Knn)+torch.diag_embed(self.sigma2),
        #         torch.multiply(P,motion)
        #     )
        # motion_hat = torch.matmul(Knn,C).transpose(1,2)

        # SigmaKmn = torch.linalg.solve(Kmm+torch.matmul(Knm.transpose(1,2),(torch.mul(torch.unsqueeze(P,-1),Knm)))/self.sigma2,Knm.transpose(1,2))
        # C = torch.matmul(SigmaKmn,(torch.mul(torch.unsqueeze(P,-1),motion)))/self.sigma2
        # motion_hat = torch.matmul(Knm,C).transpose(1,2)
        # variance = torch.sum(torch.mul(Knm,SigmaKmn.transpose(1,2)),-1)
        return C

class P_pred(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.error_embed = PositionEncoder(channels)
        self.inlier_pre = InlinerPredictor(channels)
        self.scale = nn.Parameter(torch.tensor(10, dtype=torch.float32, device='cuda'))
    def forward(self,d, motion_error, variance, eps=1e-7):
        P_motion_error = torch.sigmoid(torch.squeeze(self.inlier_pre(d+self.error_embed(motion_error)),1))
        P_variance = torch.exp(-variance / self.scale)
        P = torch.maximum(torch.mul(P_motion_error,P_variance),torch.tensor(eps))
        return P





class deep_VFC(nn.Module):
    def __init__(self, config, use_gpu=True):
        nn.Module.__init__(self)
        self.layer_num = config.layer_num
        self.GPR_num = config.GPR_num
        self.grid_center = GridPosition(config.grid_num, use_gpu=use_gpu)
        self.pos_embed = PositionEncoder(config.net_channels)
        self.grid_pos_embed = PositionEncoder(config.net_channels)
        self.init_project = PositionEncoder(config.net_channels)
        self.conv_match_layer_blocks = nn.Sequential(
            *[ConvMatchLayerBlock(config.net_channels, config.head, config.grid_num) for _ in range(self.layer_num)]
        )
        self.GPR_solver = Exact_GPR_solver(config)
        self.inlier_pre = InlinerPredictor(config.net_channels)
        self.kernel_pos_embed = PositionEncoder(config.net_channels)
        self.Cos_kernel = CosKernel(config)
        self.error_embed = PositionEncoder(config.net_channels)
        self.topK = config.topK
        self.deep_kernel = AttentionPropagation(config.net_channels, config.head)
    
    def forward(self, data, quary_x):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # B1NC -> BCN
        input = data['xs'].transpose(1,3).squeeze(3)
        x1, x2 = input[:,:2,:], input[:,2:,:]
        motion = x2 - x1

        pos = x1 # B2N
        grid_pos = self.grid_center(batch_size) # B2N

        pos_embed = self.pos_embed(pos) # BCN
        grid_pos_embed = self.grid_pos_embed(grid_pos)

        d = self.init_project(motion) + pos_embed # BCN

        res_logits, res_e_hat = [], []
        for i in range(self.layer_num):
            d, logits, e_hat, grid_d = self.conv_match_layer_blocks[i](data['xs'], d, grid_pos_embed) # BCN
            res_logits.append(logits), res_e_hat.append(e_hat)
        # Gaussian Process Regression Optimization
        # B(C+2)N
        deep_kernel_embed = torch.cat([pos,self.deep_kernel(self.kernel_pos_embed(pos),grid_d)],axis=1)
        deep_kernel_embed_quary = torch.cat([quary_x,self.deep_kernel(self.kernel_pos_embed(quary_x),grid_d)],axis=1)
        P = torch.sigmoid(logits)
        Knn = self.Cos_kernel(deep_kernel_embed.transpose(1,2),deep_kernel_embed.transpose(1,2))
        Kqn = self.Cos_kernel(deep_kernel_embed_quary.transpose(1,2),deep_kernel_embed.transpose(1,2))
        C = self.GPR_solver(Knn,P,motion.transpose(1,2))
        motion_hat = torch.matmul(Knn,C).transpose(1,2)
        motion_quary_hat = torch.matmul(Kqn,C).transpose(1,2)
        res_motion_hat = [motion_hat, motion_quary_hat]
        logits = torch.squeeze(self.inlier_pre(d+self.error_embed(motion_hat-motion)),1)
        e_hat = weighted_8points(data['xs'], logits)
        res_logits.append(logits), res_e_hat.append(e_hat)
        
        return res_logits, res_e_hat, res_motion_hat
        

        
        



def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        # e,v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO="U")
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat







