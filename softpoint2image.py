import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')
class SoftPoint2Image(nn.Module):

    def __init__(self, d_s, d_f, kernel_sigma = 0.005, feature_sigma = 0.02, res = 384, spatial_varying_pcf = False):
        '''
        :param d_s: spatial dimension
        :param d_f: feature dimension
        :param kernel_sigma: density estimation, Gaussian kernel's standard deviatio relative to domain size
        :param spatial_varying_pcf: whether consider spatially
        varying pair correlation function. Default is false
        :param res: image resolution = res*res
        :return:
        '''

        super(SoftPoint2Image, self).__init__()
        self.d_s = d_s
        self.d_f = d_f
        self.kernel_sigma = kernel_sigma
        self.feature_sigma = feature_sigma
        self.res = res
        self.spatial_varying_pcf = spatial_varying_pcf
        ksize = round(res * kernel_sigma * 6)
        sigma = kernel_sigma * res

        kernel = np.fromfunction(lambda x, y:  np.e ** (
                    (-1 * ((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)) / (2 * sigma ** 2)), (ksize, ksize))
        self.kernel = torch.from_numpy(kernel)
        self.kernel = torch.from_numpy(kernel).float()
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        self.ksize = ksize

        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        xv, yv = np.meshgrid(x, y)
        txv = torch.from_numpy(xv).unsqueeze(0).float()
        tyv = torch.from_numpy(yv).unsqueeze(0).float()
        self.mesh = torch.cat((txv, tyv), 0).to(device=device)

    def forward(self, p):

        img = torch.zeros((self.d_f + 1), self.res, self.res).to(device)

        for i in range(p.size()[0]):

            center = p[i,:self.d_s]
            hw = round(3*self.kernel_sigma*self.res)
            coor_center = torch.floor(center * self.res)
            up = torch.max(coor_center[1] - hw, torch.Tensor([0]).to(device))
            down = torch.min(coor_center[1] + hw + 1, torch.Tensor([self.res]).to(device))
            left = torch.max(coor_center[0] - hw,torch.Tensor([0]).to(device))
            right = torch.min(coor_center[0] + hw + 1, torch.Tensor([self.res]).to(device))
            up = up.long()
            down = down.long()
            left = left.long()
            right = right.long()


            img[0, up:down, left:right] += p[i,2]*torch.exp(-(self.mesh.permute(1,2,0)[up:down, left:right, :]
                                                   - center).pow(2).sum(2)/(2*self.kernel_sigma**2))

        return img.unsqueeze(0)



