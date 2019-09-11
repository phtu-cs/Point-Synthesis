import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from math import floor,ceil


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
        #(1 / (2 * np.pi * sigma ** 2)) *
        kernel = np.fromfunction(lambda x, y:  np.e ** (
                    (-1 * ((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)) / (2 * sigma ** 2)), (ksize, ksize))
        self.kernel = torch.from_numpy(kernel)
        #kernel /=np.sum(kernel)
        self.kernel = torch.from_numpy(kernel).float()
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        self.ksize = ksize

        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        xv, yv = np.meshgrid(x, y)
        # mx = torch.from_numpy(xv.reshape((-1, 1)))
        # my = torch.from_numpy(yv.reshape((-1, 1)))
        txv = torch.from_numpy(xv).unsqueeze(0).float()
        tyv = torch.from_numpy(yv).unsqueeze(0).float()
        self.mesh = torch.cat((txv, tyv), 0).to(device=device)

    def forward(self, p):
        # res_img = []
        # index = []
        # for i in range(self.d_s):
        #     res_img.append(self.res)
        #     index.append(torch.floor(p[:, i] * self.res).long())
        #
        #
        # img = torch.zeros(res_img)

        # img[index] = 1
        # img = img.unsqueeze(0).unsqueeze(0)
        # density = F.conv2d(img, self.kernel, padding= floor(self.ksize/2)).repeat(1,3,1,1)
        # return density


        img = torch.zeros((self.d_f + 1), self.res, self.res).to(device)
        out = torch.zeros((self.d_f + 1), self.res, self.res).to(device)

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

            hwf = round(3*self.feature_sigma*self.res)
            upf = torch.max(coor_center[1] - hwf, torch.Tensor([0]).to(device))
            downf = torch.min(coor_center[1] + hwf + 1, torch.Tensor([self.res]).to(device))
            leftf = torch.max(coor_center[0] - hwf,torch.Tensor([0]).to(device))
            rightf = torch.min(coor_center[0] + hwf + 1, torch.Tensor([self.res]).to(device))
            upf = upf.long()
            downf = downf.long()
            leftf = leftf.long()
            rightf = rightf.long()




            img[0, up:down, left:right] += p[i,2]*torch.exp(-(self.mesh.permute(1,2,0)[up:down, left:right, :]
                                                   - center).pow(2).sum(2)/(2*self.kernel_sigma**2))

        return img.unsqueeze(0)


if __name__ == '__main__':
    x = np.linspace(0, 0.99, 50)
    y = np.linspace(0, 0.99, 50)
    xv, yv = np.meshgrid(x, y)

    xc = torch.from_numpy(xv.reshape((-1,1))).to(device).float()
    yc = torch.from_numpy(yv.reshape((-1,1))).to(device).float()

    f1 = torch.rand(xc.size()).to(device)
    f2 = torch.rand(xc.size()).to(device)
    p = torch.cat((xc,yc,f1,f2),1).to(device).float()

    #p = torch.rand(4096,2)


    p2i = Point2Image(2,2)

    density = p2i(p)
    plt.figure(1)
    plt.imshow(density[:,0,:,:].squeeze(0).cpu().numpy())
    plt.figure(2)
    plt.imshow(density[:, 2, :, :].squeeze(0).cpu().numpy())
    #plt.imshow(Point2Image(2,0).kernel.numpy())

