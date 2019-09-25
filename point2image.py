import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
from diff_func import gaussian_kernel, d_kernel
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')
class Point2Image(nn.Module):

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

        super(Point2Image, self).__init__()
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

        img = torch.zeros((self.d_f + 1), int(self.res), int(self.res)).to(device)
        out = torch.zeros((self.d_f + 1), int(self.res), int(self.res)).to(device)

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



            for j in [t for t in range(p.size()[1]) if t!=1]:
                if j == 0:
                    img[j, up:down, left:right] += torch.exp(-(self.mesh.permute(1,2,0)[up:down, left:right, :]
                                                   - center).pow(2).sum(2)/(2*self.kernel_sigma**2))
                else:
                    img[j-1, upf:downf, leftf:rightf] += p[i,j]*torch.exp(-(self.mesh.permute(1,2,0)[upf:downf, leftf:rightf, :]
                                                    - center).pow(2).sum(2)/(2*self.feature_sigma**2))


        return img.unsqueeze(0)

class Point2Image_fast(Function):

    @staticmethod
    def forward(ctx, input, kernel_sigma, res):
        '''
        :param d_s: spatial dimension
        :param d_f: feature dimension
        :param kernel_sigma: density estimation, Gaussian kernel's standard deviatio relative to domain size
        :param spatial_varying_pcf: whether consider spatially
        varying pair correlation function. Default is false
        :param res: image resolution = res*res
        :return:
        '''

        center = input[:, :2]
        coor_center = torch.floor(center * (res-1))
        np_center = coor_center.cpu().numpy()
        img = torch.from_numpy(np.histogram2d(np_center[:,1],np_center[:,0],res)[0]).to(device).float()
        img = img.unsqueeze(0).unsqueeze(0)

        kernel, kernel_size = gaussian_kernel(kernel_sigma*res)

        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(1, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=kernel_size, padding=kernel_size // 2, groups=1, bias=False).to(device)

        gaussian_filter.weight.data = kernel.to(device).float()
        gaussian_filter.weight.requires_grad = False
        output = gaussian_filter(img)

        ctx.img=img
        ctx.kernel_sigma = kernel_sigma
        ctx.res=res
        ctx.num_points=input.size()[0]
        ctx.input = input

        return output


    @staticmethod
    def backward(ctx, gradoutput):

        img=ctx.img
        res=ctx.res
        kernel_sigma=ctx.kernel_sigma

        d_kernels = d_kernel(ctx.kernel_sigma*res)
        kernel_size = d_kernels.size()[1]

        d_gaussian_kernel_x = -d_kernels[:,:,0]
        d_gaussian_kernel_x = d_gaussian_kernel_x.view(1, 1, kernel_size, kernel_size)
        d_gaussian_kernel_x = d_gaussian_kernel_x.repeat(ctx.num_points, 1, 1, 1)
        gaussian_filter_x = nn.Conv2d(in_channels=ctx.num_points, out_channels=ctx.num_points,
                                    kernel_size=kernel_size,padding=kernel_size//2, groups=ctx.num_points, bias=False).to(device).float()
        gaussian_filter_x.weight.requires_grad = False
        gaussian_filter_x.weight.data = d_gaussian_kernel_x.to(device).float()

        d_gaussian_kernel_y = -d_kernels[:,:,1]
        d_gaussian_kernel_y = d_gaussian_kernel_y.view(1, 1, kernel_size, kernel_size)
        d_gaussian_kernel_y = d_gaussian_kernel_y.repeat(ctx.num_points,1,1,1)
        gaussian_filter_y = nn.Conv2d(in_channels=ctx.num_points, out_channels=ctx.num_points,
                                    kernel_size=kernel_size,padding=kernel_size//2, groups=ctx.num_points, bias=False).to(device).float()
        gaussian_filter_y.weight.requires_grad = False
        gaussian_filter_y.weight.data = d_gaussian_kernel_y.to(device).float()


        gradinput = torch.zeros(ctx.num_points, ctx.res, ctx.res).to(device)


        center = ctx.input[:, :2]
        coor_center = torch.floor(center * (res - 1))
        gradinput[list(range(ctx.num_points)), coor_center[:,1].long(), coor_center[:, 0].long()] = 1

        gradinput_x = gaussian_filter_x(gradinput.reshape(1, ctx.num_points, ctx.res, ctx.res))
        gradinput_y = gaussian_filter_y(gradinput.reshape(1, ctx.num_points, ctx.res, ctx.res))


        gradinput = torch.cat((gradinput_x.squeeze(0),gradinput_y.squeeze(0)), 1).to(device)

        gradinput = gradinput.reshape(ctx.num_points*2,ctx.res**2).to(device)


        return torch.mm(gradinput,gradoutput.reshape(-1,1)).reshape(ctx.num_points,2), None, None, None, None



if __name__ == '__main__':

    x = np.linspace(0, 0.99, 5)
    y = np.linspace(0, 0.99, 5)
    xv, yv = np.meshgrid(x, y)

    xc = torch.from_numpy(xv.reshape((-1,1))).to(device).float()
    yc = torch.from_numpy(yv.reshape((-1,1))).to(device).float()

    f1 = torch.rand(xc.size()).to(device)
    f2 = torch.rand(xc.size()).to(device)
    p = torch.cat((xc,yc,f1,f2),1).to(device).float()
    p = Variable(p, requires_grad=True)

    pp = p.clone()
    pp[13,1]+=0.008

    density = Point2Image_fast.apply(p,  0.05, 128)
    density2 = Point2Image_fast.apply(pp, 0.05, 128)


    a = density.sum()
    a.backward()
    grad = (density2-density)/0.008

    plt.figure(1)
    plt.imshow(grad.squeeze().data.cpu())
    plt.figure(2)
    plt.imshow(density2.squeeze().data.cpu())
    plt.figure(3)
    plt.imshow(density.squeeze().data.cpu())

