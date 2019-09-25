import torch
import torch.nn as nn
from neighbors_finder import get_neighbors_fixed_field_fast
import numpy as np

from torch.autograd import Function
from torch.autograd import Variable
import torch.optim as optim
from datetime import datetime
import os
from utils.poisson_disk import generate_possion_dis
from data_importer import normalization


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')

class DiffFunc(Function):

    @staticmethod
    def forward(ctx, input, target_points, neighborhood_size=0.2, res=128, kernel_sigma=2):


        target,_ = d_func(target_points, neighborhood_size=neighborhood_size,
               res=res, kernel_sigma=kernel_sigma)
        df_input, np = d_func(input, neighborhood_size=neighborhood_size,
               res=res, kernel_sigma=kernel_sigma)

        ctx.np = np
        ctx.neighborhood_size = neighborhood_size
        ctx.res = res
        ctx.kernel_sigma = kernel_sigma

        ctx.save_for_backward(input,  target_points, target, df_input)

        output = (target - df_input).pow(2).sum()
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, target_points, target, df_input = ctx.saved_tensors

        kernel_size = 6 * ctx.kernel_sigma + 1

        d_gaussian_kernel_x = d_kernel(ctx.kernel_sigma,res=ctx.res,neighborhood_size=ctx.neighborhood_size)[:,:,0]
        d_gaussian_kernel_x = d_gaussian_kernel_x.view(1, 1, kernel_size, kernel_size)
        gaussian_filter_x = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=kernel_size,padding=kernel_size//2, groups=1, bias=False).to(device).float()
        gaussian_filter_x.weight.requires_grad = False
        gaussian_filter_x.weight.data = d_gaussian_kernel_x.to(device).float()

        d_gaussian_kernel_y = d_kernel(ctx.kernel_sigma,res=ctx.res,neighborhood_size=ctx.neighborhood_size)[:,:,1]
        d_gaussian_kernel_y = d_gaussian_kernel_y.view(1, 1, kernel_size, kernel_size)
        gaussian_filter_y = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=kernel_size,padding=kernel_size//2, groups=1, bias=False).to(device).float()
        gaussian_filter_y.weight.requires_grad = False
        gaussian_filter_y.weight.data = d_gaussian_kernel_y.to(device).float()


        dis_x = gaussian_filter_x(df_input - target)
        dis_y = gaussian_filter_y(df_input - target)

        output_x = torch.zeros(input.size()[0])
        output_y = torch.zeros(input.size()[0])

        for i in range(input.size()[0]):

            neighbors_index = ctx.np[i]

            if len(neighbors_index)!=0:

                indices = torch.floor((input_points[neighbors_index, :] - input_points[i, :] + ctx.neighborhood_size) / (
                        2 * ctx.neighborhood_size) * ctx.res).reshape(-1, 2)

                x = dis_x[0,0,indices[:, 0].long(), indices[:, 1].long()]
                y = dis_y[0,0,indices[:, 0].long(), indices[:, 1].long()]

                output_x[i] = 4*x.sum()/input.size()[0]
                output_y[i] = 4*y.sum()/input.size()[0]

        import pdb
        pdb.set_trace()
        grad_input = torch.cat((output_x.unsqueeze(1),output_y.unsqueeze(1)),1)

        import pdb
        pdb.set_trace()

        return grad_output * grad_input, None, None, None, None


# used in calculting derivatives
def d_kernel(kernel_sigma):

    kernel_size = round(6 * kernel_sigma + 1)
    kernel_size = kernel_size+1 if (6 * kernel_sigma + 1)%2 == 0 else kernel_size
    mean = (kernel_size - 1) / 2

    x_cord = torch.arange(kernel_size).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    variance_space = kernel_sigma**2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)

    d_gaussian_kernel = (2*((xy_grid-mean))/variance_space) * \
                      torch.exp(
                          -torch.sum(((xy_grid - mean)) ** 2., dim=-1) / \
                          (variance_space)
                      ).repeat(2,1,1).permute(1,2,0)

    return d_gaussian_kernel


def gaussian_kernel(kernel_sigma):

    kernel_size = round(6 * kernel_sigma + 1)
    mean = (kernel_size - 1) / 2

    x_cord = torch.arange(kernel_size).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    variance = (kernel_sigma) ** 2.


    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)

    gaussian_kernel = torch.exp(
        -torch.sum(((xy_grid - mean)) ** 2., dim=-1) / \
        (variance)
    )

    return gaussian_kernel,kernel_size



class diff_func(Function):

    @staticmethod
    def forward(ctx, input, neighborhood_size=0.1, res=128, kernel_sigma=2):


        ctx.neighborhood_size = neighborhood_size
        ctx.res = res
        ctx.kernel_sigma = kernel_sigma
        ctx.num_points = input.size()[0]
        ctx.input = input

        hist = torch.zeros(res, res).to(device)
        np = get_neighbors_fixed_field_fast(input, 2, neighborhood_size)

        kernel_size = 6 * kernel_sigma + 1
        mean = (kernel_size - 1) / 2

        x_cord = torch.arange(kernel_size).float()
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        variance = kernel_sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = torch.exp(
            -torch.sum(((xy_grid - mean)) ** 2., dim=-1) / \
            (variance)
        )

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=kernel_size, padding=kernel_size // 2, groups=1, bias=False).to(device)

        gaussian_filter.weight.data = gaussian_kernel.to(device).float()
        gaussian_filter.weight.requires_grad = False

        for i in range(input.size()[0]):
            neighbors_index = np[i]
            neighbors_index.remove(i)
            if len(neighbors_index) != 0:
                indices = torch.floor((input[neighbors_index, :] - input[i, :] + neighborhood_size) / (
                        2 * neighborhood_size) * res).reshape(-1, 2)
                hist[indices[:, 0].long(), indices[:, 1].long()] += 1

        ctx.np = np
        output = gaussian_filter(hist.reshape(1, 1, res, -1)) / input.size()[0]

        return output

    @staticmethod
    def backward(ctx, gradoutput):

        kernel_size = 6 * ctx.kernel_sigma + 1

        d_gaussian_kernel_x = d_kernel(ctx.kernel_sigma)[:,:,1]
        d_gaussian_kernel_x = d_gaussian_kernel_x.view(1, 1, kernel_size, kernel_size)
        gaussian_filter_x = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=kernel_size,padding=kernel_size//2, groups=1, bias=False).to(device).float()
        gaussian_filter_x.weight.requires_grad = False
        gaussian_filter_x.weight.data = d_gaussian_kernel_x.to(device).float()

        d_gaussian_kernel_y = d_kernel(ctx.kernel_sigma)[:,:,0]
        d_gaussian_kernel_y = d_gaussian_kernel_y.view(1, 1, kernel_size, kernel_size)
        gaussian_filter_y = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=kernel_size,padding=kernel_size//2, groups=1, bias=False).to(device).float()
        gaussian_filter_y.weight.requires_grad = False
        gaussian_filter_y.weight.data = d_gaussian_kernel_y.to(device).float()


        gradinput_x = torch.zeros(ctx.num_points, ctx.res, ctx.res)
        gradinput_y = torch.zeros(ctx.num_points, ctx.res, ctx.res)

        for i in range(ctx.num_points):
            hist = torch.zeros(ctx.res, ctx.res).to(device)
            neighbors_index = ctx.np[i]

            if len(neighbors_index) == 0:
                continue

            indices_forward = torch.floor((ctx.input [neighbors_index, :] - ctx.input [i, :] + ctx.neighborhood_size) / (
                        2 * ctx.neighborhood_size) * ctx.res).reshape(-1, 2)
            indices_backward = torch.floor((ctx.input [i, :] - ctx.input [neighbors_index, :] + ctx.neighborhood_size) / (
                        2 * ctx.neighborhood_size) * ctx.res).reshape(-1, 2)


            hist[indices_forward[:, 0].long(), indices_forward[:, 1].long()] += 1
            hist[indices_backward[:, 0].long(), indices_backward[:, 1].long()] += (-1)


            gradinput_xi = gaussian_filter_x(hist.reshape(1, 1, ctx.res, ctx.res))
            gradinput_yi = gaussian_filter_y(hist.reshape(1, 1, ctx.res, ctx.res))

            gradinput_x[i, :, :] = gradinput_xi.squeeze()
            gradinput_y[i, :, :] = gradinput_yi.squeeze()



        gradinput = torch.cat((gradinput_x.unsqueeze(1),gradinput_y.unsqueeze(1)), 1).to(device)
        gradinput = gradinput.reshape(ctx.num_points*2,ctx.res**2).to(device)

        return torch.mm(gradinput,gradoutput.reshape(-1,1)).reshape(ctx.num_points,2), None, None, None



def d_func(input_points, neighborhood_size=0.1, res=128, kernel_sigma=2):

    hist = torch.zeros(res, res).to(device)
    np = get_neighbors_fixed_field_fast(input_points, 2, neighborhood_size)
    kernel_size = 6 * kernel_sigma + 1
    mean = (kernel_size - 1) / 2

    x_cord = torch.arange(kernel_size).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    variance = kernel_sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = torch.exp(
                          -torch.sum(((xy_grid - mean)) ** 2., dim=-1) / \
                          (variance)
                      )

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                     kernel_size=kernel_size,padding=kernel_size//2, groups=1, bias=False).to(device)

    gaussian_filter.weight.data = gaussian_kernel.to(device).float()
    gaussian_filter.weight.requires_grad = False

    for i in range(input_points.size()[0]):
        neighbors_index = np[i]
        neighbors_index.remove(i)
        if len(neighbors_index) != 0:
            indices = torch.floor((input_points[neighbors_index, :] - input_points[i, :] + neighborhood_size) / (
                        2 * neighborhood_size) * res).reshape(-1, 2)
            hist[indices[:, 0].long(), indices[:, 1].long()] += 1

    out = gaussian_filter(hist.reshape(1, 1, res, -1))/input_points.size()[0]

    return out, np


if __name__ == "__main__":

    #test the module

    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(datestr)

    num_steps = 10000

    x = np.linspace(0, 0.99, 20)
    y = np.linspace(0, 0.99, 20)
    xv, yv = np.meshgrid(x, y)

    xc = torch.from_numpy(xv.reshape((-1,1)))
    yc = torch.from_numpy(yv.reshape((-1,1)))

    target_points = Variable(torch.cat((xc,yc),1).to(device).float(),requires_grad=False)
    normalization(target_points, edge=0.01)
    np.savetxt(datestr + '/target.txt', target_points.cpu().data.numpy())
    tmp = torch.from_numpy(generate_possion_dis(1600))

    input_points = Variable(tmp.float().to(
        device=device),requires_grad=True)

    optimizer = optim.Adam([input_points.requires_grad_()])


    run = [0]
    while run[0] <= num_steps:

        def closure():

            input_points.data.clamp_(0, 1)

            optimizer.zero_grad()
            df_input = diff_func.apply(input_points, 0.4/2)
            df_target = diff_func.apply(target_points, 0.4)

            loss = (df_input-df_target).pow(2).sum()


            loss.backward()

            run[0] += 1
            if run[0] % 5 == 0:
                print("run {}:".format(run))
                print('Loss : {:4f}'.format(loss.item(), 0))

                np.savetxt(datestr + '/out' + str(run[0]) + '.txt', input_points.cpu().data.numpy())

                print()

            return loss


        optimizer.step(closure)

    input_points.data.clamp_(0, 1)