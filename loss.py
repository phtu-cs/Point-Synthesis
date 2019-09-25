import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = (G-self.target).pow(2).sum()
        return input

class StructureLoss(nn.Module):
    def __init__(self, target_feature):
        super(StructureLoss, self).__init__()

        self.q = target_feature.size()[2]
        self.m = target_feature.size()[3]
        self.target = deep_corr_matrix(target_feature, self.q, self.m).detach()
        assert(self.q == self.m)

    def forward(self, input):
        G = deep_corr_matrix(input, self.q, self.m)
        self.loss = F.mse_loss(G, self.target)
        return input

    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class TVloss(nn.Module):
    def __init__(self):
        super(TVloss,self).__init__()

    def forward(self, input):
        batch_size, c, h, w = input.size()
        self.loss = (torch.pow(input[:, :, 1:, :]-input[:, :, :h-1, :], 2).sum(1).sum(1).sum(1) + \
            torch.pow(input[:, :, :, 1:] - input[:, :, :, :w-1], 2).sum(1).sum(1).sum(1))
        self.output = input.clone()
        return self.output
    def backward(self, retain_graph = True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def deep_corr_matrix(input,q,m):

    a, b, c, d = input.size()  # a=batch size(=1)
    assert (c == d)
    assert (q == m)
    R = torch.zeros(b, 2*c-1, 2*d-1).to(device)


    x_cord = torch.arange(1, c + 1).float().to(device)
    _x_cord = torch.arange(c-1, 0, -1).float().to(device)
    x_cord = torch.cat((x_cord,_x_cord),0)

    x_grid = x_cord.repeat(2*c-1).view(2*c-1, 2*c-1)
    y_grid = x_grid.t()


    xconv2 = nn.Conv2d(in_channels=1, out_channels=1,
                                     kernel_size=c,padding=c-1, groups=1, bias=False).to(device)

    center = (2*c-1)//2.0


    for i in range(b):
        xconv2.weight.data = input[:, i, :, :].unsqueeze(1)
        xconv2.weight.requires_grad = False
        corr_matrix = xconv2(input[:,i,:,:].unsqueeze(1))
        corr_matrix/=(x_grid*y_grid)
        R[i, :, :] = corr_matrix.squeeze(0).squeeze(0)
        if q%2 == 1:
            out = R[:, int(center - c // 2 + math.ceil((c-q)/2.0)):int(center + math.ceil(c / 2) - 1 - (c-q)//2.0 + 1),
              int(center - c // 2 + math.ceil((c - q) / 2.0)):int(center + math.ceil(c / 2) - 1 - (c - q) // 2.0 + 1)]
        else:
            out = R[:, int(center - c // 2 + math.floor((c-q)/2.0)):int(center + math.ceil(c / 2) - 1 - math.ceil((c-q)/2.0) + 1),
                  int(center - c // 2 + math.floor((c - q) / 2.0)):int(
                      center + math.ceil(c / 2) - 1 - math.ceil((c - q) / 2.0) + 1)]

    return out



class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input



class HomoLoss(nn.Module):

    def __init__(self):
        super(HomoLoss, self).__init__()

    def forward(self, input):

        tmp = torch.zeros(input.size()).to(device)

        tmp[input < 0.5] = input[input < 0.5].pow(2)
        tmp[input >= 0.5] = (input[input >= 0.5] - 1 ).pow(2)
        self.loss = tmp.sum()
        return self.loss



if __name__ == "__main__":

    input = Variable(torch.randn(1, 3, 7, 7).to(device),requires_grad=True)
    R = deep_corr_matrix(input,7,7)
    import matplotlib.pyplot as plt
    plt.imshow(R.data[0,:,:])


