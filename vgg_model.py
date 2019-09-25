import torch

import torch.nn as nn
from loss import TVloss, StyleLoss, StructureLoss
import copy
from histogram_loss import HistogramLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_style_model_and_losses(cnn,
                               target_texture_img,
                               style_layers, structure_layers):

    histogram_layers = ['pool_4','pool_12']

    input_channel = target_texture_img.size()[1]

    cnn = copy.deepcopy(cnn)

    output_channel, _, kernel_size, __ = cnn[0].weight.size()

    revised_cnn_weight = torch.zeros(output_channel, input_channel,
                                     cnn[0].kernel_size[0], cnn[0].kernel_size[1]).to(device)

    for ich in range(revised_cnn_weight.size()[1]):
        if ich < 3:
            revised_cnn_weight[:,ich,:,:] = cnn[0].weight.data[:,ich,:,:]
        else:
            t = torch.randint(3,(3,)).long()
            a = torch.rand(1).tolist()[0]
            b = (1-a)*torch.rand(1).tolist()[0]
            c = 1-a-b
            revised_cnn_weight[:,ich,:,:] = a*cnn[0].weight.data[:,t[0],:,:] +\
                                            b*cnn[0].weight.data[:, t[1], :, :] +\
                                            c*cnn[0].weight.data[:, t[2], :, :]

    new_first_layer = nn.Conv2d(input_channel,cnn[0].out_channels,
                       kernel_size=cnn[0].kernel_size,padding=0,
                       dilation=cnn[0].dilation,groups=cnn[0].groups)

    reflect_pad = nn.ReflectionPad2d((1,1,cnn[0].padding[0],cnn[0].padding[0]))

    new_first_layer.weight.data = revised_cnn_weight
    new_first_layer.bias.data = cnn[0].bias
    cnn[0] = new_first_layer
    # just in order to have an iterable access to or list of content/syle
    # losses
    structure_losses = []
    histogram_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # model = nn.Sequential(normalization)
    model = nn.Sequential()
    model.add_module("reflect_pad",reflect_pad)
    tv_loss = TVloss()
    model.add_module("tv_loss", tv_loss)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            target_feature = model(target_texture_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        if name in structure_layers:
            target_feature = model(target_texture_img).detach()
            structure_loss = StructureLoss(target_feature)
            model.add_module("structure_loss_{}".format(i), structure_loss)
            structure_losses.append(structure_loss)

        if name in histogram_layers:
            target_feature = model(target_texture_img).detach()
            histogram_loss = HistogramLoss(target_feature)
            model.add_module("histogram_loss_{}".format(i), histogram_loss)
            histogram_losses.append(histogram_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], HistogramLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, structure_losses, tv_loss, histogram_losses
