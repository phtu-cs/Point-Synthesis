import numpy as np
import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import matlab.engine
from matplotlib import pyplot as plt
from scipy.misc import ascent, face

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# eng = matlab.engine.start_matlab()
class hist_match_torch(Function):

    @staticmethod
    def forward(ctx,input,target):

        ctx.input = input
        # a = matlab.double(input.squeeze(0).clone().cpu().numpy().tolist())
        # b = matlab.double(target.squeeze(0).clone().cpu().numpy().tolist())
        # matched = eng.imhistmatch(a, b)
        # np_matched = np.array(matched._data.tolist())
        # np_matched = np_matched.reshape(matched.size).transpose()
        #
        # output = torch.from_numpy(np_matched).to(device).float()

        #
        #
        a = input.clone().cpu().numpy()
        b = target.clone().cpu().numpy()

        matched = torch.from_numpy(hist_match(a,b)).to(device).float()
        # matched = input


        return matched

    @staticmethod
    def backward(ctx,grad_output):

        grad_input = torch.zeros(ctx.input.size()).to(device)

       # grad_input = grad_output;

        return grad_input, None

class HistogramLoss(nn.Module):

    def __init__(self, target):
        super(HistogramLoss, self).__init__()
        self.target = target

    def forward(self, input):
        hist_matched = hist_match_torch.apply(input, self.target)
        self.loss = (hist_matched-input).pow(2).sum()/input.numel()

        # self.loss = hist_matched.sum()
        # self.loss =
        self.output = input
        # del hist_matched
        # self.loss = torch.zeros(1).to(device)

        return self.output
    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)


    return interp_t_values[bin_idx].reshape(oldshape)




if __name__ == '__main__':


    source = torch.rand(1000,1)
    template = torch.cat((torch.zeros(200), 0.25 * torch.ones(200), 0.5 * torch.ones(200), 0.75 * torch.ones(200), torch.ones(200)),
                  0)

    # a = matlab.double(source.numpy().tolist())
    # b = matlab.double(target.numpy().tolist())
    # c = eng.imhistmatch(a,b)
    # np_c = np.array(c._data.tolist())
    # np_c = np_c.reshape(c.size).transpose()


    # source = Variable(torch.rand(10,9,1).to(device),requires_grad=True)
    # template = torch.randn(10,9,1).to(device)
    matched = hist_match_torch.apply(source, template)
    # hist_loss = HistogramLoss(template)
    # hist_loss(source)
    # hist_loss.loss.backward()

    # print(hist_loss.loss)

    plt.figure()
    plt.hist(source.data.reshape(1,-1))
    plt.figure()
    plt.hist(template.data.reshape(1,-1))
    plt.figure()
    plt.hist(matched.data.reshape(1,-1))


    # def ecdf(x):
    #     """convenience function for computing the empirical CDF"""
    #     vals, counts = np.unique(x, return_counts=True)
    #     ecdf = np.cumsum(counts).astype(np.float64)
    #     ecdf /= ecdf[-1]
    #     return vals, ecdf
    #
    # x1, y1 = ecdf(source.ravel())
    # x2, y2 = ecdf(template.ravel())
    # x3, y3 = ecdf(matched.ravel())
    #
    # fig = plt.figure()
    # gs = plt.GridSpec(2, 3)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    # ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    # ax4 = fig.add_subplot(gs[1, :])
    # for aa in (ax1, ax2, ax3):
    #     aa.set_axis_off()
    #
    # ax1.imshow(source, cmap=plt.cm.gray)
    # ax1.set_title('Source')
    # ax2.imshow(template, cmap=plt.cm.gray)
    # ax2.set_title('template')
    # ax3.imshow(matched, cmap=plt.cm.gray)
    # ax3.set_title('Matched')
    #
    # ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
    # ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
    # ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
    # ax4.set_xlim(x1[0], x1[-1])
    # ax4.set_xlabel('Pixel value')
    # ax4.set_ylabel('Cumulative %')
    # ax4.legend(loc=5)