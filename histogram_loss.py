import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hist_match_torch(Function):

    @staticmethod
    def forward(ctx,input,target):

        ctx.input = input

        a = input.clone().cpu().numpy()
        b = target.clone().cpu().numpy()

        matched = torch.from_numpy(hist_match(a,b)).to(device).float()


        return matched

    @staticmethod
    def backward(ctx,grad_output):

        grad_input = torch.zeros(ctx.input.size()).to(device)

        return grad_input, None

class HistogramLoss(nn.Module):

    def __init__(self, target):
        super(HistogramLoss, self).__init__()
        self.target = target

    def forward(self, input):
        hist_matched = hist_match_torch.apply(input, self.target)
        self.loss = (hist_matched-input).pow(2).sum()/input.numel()
        self.output = input

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