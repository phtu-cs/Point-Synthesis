import torch

def GaussianRGF_fast(w, mean, variance, x):
    # x: (#neighboring points, dim)
    # mean: (#basis, dim)
    # variance: float
    # w: (#basis, C_in, C_out)

    assert mean.size()[0] == w.size()[0]  # check number of basis
    assert x.size()[1] == mean.size()[1]  # check dim

    num_neighboring_points, dim = x.size()
    num_basis = mean.size()[0]
    _, C_in, C_out = w.size()

    if x.dim() == 1:
        x = x.unsqueeze(0)

    conv_weights = ((torch.exp(-((x.repeat(num_basis, 1, 1).permute(1, 0, 2) - mean).pow(2).sum(2))
                     / (2 * variance))).repeat(C_in, C_out, 1, 1).permute(2, 3, 0, 1) * w).sum(1)

  #  conv_weights =

    #conv_weights = (tmp.repeat(C_in, C_out, 1, 1).permute(2, 3, 0, 1) * w).sum(1)

    return conv_weights
