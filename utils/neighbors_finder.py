
import torch
from utils.data_importer import read_point_clouds
import numpy as np
import scipy.spatial as spatial
from utils.vis import show_neighbors_2d_fast,show_neighbors_2d

'''
def get_neighbors_fixed_field(p, D, C_in, frac):

    # frac the proportion of receptive field in entire domain
    # D spatial dimension
    # C_in feature dimension

    #assert type(p.data) == torch.Tensor
    assert p.dim() == 2
    p = p.float()

    # number of points
    N = p.size()[0]

    # upper right and bottom left corner (domain size)
    ur_corner = torch.zeros(D)
    bl_corner = torch.zeros(D)
    dsize = torch.zeros(D, requires_grad=False)

    for d in range(0, D):
        ur_corner[d] = p[:, d].max()
        bl_corner[d] = p[:, d].min()
        dsize[d] = ur_corner[d] - bl_corner[d]

   # half_w =  (0.5 * frac * dsize).max()

    half_w = frac

    np = []


    # Very inefficient now, force search
    for n in range(N):
        tmp = []
        for nn in range(N):
            #if nn != n:
            dis = (p[n, :D] - p[nn, :D]).pow(2).sum().sqrt()
            if dis.data.tolist() < half_w:
                tmp.append([nn] + p[nn, :].tolist())
            tmp_th =  torch.Tensor(tmp)
        np.append(tmp_th)

    return np

'''

def get_knearest_neighbors(p, D, C_in, frac):
    if torch.cuda.is_available():
        points_tree = spatial.cKDTree(p[:,:D].cpu().data.numpy())
        np = points_tree.query_ball_point(p[:,:D].cpu().data.numpy(),frac)
    else:
        points_tree = spatial.cKDTree(p[:,:D].data.numpy())
        np = points_tree.query_ball_point(p[:,:D].data.numpy(),frac)

    return np

def get_neighbors_fixed_field_fast(p, D, C_in, frac):

    # frac the proportion of receptive field in entire domain
    # D spatial dimension
    # C_in feature dimension

    #assert type(p.data) == torch.Tensor
    assert p.dim() == 2
    p = p.float()

    # number of points
    N = p.size()[0]

    if torch.cuda.is_available():
        points_tree = spatial.cKDTree(p[:,:D].cpu().data.numpy())
        np = points_tree.query_ball_point(p[:,:D].cpu().data.numpy(),frac)
    else:
        points_tree = spatial.cKDTree(p[:,:D].data.numpy())
        np = points_tree.query_ball_point(p[:,:D].data.numpy(),frac)


    return np


if __name__ == "__main__":
    p = read_point_clouds('txt/exp.txt',2)
    p = p.float()
    p = p[:500, :]
    np1 = get_neighbors_fixed_field_fast(p, 2, 3, 0.3)
    show_neighbors_2d_fast(p, np1, 10)


    np2 = get_neighbors_fixed_field(p, 2, 3, 0.3)
    show_neighbors_2d(p,np2,10)






