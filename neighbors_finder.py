
import torch
import torch.nn as nn
from torch.autograd import Function
# from utils.data_importer import read_point_clouds
import numpy as np
import scipy.spatial as spatial
from utils.vis import show_neighbors_2d_fast,show_neighbors_2d
from sklearn.neighbors import NearestNeighbors
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

def get_knearest_neighbors(p, D, num_neighbors):
    if torch.cuda.is_available():
        X = p[:, :D].cpu().data.numpy()
        nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
    else:
        X = p[:, :D].data.numpy()
        nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
    return torch.from_numpy(distances)[:, 1:], torch.from_numpy(indices)[:, 1:]

class get_knearest_neighbors_torch(nn.Module):


    def __init__(self,num_neighbors):
        super(get_knearest_neighbors_torch,self).__init__()

        self.num_neighbors = num_neighbors

    def forward(self, p):

        if torch.cuda.is_available():
            X = p[:, :2].cpu().data.numpy()
            nbrs = NearestNeighbors(n_neighbors=self.num_neighbors + 1, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
        else:
            X = p[:, :2].data.numpy()
            nbrs = NearestNeighbors(n_neighbors=self.num_neighbors + 1, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

        indices_torch = torch.from_numpy(indices)

        distances_torch = torch.zeros(p.size()[0],self.num_neighbors)

        for ip in range(p.size()[0]):

            distances_torch[ip,:] = (p[ip,:2]-p[indices_torch[ip,1:],:2]).pow(2).sum(1).pow(0.5)

        # output = torch.zeros(distances_torch.size())
        # for io in range(self.num_neighbors):
        #     output[:,io] = distances_torch[:,io]/distances_torch[:,io].max()
        # # distances_mean = distances_torch.mean(0)
        # # distances_std = distances_torch.std(0)
        # # out = distances_torch -distances_mean + 0.5

        return distances_torch


class get_knearest_neighbors_torch_xy(nn.Module):

    def __init__(self,num_neighbors):
        super(get_knearest_neighbors_torch_xy,self).__init__()

        self.num_neighbors = num_neighbors

    def forward(self, p):

        if torch.cuda.is_available():
            X = p[:, :2].cpu().data.numpy()
            nbrs = NearestNeighbors(n_neighbors=self.num_neighbors + 1, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
        else:
            X = p[:, :2].data.numpy()
            nbrs = NearestNeighbors(n_neighbors=self.num_neighbors + 1, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

        indices_torch = torch.from_numpy(indices)

        distances_torch = torch.zeros(p.size()[0],self.num_neighbors,2)

        for ip in range(p.size()[0]):
            distances_torch[ip,:, 0] = p[ip, 0] - p[indices_torch[ip, 1:], 0]
            distances_torch[ip,:, 1] = p[ip, 1] - p[indices_torch[ip, 1:], 1]
        # output = torch.zeros(distances_torch.size())
        # for io in range(self.num_neighbors):
        #     output[:,io] = distances_torch[:,io]/distances_torch[:,io].max()
        # # distances_mean = distances_torch.mean(0)
        # # distances_std = distances_torch.std(0)
        # # out = distances_torch -distances_mean + 0.5

        return distances_torch






def get_neighbors_fixed_field_fast(p, D, frac):

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
    # p = read_point_clouds('txt/exp.txt',2)
    # p = p.float()
    # p = p[:500, :]
    p = torch.rand(100,5)
    distances, indices =get_knearest_neighbors_torch.apply(p,20)


    # np1 = get_neighbors_fixed_field_fast(p, 2, 3, 0.3)
    # show_neighbors_2d_fast(p, indices, 10)

    #
    # np2 = get_neighbors_fixed_field(p, 2, 3, 0.3)
    # show_neighbors_2d(p,np2,10)






