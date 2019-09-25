
import torch
import torch.nn as nn
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors

'''find neighborhoods within the point patterns'''

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

        return distances_torch



def get_neighbors_fixed_field_fast(p, D, frac):


    assert p.dim() == 2
    p = p.float()

    if torch.cuda.is_available():
        points_tree = spatial.cKDTree(p[:,:D].cpu().data.numpy())
        np = points_tree.query_ball_point(p[:,:D].cpu().data.numpy(),frac)
    else:
        points_tree = spatial.cKDTree(p[:,:D].data.numpy())
        np = points_tree.query_ball_point(p[:,:D].data.numpy(),frac)


    return np






