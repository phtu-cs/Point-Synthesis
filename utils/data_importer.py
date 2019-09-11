import os
from numpy import array
import torch
import utils.vis as vis

#Read point clouds from txt file

def read_point_clouds(filename, D,normalize=True):
    assert type(filename) == str
    f = open(filename, 'r')
    l = [[float(num) for num in line.split(',')] for line in f]
    np_l = array(l)
    th_l = torch.from_numpy(np_l)

    position = th_l[:,:D]

    if normalize == True:

        min = position.min(0)[0]
        max = position.max(0)[0]
        r = max - min

        for id in range(D):
           th_l[:, id] = (th_l[:, id]-min[id])/r[id]

        th_l[:,D:] /=255


    return th_l


if __name__ == "__main__":
    th_l = read_point_clouds('exp.txt', 2, normalize=False)
    print(th_l)

    vis.show_2d(th_l)








#