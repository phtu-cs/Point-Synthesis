from numpy import array
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_point_clouds(filename, D, split = ',' ,normalize=True):
    assert type(filename) == str
    f = open(filename, 'r')
    l = [[float(num) for num in line.split(split)] for line in f]
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

def normalization(points, dd_feature, d_space=2, edge_space=0.1, egde_feature = 0.2, norm=True):


    min = points.min(0)[0]
    max = points.max(0)[0]

    for id in range(d_space):
        r = max[id] - min[id]
        points[:, id] = (((points[:, id] - min[id]) / r)*(1-2*edge_space) + edge_space)

    if norm == True:
        if points.size()[1] > 2:

            for id in range(d_space+dd_feature, points.size()[1]):
                r = max[id] - min[id]

                if r == 0:
                    continue

                points[:, id] = ((points[:, id] - min[id]) / r)*(1-egde_feature) + egde_feature

    return points



def OneHot(points, dd_feature, dc_feature, edge_feature=0.2, d_space=2):


    points_dfeat= points[:, d_space:d_space+dd_feature]
    if dc_feature!=0:
        points_cfeat = points[:, d_space+dd_feature:d_space + dd_feature+dc_feature]
    channels,_ = points_dfeat.max(0)
    output_list = []

    for ic in range(channels.size()[0]):
        for i in range(int(channels[ic].tolist())):
            if channels[ic] == 2:
                tmp = torch.ones(points.size()[0]).to(device)
                tmp[i+1 == points_dfeat[:,ic]]=edge_feature
                output_list.append(tmp.unsqueeze(1))
                break
            else:
                tmp = edge_feature*torch.ones(points.size()[0]).to(device)
                tmp[i+1 == points_dfeat[:,ic]]=1
                output_list.append(tmp.unsqueeze(1))
    if dc_feature != 0:
        output_list.append(points_cfeat)
    output = torch.cat(output_list,1)
    channels[channels==2] =1
    return output,channels





#