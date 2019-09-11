import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# from utils.data_importer import read_point_clouds
# from utils.neighbors_finder import get_neighbors_fixed_field

def show_2d(p,filename=None):
    x = p[:, 0].clone().cpu().numpy()
    y = p[:, 1].clone().cpu().numpy()
    if p[:, 2:5].max() > 1:
        c = p[:, 2:5].clone().cpu().numpy() / 255
    else:
        c = p[:, 2:5].clone().cpu().numpy()
    if filename == None:

        plt.scatter(x, y, s=10, c=c, marker='o')
        plt.show()
    else:
        plt.scatter(x, y, s=10, c=c, marker='o')
        plt.savefig(filename)
        plt.close()

# show the neighbors of #index point
def show_neighbors_2d(p, np, index):
    plt.figure()
    x = p[:, 0].numpy()
    y = p[:, 1].numpy()
    if p[:, 2:5].max() > 1:
        c = p[:, 2:5].clone().numpy()/255
    else:
        c = p[:, 2:5].clone().numpy()

    # c[index, 0] = 0
    # c[index, 1] = 1
    # c[index, 2] = 0
    for i in range(len(np[index])):
        c[np[index][i, 0].int(), 0] = 1
        c[np[index][i, 0].int(), 1] = 0
        c[np[index][i, 0].int(), 2] = 0

    c[index, 0] = 0
    c[index, 1] = 1
    c[index, 2] = 0

    plt.scatter(x, y, s=10, c=c, marker='o')

    plt.show()

def show_neighbors_2d_fast(p, np, index):
    plt.figure()
    x = p[:, 0].numpy()
    y = p[:, 1].numpy()
    if p[:, 2:5].max() > 1:
        c = p[:, 2:5].clone().numpy()
    else:
        c = p[:, 2:5].clone().numpy()

    # c[index, 0] = 0
    # c[index, 1] = 1
    # c[index, 2] = 0
    for i in range(len(np[index])):
        c[np[index][i], 0] = 1
        c[np[index][i], 1] = 0
        c[np[index][i], 2] = 0

    c[index, 0] = 0
    c[index, 1] = 1
    c[index, 2] = 0

    plt.scatter(x, y, s=10, c=c, marker='o')

    plt.show()


# if __name__ == "__main__":
#
#     p = read_point_clouds('exp.txt',2,True)
#     p = p[:100,:]
#     np = get_neighbors_fixed_field(p, 2, 3, 0.3)
#     show_neighbors_2d(p, np, 50)
