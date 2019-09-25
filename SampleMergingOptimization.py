import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--optim_method',type=str,default='Adam',help='optimization method')
parser.add_argument('--iteration', type=int, default=1,help='iteration')
parser.add_argument('--upscaling_rate',type=float, default=2,help='Upscaling rate')
parser.add_argument('--exemplar_filename', type=str, default='Tree1', help='exemplar filename')
parser.add_argument('--img_res', type=int, default=64, help='density image resolution')
parser.add_argument('--img_edge', type=float, default=0.03, help='the location of an edge point')
parser.add_argument('--density_kernel', type=float, default=2, help='please look at the code')

parser.add_argument('--init_guess_filename', type=str, default='final_output', help='read init point distribution for texture optimizaiton')
parser.add_argument('--num_optim_step', type=int, default=5000, help='maximum optimization steps')
parser.add_argument('--texture_layers', nargs='+', default=['pool_2', 'pool_4', 'pool_8',
                    'pool_12'], help='texture layers')
parser.add_argument('--texture_weight', type=float, default=1, help='texture layers')
parser.add_argument('--structure_layers', nargs='+', default=['pool_4'],help='layers for computing deep correlations')
parser.add_argument('--structure_weight', type=float, default=100, help='structure weight')
parser.add_argument('--input_dir', type=str, default='results_final', help='input_dir')
parser.add_argument('--output_dir', type=str, default='results_final', help='output_dir')
parser.add_argument('--kernel_sigma1', type=float,default=2, help='kernel_sigma1')
parser.add_argument('--kernel_sigma2', type=float,default=4, help='kernel_sigma2')
parser.add_argument('--histogram_weight', type=float, default=100, help='histogram weight')

args = parser.parse_args()


import torch
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':


    upscaling_rate = args.upscaling_rate
    num_optim_step = args.num_optim_step
    optim_method = args.optim_method

    texture_layers = args.texture_layers
    structure_layers = args.structure_layers

    filename = args.exemplar_filename
    target_exemplar = torch.load('data/' + filename + '.pt').to(device)
    target_exemplar = target_exemplar[:, :2]


    number_points_in_target_exemplar = target_exemplar.size()[0]


    from data_importer import normalization, read_point_clouds
    normalization(target_exemplar, 0, edge_space=0)

    from neighbors_finder import get_knearest_neighbors
    distances5, _ = get_knearest_neighbors(target_exemplar,2,10)
    distances1, _ = get_knearest_neighbors(target_exemplar,2,1)
    distances = distances1.reshape(-1)
    sorted_distances, _ = distances.sort()
    sorted_distances = sorted_distances[sorted_distances!=0]
    num_distances = distances.numel()


    kernel_sigma = (distances1.mean() / args.kernel_sigma1).tolist()
    kernel_sigma2 = (distances1.mean()/args.kernel_sigma2).tolist()

    kernel_sigma1 = (kernel_sigma + kernel_sigma2) / 2

    img_edge = (kernel_sigma + kernel_sigma2) / 2


    normalization(target_exemplar, 0, edge_space=img_edge)

    img_res = torch.round((1)/(2*sorted_distances[0])).tolist()
    img_res1 = min(512, img_res)
    img_res1 = max(128, img_res)

    img_res2 = img_res1


    init_guess_filename = args.init_guess_filename


    if init_guess_filename=='blue':
        from utils.poisson_disk import generate_possion_dis
        init_guess = torch.from_numpy(generate_possion_dis(1*round((upscaling_rate**2)*number_points_in_target_exemplar),0.01,0.99)).to(device).float()

    elif init_guess_filename == 'final_output':

        init_guess = read_point_clouds(
            args.input_dir + '/' + filename + '_ee_before' + str(args.iteration) + '/' + init_guess_filename + '.txt', 2,
            split=' ', normalize=False).to(
            device).float()

    texture_weight = args.texture_weight  # 1
    structure_weight = args.structure_weight

    if not args.output_dir:
        results_dir = args.output_dir + '/' + args.exemplar_filename + '_ee' + str(args.iteration)
    else:
        results_dir = args.output_dir + '/' + args.exemplar_filename + '_ee' + str(args.iteration)

    import os
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    import shutil
    shutil.copyfile(os.path.abspath('SampleMergingOptimization.py'), results_dir + '/SampleMergingOptimization.py')
    shutil.copyfile(os.path.abspath('function_sample_merging_optimization.py'), results_dir + '/function_sample_merging_optimization.py')
    print(args)
    from function_sample_merging_optimization import sample_merging_optimization
    optimized_points = sample_merging_optimization(init_guess, target_exemplar, upscaling_rate, img_res1,
                                                               img_res2, kernel_sigma1,kernel_sigma2, num_optim_step, texture_weight, structure_weight, args.histogram_weight,
                              texture_layers, structure_layers, optim_method, results_dir)







