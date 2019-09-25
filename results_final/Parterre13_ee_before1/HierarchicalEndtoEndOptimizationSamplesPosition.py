import torch
import os
from function_hierarchical_end_to_end_optimization_sample_position import hierarchical_end_to_end_optimization_sample_position
from neighbors_finder import get_knearest_neighbors
from data_importer import normalization, read_point_clouds
import argparse
from utils.poisson_disk import generate_possion_dis
import shutil

torch.cuda.manual_seed_all(2)
torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--optim_method',type=str, default='Adam',help='optimization method')
parser.add_argument('--iteration',type=int, default=1,help='iteration')
parser.add_argument('--upscaling_rate',type=float, default=2,help='Upscaling rate')
parser.add_argument('--exemplar_filename', type=str, default='Parterre13', help='exemplar filename')
parser.add_argument('--img_res', type=int, default=64, help='density image resolution')
parser.add_argument('--img_edge', type=float, default=0.03, help='the location of an edge point')
parser.add_argument('--density_kernel', type=float, default=2, help='please look at the code')

parser.add_argument('--init_guess_filename', type=str, default='blue', help='read init point distribution for texture optimizaiton')
parser.add_argument('--num_optim_step', type=int, default=5000, help='maximum optimization steps')
parser.add_argument('--texture_layers', nargs='+', default=['pool_2', 'pool_4', 'pool_8',
                    'pool_12'], help='texture layers')
parser.add_argument('--texture_weight', type=float, default=1, help='texture layers')
parser.add_argument('--structure_layers', nargs='+', default=['pool_4'],help='layers for computing deep correlations')
parser.add_argument('--structure_weight', type=float, default=10, help='texture layers')
parser.add_argument('--output_dir', type=str,default='results_final', help='output_dir')
parser.add_argument('--kernel_sigma1', type=float,default=0.5, help='kernel_sigma')
parser.add_argument('--kernel_sigma2', type=float,default=2, help='kernel_sigma')

parser.add_argument('--histogram_weight', type=float, default=0, help='histogram weight')
parser.add_argument('--image_histogram_weight', type=float, default=0 , help='image histogram weight')


args = parser.parse_args()

if __name__ == '__main__':

    print(args)

    upscaling_rate = args.upscaling_rate

    num_optim_step = args.num_optim_step
    optim_method = args.optim_method

    texture_layers = args.texture_layers
    structure_layers = args.structure_layers

    filename = args.exemplar_filename
    target_exemplar = torch.load('data/' + filename + '.pt').to(device)
    target_exemplar = target_exemplar[:, :2]

    number_points_in_target_exemplar = target_exemplar.size()[0]


    normalization(target_exemplar, 0, edge_space=0)

    distances5, _ = get_knearest_neighbors(target_exemplar,2,10)
    distances1, _ = get_knearest_neighbors(target_exemplar,2,1)
    distances = distances1.reshape(-1)
    sorted_distances, _ = distances.sort()
    sorted_distances = sorted_distances[sorted_distances!=0]
    num_distances = distances.numel()



    kernel_sigma1 = (distances1.mean()/args.kernel_sigma1).tolist()
    kernel_sigma2 = (distances1.mean()/args.kernel_sigma2).tolist()


    img_edge = (kernel_sigma1+kernel_sigma2)/2

    normalization(target_exemplar, 0, edge_space=img_edge)


    img_res = torch.round((1)/(2*sorted_distances[0])).tolist()
    img_res1 = min(512, img_res)
    img_res1 = max(128, img_res)

    img_res2 = img_res1


    init_guess_filename = args.init_guess_filename


    if init_guess_filename=='blue':

        init_guess = torch.from_numpy(generate_possion_dis(round((upscaling_rate**2)*number_points_in_target_exemplar),0.01,0.99)).to(device).float()
    else:

        init_guess = read_point_clouds('results/20180910_133232/out_points880_3.txt',2,split=' ',normalize=False).to(device).float()
        blue_noise = torch.from_numpy(generate_possion_dis(round(1.1*(upscaling_rate**2)*number_points_in_target_exemplar),0.01,0.99)).to(device).float()
        init_guess = torch.cat((init_guess, blue_noise), 0)

    texture_weight = args.texture_weight  # 1
    structure_weight = args.structure_weight


    results_dir = args.output_dir + '/' + args.exemplar_filename + '_ee_before' + str(args.iteration)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    shutil.copyfile(os.path.abspath('HierarchicalEndtoEndOptimizationSamplesPosition.py'), results_dir + '/HierarchicalEndtoEndOptimizationSamplesPosition.py')
    shutil.copyfile(os.path.abspath('function_hierarchical_end_to_end_optimization_sample_position.py'), results_dir + '/function_hierarchical_end_to_end_optimization_sample_position.py')


    optimized_points = hierarchical_end_to_end_optimization_sample_position(init_guess, target_exemplar, upscaling_rate, img_res1,
                                                               img_res2, kernel_sigma1,kernel_sigma2, num_optim_step, texture_weight, structure_weight, args.histogram_weight,args.image_histogram_weight,
                              texture_layers, structure_layers, optim_method, results_dir)




