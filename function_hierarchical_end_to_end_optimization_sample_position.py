# -*- coding: utf-8 -*-
from __future__ import print_function
import torch

import torch.optim as optim

import matplotlib.pyplot as plt

from utils.utils_py import save_image

from loss import StyleLoss
from histogram_loss import HistogramLoss
import numpy as np
from vgg_model_v2 import get_style_model_and_losses
import torchvision.models as models
from point2image import Point2Image_fast

break_loop = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

cnn = models.vgg19(pretrained=True).features.to(device).eval()


def hierarchical_end_to_end_optimization_sample_position(init_guess, target_exemplar, upscaling_rate, img_res1,img_res2, kernel_sigma1,kernel_sigma2,
                              num_optim_step, texture_weight, structure_weight, histogram_weight, image_histogram_weight,
                              texture_layers, structure_layers, optim_method, results_dir):
    global break_loop
    kernel_sigma_list = torch.linspace(kernel_sigma1, kernel_sigma2, 3)
    img_res_list = torch.linspace(img_res1, img_res2, 3)
    stopping_crit_list = torch.linspace(0.01, 0.01, 3)
    outerloop = 0

    lr_list = [0.02, 0.01, 0.01, 0.002, 0.002, 0.002, 0.002, 0.002]
    run = [0]
    while outerloop < 3:
        stopping_crit = stopping_crit_list[outerloop].tolist()
        kernel_sigma = kernel_sigma_list[outerloop].tolist()
        img_res = img_res_list[outerloop].tolist()
        img_res_input = round(img_res * upscaling_rate)
        kernel_sigma_input = kernel_sigma/upscaling_rate

        # if outerloop%2 == 0:
        from point2image import Point2Image
        # else:
        #     from point2image_merge import Point2Image

        target_p2i = Point2Image(2, 0, kernel_sigma=kernel_sigma,
                                              feature_sigma=0, res=img_res)
        input_p2i = Point2Image(2, 0, kernel_sigma=kernel_sigma_input,
                                              feature_sigma=0, res=img_res_input)

        if optim_method == 'LBFGS':
        # input_soft_points = torch.randn(img_res_input,img_res_input).to(device)
            optimizer = optim.LBFGS([init_guess.requires_grad_()],lr=0.5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        elif optim_method == 'Adam':
            optimizer = optim.Adam([init_guess.requires_grad_()], lr =lr_list[outerloop])
        print('step:', lr_list[outerloop])
        target_texture_img = target_p2i(target_exemplar).repeat(1, 3, 1, 1).to(device)

        # target_texture_img = Point2Image_fast.apply(target_exemplar,kernel_sigma,img_res).repeat(1, 3, 1, 1).to(device)

        fstyle_loss = StyleLoss(target_texture_img)


        save_image(target_texture_img.squeeze()/ target_texture_img.max(), results_dir + '/target' + str(outerloop) + '.jpg')

        np.savetxt(results_dir + '/target' + str(outerloop) + '.txt', target_texture_img[0,0,:,:].cpu().data.numpy())
        np.savetxt(results_dir + '/target_points' + str(outerloop) + '.txt', target_exemplar.cpu().data.numpy())


        fig=plt.figure()
        plt.scatter(target_exemplar.data[:, 0], target_exemplar.data[:, 1])
        plt.savefig(results_dir + '/scatter_target' + str(outerloop) +  '.jpg')
        plt.close(fig)


        img_hist_loss = HistogramLoss(target_texture_img)
        print('Building the texture model..')
        model, texture_losses, structure_losses,_, histogram_losses = get_style_model_and_losses(cnn, target_texture_img,
                                                                             texture_layers, structure_layers)
        log_losses = []

        print('Optimizing..')
        break_loop = False
        inner_run = [0]
        while run[0] <= num_optim_step and not break_loop:

            def closure():
                global break_loop
                init_guess.data.clamp_(0, 1)
                # scheduler.step()
                input_soft_points = input_p2i(init_guess)
                #
                # input_soft_points = Point2Image_fast.apply(init_guess,kernel_sigma_input,img_res_input)

                input_soft_points.clamp(min=target_texture_img.min(), max=target_texture_img.max())

                optimizer.zero_grad()

                input_density_img = input_soft_points.repeat(1, 3, 1, 1)
                img_hist_loss(input_density_img)

                if run[0] == 0:
                    save_image(input_density_img.squeeze() / input_density_img.max(),
                               results_dir + '/init'+ str(outerloop) +'.jpg')
                    np.savetxt(results_dir + '/init'+ str(outerloop) +'.txt', input_density_img[0,0,:,:].cpu().data.numpy())
                    np.savetxt(results_dir + '/init_points' + str(outerloop) + '.txt', init_guess.cpu().data.numpy())
                    fig = plt.figure()
                    plt.scatter(init_guess.data[:, 0], init_guess.data[:, 1])
                    plt.savefig(results_dir + '/init_points' + str(outerloop) + '.jpg')
                    plt.close(fig)


                model(input_density_img)

                texture_score = torch.zeros(1).to(device)
                structure_score = torch.zeros(1).to(device)
                histogram_score = torch.zeros(1).to(device)

                img_hist_score = torch.zeros(1).to(device)

                # img_hist_score = image_histogram_weight * img_hist_loss.loss

                for tl in texture_losses:
                    texture_score += texture_weight * tl.loss
                for sl in structure_losses:
                    structure_score += structure_weight * sl.loss  #100000
                # for hl in histogram_losses:
                #     histogram_score += histogram_weight * hl.loss.data[0]

                fstyle_loss(input_density_img)
                ftexture_score = fstyle_loss.loss

                # ftexture_score = torch.zeros(1).to(device)

                loss = texture_score + structure_score + histogram_score + img_hist_score + ftexture_score  #+ homo_score

                loss.backward()

                log_losses.append(loss)
                run[0] += 1
                inner_run[0] += 1
                if run[0] % 5 == 0:

                    for param_group in optimizer.param_groups:
                        print(param_group['lr'])


                    print("run {}:".format(run))
                    print('Texture Loss : {:4f}, FTexture_Loss: {:4f}, Structure Loss : {:4f}, Histogram Loss : {:4f}, Image Histogram Loss : {:4f}'.format(
                        texture_score.item(), ftexture_score.item(), structure_score.item(),histogram_score.item(),img_hist_score.item()))

                    save_image((input_density_img/input_density_img.max()),
                               results_dir + '/out' + str(run[0]) + '_' + str(outerloop) + '.jpg')
                    np.savetxt(results_dir + '/out' + str(run[0]) + '_' + str(outerloop) + '.txt', input_density_img[0,0,:,:].cpu().data.numpy())

                    np.savetxt(results_dir + '/out_points' + str(run[0]) + '_' + str(outerloop) + '.txt', init_guess.cpu().data.numpy())
                    fig = plt.figure()
                    plt.figure()
                    plt.scatter(init_guess.data[:, 0], init_guess.data[:, 1],s=2)
                    plt.savefig(results_dir + '/out_points' + str(run[0]) + '_' + str(outerloop) + '.jpg')
                    plt.close('all')

                    print('inner_run',inner_run)
                    print('stopping_crit',stopping_crit)
                    if inner_run[0] > 100:
                        loss_init = log_losses[0]
                        loss_pre = log_losses[inner_run[0]-100]
                        loss_now = log_losses[inner_run[0]-1]

                        decrease_perc = ((loss_pre-loss_now)/(loss_init-loss_now)).tolist()[0]
                        print(decrease_perc)
                        if decrease_perc < stopping_crit:
                            print('converged')
                            break_loop = True
                        else:
                            np.savetxt(results_dir + '/final_output' + '_' + str(outerloop) + '.txt',
                                       init_guess.cpu().data.numpy())
                            np.savetxt(results_dir + '/final_output_density' + '_' + str(outerloop) + '.txt',
                                       input_density_img[0, 0, :, :].cpu().data.numpy())

                            np.savetxt(results_dir + '/final_output' + '.txt', init_guess.cpu().data.numpy())
                            np.savetxt(results_dir + '/final_output_density.txt', input_density_img[0, 0, :, :].cpu().data.numpy())
                    print()

                    with open(results_dir + '/log_losses_' + str(outerloop) + '.txt', "w") as f:
                        for s in log_losses:
                            f.write(str(s.tolist()[0]) + "\n")

                return loss

            optimizer.step(closure)
        outerloop +=1
    init_guess.data.clamp_(0, 1)

    return init_guess

