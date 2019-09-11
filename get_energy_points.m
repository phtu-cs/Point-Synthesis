function [output,kernels] = get_energy_points(input,kx,ky,res)

kernel_height = input(:,3);

grids =  permute(repmat(kx.^2+ky.^2,1,1,numel(kernel_height)),[3,1,2]);

kernel_sigmas = input(:,4)*res;

kernels = repmat(kernel_height,1,size(kx,1),size(kx,2)).*...
       exp(-grids./repmat(2*kernel_sigmas.^2,1,size(kx,1),size(kx,2)));
    
output = sum(sum(kernels,2),3);

end