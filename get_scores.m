function [scores1,scores2] = get_scores(final_points,energies,kernel_energy)


% hwtimes = 6;

% [kx,ky] = meshgrid(floor(-kernel_sigma_res*hwtimes):ceil(kernel_sigma_res*hwtimes ),floor(-kernel_sigma_res*hwtimes):ceil(kernel_sigma_res*hwtimes));

% kernel = exp(-(kx.^2+ky.^2)/(2*kernel_sigma_res^2));
% kernel_energy = sum(kernel(:));

% [energies,kernels] = get_energy_points(final_points,kx,ky,res);
% scatter(final_points(:,1),final_points(:,2),[],energies);

K = 5; %number of neighbors
pc = pointCloud(cat(2,final_points(:,1:2),ones(size(final_points(:,1:2),1),1)));

scores_removed = zeros(size(final_points,1),1);
scores_added = zeros(size(final_points,1),1);
timings_removed = zeros(size(final_points,1),1);
timings_added = zeros(size(final_points,1),1);

for ip = 1:pc.Count
    point = pc.Location(ip,:);
    [indices, ~] = findNearestNeighbors(pc,point,K+1);
    num_points = sum(energies(indices))/kernel_energy;
    
    if num_points <= K + 1
        average_point = energies(indices)/kernel_energy;    
        diff_point =  K+1-num_points;       
        tmp = 1-average_point;  
        tmp1 = tmp;
        tmp1(tmp<=0)= [];     
%         indices(tmp<=0) = [];
        tmp1 = tmp1/sum(tmp1(:));
        scores_removed(indices(find(tmp>0))) = tmp1*diff_point + scores_removed(indices(find(tmp>0)));
        timings_removed(indices(find(tmp>0))) =  timings_removed(indices(find(tmp>0))) + 1;   
    else 
        average_point = energies(indices)/kernel_energy; 
        diff_point =  num_points-(K+1);
        tmp = average_point-1;
        tmp2 = tmp;      
        tmp2(tmp<=0)= [];
        tmp2 = tmp2/sum(tmp2(:));     
%         ip
% ip
        scores_added(indices(find(tmp>0))) = scores_added(indices(find(tmp>0)))+ tmp2*diff_point;
        timings_added(indices(find(tmp>0))) =  timings_added(indices(find(tmp>0))) + 1;
        
    end
end

scores1 = scores_removed./timings_removed;
scores2 = scores_added./timings_added;
% scores1 = scores_removed;
% scores2 = scores_added;

end