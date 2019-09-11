function MergePoints_v5(name,input_folder,output_folder,output)

it = 1;
final_img = load(fullfile(input_folder,[name,'_ee',num2str(it)],'final_output_density.txt'));

res = size(final_img,1);

init_points = load(fullfile(input_folder,[name,'_ee',num2str(it)],'final_output.txt'));
target_points = load(fullfile(input_folder,[name,'_ee',num2str(it)],'target_points0.txt'));
% target_pc = pointCloud(cat(2,target_points(:,1:2),ones(size(target_points(:,1:2),1),1)));

% [indices, distances] = findNearestNeighbors(pc,pc.Location(added,:),K+1);



hwtimes = 6;
% kernel_sigma_res = init_points(1,4)*res;
kernel_sigma_res = 0.001*res;
[kx,ky] = meshgrid(floor(-kernel_sigma_res*hwtimes):ceil(kernel_sigma_res*hwtimes ),floor(-kernel_sigma_res*hwtimes):ceil(kernel_sigma_res*hwtimes));

kernel = exp(-(kx.^2+ky.^2)/(2*kernel_sigma_res^2));
kernel_energy = sum(kernel(:));

point_diff = round((sum(final_img(:))-size(init_points,1)*kernel_energy)/kernel_energy);
final_points = load(fullfile(input_folder,[name,'_ee',num2str(it)],'final_output.txt'));

previous = load(fullfile(output_folder,[name,'_ee_before',num2str(it)],'final_output.txt'));


% final_points(:,1:2) = final_points(:,1:2)+0.001*randn(size(final_points,1),2); 

output_points = final_points;

output_points = cat(2,output_points,0.001*ones(size(output_points,1),1));
[final_energies,~] = get_energy_points(output_points,kx,ky,res);

energies = final_energies;

[scores1,scores2]=get_scores(output_points,energies,kernel_energy);
[~,removed]=max(scores1);
[~,added]=max(scores2);

K=7;
change = 0;
break_loop = false;


added_energy = [];


added_point =[]

iteration = 0;
%point_diff+change<=0 &&
while  ~break_loop && iteration < size(final_points,1)
    break_loop = true;
%     figure(1);scatter(output_points(:,1),output_points(:,2),[],scores1,'filled');colorbar;colormap jet
%     figure(2);scatter(output_points(:,1),output_points(:,2),[],energies/kernel_energy-1,'filled');colorbar;colormap jet
    pc = pointCloud(cat(2,output_points(:,1:2),ones(size(output_points(:,1:2),1),1)));
    
    if energies(added)>2*kernel_energy
        
        [indices, distances] = findNearestNeighbors(pc,pc.Location(added,:),K+1);

        indices(1)=[];
        distances(1)=[];
        added_displacement = [0.001*randn(1,2),1];
        
        point_added = [pc.Location(added,:)+added_displacement,1];
        
        distances_square = (1./(distances+1e-8).^2);
        
        %give the energy to the nearby point 
        coes = distances_square/sum(distances_square);
        energies(indices)= energies(indices) - coes.*repmat(2*kernel_energy-energies(added),K,1);
  
        output_points = cat(1,output_points,point_added);
%         added_point = [added_point;point_added];
        energies(added)=kernel_energy;
        energies = [energies;kernel_energy];
        
        change = change -1
        break_loop = false;
    end
    pc = pointCloud(cat(2,output_points(:,1:2),ones(size(output_points(:,1:2),1),1)));
    
    if energies(removed)<0.8*kernel_energy
        
        point_removed = pc.Location(removed,:);
        
        if abs(point_removed(1)-0.4618)<0.00005 && abs(point_removed(2)-0.602)<0.0005
           point_removed
        end
        
        
        [indices, distances] = findNearestNeighbors(pc,point_removed,K+1);
        distances(indices==removed)=[];
        indices(indices==removed)=[];
%         indices(1)=[];
%         distances(1)=[];
        distances_square = (1./(distances+1e-8).^2);
        
        %give the energy to the nearby point 
        coes = distances_square/sum(distances_square);
        energies(indices)=coes.*repmat(energies(removed),K,1) + energies(indices);
        energies(removed)= [];
        output_points(removed,:)= [];    
        change = change + 1
        break_loop = false;
        isremoved = true;
    end
    
%     [scores1,scores2]=get_scores(output_points,energies,kernel_energy);
    
%     scatter3(output_points(:,1),output_points(:,2),energies/kernel_energy,[],energies/kernel_energy,'fill');axis([0 1 0 1]);colormap jet;;caxis([0 2])
%     scatter3(output_points(:,1),output_points(:,2),scores1,[],scores1,'fill');axis([0 1 0 1]);colormap jet;;caxis([0 2])

    scatter3(output_points(:,1),output_points(:,2),energies/kernel_energy,[],energies/kernel_energy,'fill');axis([0 1 0 1]);colormap jet;;caxis([0 2])
    [~,removed]=min(energies); 
    [~,added]=max(energies);
%     [~,removed]=max(scores1);  
%     [~,added]=max(scores2);
    iteration = iteration + 1
    
end

% output_points = cat(1,output_points,added_point);
% energies = cat(1,energies,ones(size(added_point,1),1));
mkdir(['../figs/data_example/',name,'_',output]);
mark_size = 80;
fig1=figure;scatter(target_points(:,1),target_points(:,2),'filled');axis square
saveas(fig1,['../figs/data_example/',name,'_',output,'/target'],'png');
show_point_dist(4, target_points,['../figs/data_example/',name,'_',output,'/'],'target.svg')

if strcmp(output,'stress')
    circle_size = 4;
elseif strcmp(output,'final')
    circle_size = 2;
elseif strcmp(output,'ablation')
    circle_size = 2;
end

fig2=figure('pos',[1,1,1200,1200]);scatter(previous(:,1),previous(:,2),mark_size,'filled');axis square
saveas(fig2,['../figs/data_example/',name,'_',output,'/prev'],'png');
show_point_dist(circle_size, previous,['../figs/data_example/',name,'_',output,'/'],'prev.svg')



fig2=figure('pos',[1,1,1200,1200]);scatter(output_points(:,1),output_points(:,2),mark_size,'filled');axis square
saveas(fig2,['../figs/data_example/',name,'_',output,'/output'],'png');
show_point_dist(circle_size, output_points,['../figs/data_example/',name,'_',output,'/'],'output.svg')


% figure;scatter3(final_points(:,1),final_points(:,2),final_energies/kernel_energy,[],final_energies/kernel_energy,'fill');axis([0 1 0 1]);colormap jet;caxis([0 2])
% figure;scatter3(output_points(:,1),output_points(:,2),energies/kernel_energy,[],energies/kernel_energy,'fill');axis([0 1 0 1]);colormap jet;;caxis([0 2])
% 
% % figure;scatter3(final_points(:,1),final_points(:,2),final_energies/kernel_energy);axis([0 1 0 1]);axis square
% % figure;scatter3(output_points(:,1),output_points(:,2),energies/kernel_energy);axis([0 1 0 1]);axis square
% 
% figure;imagesc(final_img);set(gca,'YDir','normal');
% 

dlmwrite(fullfile(input_folder,[name,'_ee',num2str(it)],['cleaned_output.txt']),output_points(:,1:2),'delimiter',' ');
