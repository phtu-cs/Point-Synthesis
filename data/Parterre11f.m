input_folder1 = 'rendering/objects/shu1/';
input_folder2 = 'rendering/objects/fangzi02/';

input_file_name1 = {'xiangshu','xiangshu2','xiangshu3','xiangshu4','xiangshu5'};
input_file_name2 = {'hongfangzi','huifangzi','lanfangzi'};

it=1;
example = 1;
name = 'Parterre12f';

if example == 1
   input_points = load(fullfile([name,'.txt']));
   input_points(:,1) = (input_points(:,1)-min(input_points(:,1)))/(max(input_points(:,1))-min(input_points(:,1)));
   input_points(:,2) = (input_points(:,2)-min(input_points(:,2)))/(max(input_points(:,2))-min(input_points(:,2)));
   tmp = input_points(:,3);
   tmp(tmp==1)=0.2064;
   tmp(tmp==2)=1;
   input_points(:,3) = tmp;
else
   input_points = load(fullfile('results_final',[name,'_features',num2str(it)],'matched_output.txt'));
end

node_xyz = read_v([input_folder1,input_file_name1{1},'.obj']);

x_diff = max(node_xyz(:,1))-min(node_xyz(:,1));
y_diff = max(node_xyz(:,3))-min(node_xyz(:,3));

about_size = (x_diff+y_diff)/2;

averge_number_object = sqrt(size(input_points,1));

%plane
x_enlarge = ones(size(input_points,1),1);
z_enlarge = ones(size(input_points,1),1); %%for different nodes
x_trans = input_points(:,1)*(6*averge_number_object*about_size);
y_trans = input_points(:,2)*(6*averge_number_object*about_size);

% height
y_enlarge = ones(size(input_points,1),1);

bottom = 0;

if example==1
    folder = ['rendering/',name,'_e'];
else
    folder = ['rendering/',name];
end

mkdir(folder)
for i = 1:size(input_file_name1,2)
copyfile([input_folder1,input_file_name1{i},'.mtl'], [folder,'/',input_file_name1{i},'.mtl']);
end
for i = 1:size(input_file_name2,2)
copyfile([input_folder2,input_file_name2{i},'.mtl'], [folder,'/',input_file_name2{i},'.mtl']);
end

for iobj = 1:size(input_points,1)
    
    output_file_name = [folder,'/',num2str(iobj),'.obj'];
    

    if abs(input_points(iobj,3)-0.2064)<0.01
        input_file_name = [input_folder1,input_file_name1{randi([1,size(input_file_name1,2)])},'.obj'];
    else
        input_file_name = [input_folder2,input_file_name2{randi([1,size(input_file_name2,2)])},'.obj'];
    end
%     
%     if ( output_file_unit < 0 )
%     fprintf ( 1, '\n' );
%     fprintf ( 1, 'OBJ_WRITE - Fatal error!\n' );
%     fprintf ( 1, '  Could not open the output file "%s".\n', ...
%         output_file_name );
%     return
%     end
    input_file_unit = fopen ( input_file_name, 'r' );
    output_file_unit = fopen ( output_file_name, 'wt' );
    node = 0;
    while ( 1 )


        text = fgetl ( input_file_unit );
        
        if ( text == -1 )
            break
        end
        
        s_control_blank ( text );
        
        done = 1;
        word_index = 0;
        %
        %  Read a word from the line.
        %
        [ word, done ] = word_next_read ( text, done );
        
        %
        %  If no more words in this line, read a new line.
        %
        if ( done )
            continue
        end
        %
        %  If this word begins with '#' or '$', then it's a comment.  Read a new line.
        %
        %     if ( word(1) == '#' || word(1) == '$' )
        %         continue
        %     end
        
        word_index = word_index + 1;
        
        if ( word_index == 1 )
            word_one = word;
        end
        
        if ( s_eqi ( word_one, 'V' ) )
            
            node = node + 1;
            
            for i = 1 : 3
                [ word, done ] = word_next_read ( text, done );
                temp = s_to_r8 ( word );
                node_xyz(i,node) = temp;
            end
            
            transformed_xyz = node_xyz(1:3,node);
            transformed_xyz(1) = transformed_xyz(1)*x_enlarge(iobj) + x_trans(iobj);
            transformed_xyz(2) = transformed_xyz(2)*z_enlarge(iobj) ;
            transformed_xyz(3) = transformed_xyz(3)*y_enlarge(iobj) + y_trans(iobj);
%             transformed_xyz(3) = transformed_xyz(3) - y_enlarge(iobj)*bottom;
            
            fprintf ( output_file_unit, 'v  %f  %f  %f  \n',  transformed_xyz );
            %             text_num = text_num + 1;
        elseif ( s_eqi ( word_one, 'G' ) )
            
            fprintf ( output_file_unit, [text,num2str(iobj),'\n' ]);
        elseif (s_eqi ( word_one, 'F' ))
            fprintf ( output_file_unit, [text,'\n' ]);
        else
            fprintf ( output_file_unit, [text,'\n' ]);
            
        end
    end
    
    fclose ( output_file_unit );
    fclose ( input_file_unit );
end
