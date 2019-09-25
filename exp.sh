#!/bin/csh

set exemplars = (Tree1 Parterre5  Building1 Building2 Parterre3 Parterre7 Parterre9 Parterre10 Parterre11 Parterre12 random_yes circles_regular circles_irregular Building3f Building4f Parterre13 Fluid1 lines1 Fluid2 lines2 Parterre12f maze1 maze2 Ducks Cattles Goose Army circles_regular_small Fluid3) 
set kernel1 = (2 1 1 1 1 2 1 2 1 1 0.5 0.5 0.5 1 1 1 2 1 1 1 1 2 1 1 2 1 1 0.5 2)
set kernel2 = (4 3 4 3 3 4 3 4 3   2 2 2 2 4 4 3 4 3 3 3 3 4 3 4 4 4 4 2 3)


set j = 1
while ($j < 30)
   echo $exemplars[$j]	
   echo kernel1 = $kernel1[$j]
   echo kernel2 = $kernel2[$j]


python3 HierarchicalEndtoEndOptimizationSamplesPosition.py --kernel_sigma1 $kernel1[$j] --kernel_sigma2 $kernel2[$j] --exemplar_filename $exemplars[$j]

python3 SampleMergingOptimization.py --kernel_sigma1 $kernel1[$j] --kernel_sigma2 $kernel2[$j] --exemplar_filename $exemplars[$j] --output_dir results_final --input_dir results_final


matlab -nodesktop -nosplash -r "MergePoints('$exemplars[$j]','results_final','results_final','final');exit()"
   @ j++

end 


