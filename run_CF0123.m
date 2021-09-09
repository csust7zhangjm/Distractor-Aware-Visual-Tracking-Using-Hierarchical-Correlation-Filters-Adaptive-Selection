function results = run_MCCT(seq, res_path, bSaveImage)

% RUN_CF2:
% process a sequence using CF2 (Correlation filter tracking with convolutional features)
%
% Input:
%     - seq:        sequence name
%     - res_path:   result path
%     - bSaveImage: flag for saving images
% Output:
%     - results: tracking results, position prediction over time
%
%   It is provided for educational/researrch purpose only.
%   If you find the software useful, please consider cite our paper.
%
%   Hierarchical Convolutional Features for Visual Tracking
%   Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang
%   IEEE International Conference on Computer Vision, ICCV 2015
%
% Contact:
%   Chao Ma (chaoma99@gmail.com), or
%   Jia-Bin Huang (jbhuang1@illinois.edu).

% ================================================================================
% Environment setting
% ================================================================================
addpath('./tracker');                  
addpath('./utility');
addpath('model','E:\benchmark_v3.0\trackers\MCCT\matconvnet');
addpath('E:\benchmark_v3.0\trackers\MCCT\TrackerMCCT\vlfeat-0.9.20');
vl_setupnn();
vl_compilenn('enableGpu',true);
% Image file names
img_files = seq.s_frames;
% Seletected target size
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
% Initial target position
pos       = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);
% lefttop = [seq.init_rect(1,1), seq.init_rect(1,2)];

% Extra area surrounding the target for including contexts
% padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);
params.visualization = 0;
params.visualization_dbg=0;
%% DCF related 
params.hog_cell_size = 4;
params.fixed_area = 200^2;                 % standard area to which we resize the target
params.n_bins = 2^5;                       % number of bins for the color histograms (bg and fg models)
params.lr_pwp_init = 0.01;                 % bg and fg color models learning rate 
params.inner_padding = 0.2;                % defines inner area used to sample colors from the foreground
params.output_sigma_factor = 0.1;          % standard deviation for the desired translation filter output 
params.lambda = 1e-4;                      % regularization weight
params.lr_cf_init = 0.01;                  % DCF learning rate
params.period = 5;                         % time period, \Delta t
params.update_thres = 0;                 % threshold for adaptive update
params.expertNum = 7; 

%% scale related
params.hog_scale_cell_size = 4;            % from DSST 
params.learning_rate_scale = 0.025;      
params.scale_sigma_factor = 1/2;
params.num_scales = 33;       
params.scale_model_factor = 1.0;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16;
show_visualization=false;

%% start trackerMain.m
im = imread([img_files{1}]);
% is a grayscale sequence ?
if(size(im,3)==1)
    params.grayscale_sequence = true;
end
if(size(im,3)==3)
    params.grayscale_sequence = false;
end
params.img_files = img_files;
 params.img_path = seq.path;
% init_pos is the centre of the initial bounding box
params.init_pos = pos;
params.target_sz = target_sz;
[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);
if params.visualization
    params.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
end
% ================================================================================
% Main entry function for visual tracking
% ================================================================================
[rects,time] = trackerMain_7expert_3conv_cent0_8_update(params, im, bg_area, fg_area, area_resize_factor);

% ================================================================================
% Return results to benchmark, in a workspace variable
% ================================================================================
results.type   = 'rect';
results.res    = rects.res;
results.fps    = numel(img_files)/time;

end

