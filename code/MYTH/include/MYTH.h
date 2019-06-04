void InvertCams_gpu(
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs
);

void DepthReprojectionCompleteBound_updateOutput_gpu(
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs,
    float dmin,
    float dmax,
    float dstep
);

void DepthReprojectionNonzeroCompleteBound_updateOutput_gpu(
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs,
    float dmin,
    float dmax,
    float dstep
);

void DepthReprojectionNeighbours_updateOutput_gpu(
    THCudaTensor *input_depth,
    THCudaTensor *output_depth,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs
);

void DepthReprojectionNeighboursAlt_updateOutput_gpu(
    THCudaTensor *input_depth,
    THCudaTensor *output_depth,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs
);

void DepthColorAngleReprojectionNeighbours_updateOutput_gpu(
    THCudaTensor *input_depth,
    THCudaTensor *output_depth,
    THCudaTensor *input_color,
    THCudaTensor *output_color,
    THCudaTensor *output_angle,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs
);

void DepthColorAngleReprojectionNeighbours_updateGradInput_gpu(
    THCudaTensor *input_depth,
    THCudaTensor *output_depth,
    THCudaTensor *dloss_input_color,
    THCudaTensor *dloss_output_color,
    THCudaTensor *cameras,
    THCudaTensor *invKRs,
    THCudaTensor *camlocs
);

void AverageReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features
);

void AverageReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features
);

void MaximumReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features,
    THCudaTensor *input_mask
);

void MaximumReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features,
    THCudaTensor *input_mask
);

void MinimumAbsReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features,
    THCudaTensor *input_mask
);

void MinimumAbsReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features,
    THCudaTensor *input_mask
);

void VariationReprojectionPooling_updateOutput_gpu(
    THCudaTensor *avg_bound,
    THCudaTensor *avg_depth,
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *input_mask
);

void VariationReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *avg_bound,
    THCudaTensor *avg_depth,
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_avg_bound,
    THCudaTensor *dloss_input_avg_depth,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *input_mask
);

void PatchedReprojectionNeighbours_updateOutput_gpu(
    THCudaTensor *input_depths_neighbours,
    THCudaTensor *input_colors_neighbours,
    THCudaTensor *output_depth_reprojected,
    THCudaTensor *output_color_reprojected,
    THCudaTensor *camera_center,
    THCudaTensor *invKRs_neighbours,
    THCudaTensor *camlocs_neighbours
);

void PatchedReprojectionNeighbours_updateGradInput_gpu(
    THCudaTensor *input_depths_neighbours,
    THCudaTensor *output_depth_reprojected,
    THCudaTensor *dloss_dinput_colors_neighbours,
    THCudaTensor *dloss_doutput_color_reprojected,
    THCudaTensor *camera_center,
    THCudaTensor *invKRs_neighbours,
    THCudaTensor *camlocs_neighbours
);

void PatchedReprojectionCompleteBound_updateOutput_gpu(
    THCudaTensor *depth_center,
    THCudaTensor *depths_neighbours,
    THCudaTensor *bounds_neighbours,
    THCudaTensor *cameras_neighbours,
    THCudaTensor *invKR_center,
    THCudaTensor *camloc_center,
    float dmin,
    float dmax,
    float dstep
);

void PatchedAverageReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features
);

void PatchedAverageReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features
);

void PatchedMaximumReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features,
    THCudaTensor *input_mask
);

void PatchedMaximumReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features,
    THCudaTensor *input_mask
);

void PatchedMinimumAbsReprojectionPooling_updateOutput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *input_features,
    THCudaTensor *output_bound,
    THCudaTensor *output_depth,
    THCudaTensor *output_features,
    THCudaTensor *input_mask
);

void PatchedMinimumAbsReprojectionPooling_updateGradInput_gpu(
    THCudaTensor *input_bound,
    THCudaTensor *input_depth,
    THCudaTensor *dloss_input_bound,
    THCudaTensor *dloss_input_depth,
    THCudaTensor *dloss_input_features,
    THCudaTensor *dloss_output_bound,
    THCudaTensor *dloss_output_depth,
    THCudaTensor *dloss_output_features,
    THCudaTensor *input_mask
);

void bruteforce_sparsity_count_gpu(
    THCudaTensor *points,
    THCudaTensor *counts1,
    THCudaTensor *counts2,
    int N,
    float sparsity1,
    float sparsity2
);

void bruteforce_distance_gpu(
    THCudaTensor *points_from,
    THCudaTensor *points_to,
    THCudaTensor *dists,
    int N,
    int M
);