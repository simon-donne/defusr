#!/bin/bash

set -e
. /is/sg/sdonne/anaconda3/etc/profile.d/conda.sh

export PATH="/is/software/nvidia/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/is/software/nvidia/cuda-9.0/lib64:/is/software/nvidia/cudnn-7.3-cu9.0/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/is/software/nvidia/cuda-9.0/lib64:/is/software/nvidia/cudnn-7.3-cu9.0/lib64:$LIBRARY_PATH"

MVSNET_FUSIBILE="/is/sg/sdonne/Desktop/research/MVSNet_fusibile/fusibile"
WORKPATH=${1:-"/tmp/sdonne/MVSNet_space/"}
base_data_folder=${3:-"/is/rg/avg/sdonne/data/flying_things_MVS_0.25/"}

for scn in $(ls -1d $base_data_folder/* | shuf)
do
    IMGPATH="$scn/"
    mkdir -p "$scn/mvsnet/"
    DEPTHPATH="$scn/mvsnet/depth/"
    PTPATH="$scn/mvsnet/cloud_total.ply"

    if [ -f $PTPATH ]
    then
        echo "$PTPATH already exists -- skipping"
        continue
    fi
    echo "Running $scn"

    rm -rf $WORKPATH

    conda activate ddf_pytorch
    mkdir -p $WORKPATH
    mkdir -p $WORKPATH/images
    mkdir -p $WORKPATH/cams
    python data/flying_things/_mvsnet_create_files.py $IMGPATH $WORKPATH

    cd /is/sg/sdonne/Desktop/research/MVSNet/mvsnet/
    . /is/sg/sdonne/anaconda3/etc/profile.d/conda.sh
    conda activate MVSNet
    python test.py --dense_folder $WORKPATH --max_w 1536 --max_h 864 &> /dev/null
    python depthfusion.py --dense_folder $WORKPATH --fusibile_exe_path $MVSNET_FUSIBILE &> /dev/null

    mkdir -p $WORKPATH/depth
    cd -
    conda activate ddf_pytorch
    python data/flying_things/_mvsnet_convert_results.py $WORKPATH $IMGPATH

    mv $WORKPATH/depth $DEPTHPATH
    mv $WORKPATH/points_mvsnet/consistencyCheck*/final3d_model.ply $PTPATH
done
