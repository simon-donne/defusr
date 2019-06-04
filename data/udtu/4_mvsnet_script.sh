#!/bin/bash

set -e
. /is/sg/sdonne/anaconda3/etc/profile.d/conda.sh

export PATH="/is/software/nvidia/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/is/software/nvidia/cuda-9.0/lib64:/is/software/nvidia/cudnn-7.3-cu9.0/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/is/software/nvidia/cuda-9.0/lib64:/is/software/nvidia/cudnn-7.3-cu9.0/lib64:$LIBRARY_PATH"

MVSNET_FUSIBILE="/is/sg/sdonne/Desktop/research/MVSNet_fusibile/fusibile"
WORKPATH=${1:-"/tmp/sdonne/MVSNet_space/"}
CAMPATH="/is/rg/avg/sdonne/data/unrealDTU/Calibration/cal18"

array=$(seq 1 68 | shuf)

PTPATH_BASE="/is/rg/avg/sdonne/data/unrealDTU/Points/MVSNet/0.25"
mkdir -p PTPATH_BASE

for SCANNR in ${array[@]}
do
    IMGPATH="/is/rg/avg/sdonne/data/unrealDTU/Rectified_rescaled/0.25/scan$SCANNR/"
    DEPTHPATH="/is/rg/avg/sdonne/data/unrealDTU/MVSNet/Depth/0.25/scan$SCANNR/"
    PTPATH="$PTPATH_BASE/scan${SCANNR}_total.ply"

    if [ -f $PTPATH ]
    then
        echo "$PTPATH already exists -- skipping"
        continue
    fi
    echo "Running scan $SCANNR"

    rm -rf $WORKPATH

    conda activate ddf_pytorch
    mkdir -p $WORKPATH
    mkdir -p $WORKPATH/images
    mkdir -p $WORKPATH/cams
    python datasets/dataset_procedures/udtu/mvsnet_create_files.py $IMGPATH $CAMPATH $WORKPATH

    cd /is/sg/sdonne/Desktop/research/MVSNet/mvsnet/
    . /is/sg/sdonne/anaconda3/etc/profile.d/conda.sh
    conda activate MVSNet
    python test.py --dense_folder $WORKPATH &> /dev/null
    python depthfusion.py --dense_folder $WORKPATH --fusibile_exe_path $MVSNET_FUSIBILE &> /dev/null

    mkdir -p $WORKPATH/depth
    cd -
    conda activate ddf_pytorch
    python datasets/dataset_procedures/udtu/mvsnet_convert_results.py $WORKPATH

    mv $WORKPATH/depth $DEPTHPATH
    mv $WORKPATH/points_mvsnet/consistencyCheck*/final3d_model.ply $PTPATH
done
