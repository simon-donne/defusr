#!/bin/bash

set -e

export PATH="/is/software/nvidia/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/is/software/nvidia/cuda-9.0/lib64:/is/software/nvidia/cudnn-7.3-cu9.0/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/is/software/nvidia/cuda-9.0/lib64:/is/software/nvidia/cudnn-7.3-cu9.0/lib64:$LIBRARY_PATH"

BASE_DATA_FOLDER="/is/rg/avg/sdonne/data"
MVSNET_FUSIBILE="/is/sg/sdonne/Desktop/research/MVSNet_fusibile/fusibile"
WORKPATH=${1:-"/tmp/sdonne/MVSNet_space/"}
CAMPATH="$BASE_DATA_FOLDER/dtu/Calibration/cal18"

array1=$(seq 1 77 | shuf)
array2=$(seq 82 128 | shuf)
array=("${array1[@]}" "${array2[@]}")

PTPATH_BASE="$BASE_DATA_FOLDER/dtu/Points/MVSNet/0.25"
mkdir -p PTPATH_BASE

. /home/sdonne/anaconda3/etc/profile.d/conda.sh

for SCANNR in ${array[@]}
do
    IMGPATH="$BASE_DATA_FOLDER/dtu/Rectified/scan$SCANNR/"
    DEPTHPATH="$BASE_DATA_FOLDER/dtu/MVSNet/Depth/0.25/scan$SCANNR/"
    PTPATH="$PTPATH_BASE/scan${SCANNR}_total.ply"

    if [ -f $PTPATH ]
    then
        echo "$PTPATH already exists -- skipping"
        continue
    fi
    echo "Running scan $SCANNR"

    rm -rf $WORKPATH


    conda activate defusr
    mkdir -p $WORKPATH
    mkdir -p $WORKPATH/images
    mkdir -p $WORKPATH/cams
    python data/dtu/_mvsnet_create_files.py $IMGPATH $CAMPATH $WORKPATH

    conda activate MVSNet
    cd /home/sdonne/Desktop/defusr_test/MVSNet/mvsnet/
    python test.py --dense_folder $WORKPATH &> /dev/null
    python depthfusion.py --dense_folder $WORKPATH --fusibile_exe_path $MVSNET_FUSIBILE &> /dev/null

    conda activate defusr
    mkdir -p $WORKPATH/depth
    cd -
    python data/dtu/4_mvsnet_convert_results.py $WORKPATH

    mv $WORKPATH/depth $DEPTHPATH
    mv $WORKPATH/points_mvsnet/consistencyCheck*/final3d_model.ply $PTPATH
done
