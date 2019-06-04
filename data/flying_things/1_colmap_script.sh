#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
COLMAP=${1:-"/is/sg/sdonne/Desktop/research/colmap/build/src/exe/colmap"}
base_folder=${2:-"/scratch/sdonne/data/flying_things_MVS/"}
WORKPATH=${3:-"/tmp/sdonne/colmap_workspace/"}

export PYTHONPATH=$(pwd)

SCANPATHS=`( ls $base_folder/*/ -1d | shuf )`
for SCANPATH in $SCANPATHS
do
    start=`date +%s`
    data/flying_things/_colmap_single.sh $COLMAP $SCANPATH $WORKPATH
    end=`date +%s`
    runtime=$((end-start))
    echo "Finished $SCANPATH in $runtime s"
done
