#!/bin/bash

# script for a single cluster node run
# it should run_render once, then run the colmap and gipuma scripts
# all of everything within its private subfolder for working directory etc
# afterwards, it copies the relevant files to a given directory (only when completely finished)
# this way, the final directory only contains validly finished dataset entries

# we have our cluster job index as first entry
#!/bin/bash

ID=${1:-0}
SID=${2:-0}

export PATH
echo "Initializing conda"
source ~/anaconda3/etc/profile.d/conda.sh
echo "Activating conda"
conda activate
echo "Activating the environment"
conda activate ddf_pytorch

export PYTHONPATH=$(pwd)

FINAL_OUTPUTPATH="/is/rg/avg/sdonne/data/flying_things_MVS_0.50/"
mkdir -p $FINAL_OUTPUTPATH
NODE_OUTPUTPATH="/tmp/sdonne/flying_out_${SID}_${ID}/"
mkdir -p $NODE_OUTPUTPATH

COLMAP="/is/sg/sdonne/Desktop/research/colmap/build/src/exe/colmap"
COLMAP_WORKPATH="/tmp/sdonne/colmap_workspace/"
GIPUMA_PATH="/is/sg/sdonne/Desktop/research/gipuma/"
FUSIBILE="/is/sg/sdonne/Desktop/research/fusibile/fusibile"

python data/flying_things/_run_render.py $NODE_OUTPUTPATH

mv $NODE_OUTPUTPATH/* $FINAL_OUTPUTPATH
rmdir $NODE_OUTPUTPATH
rm -rf $COLMAP_WORKPATH
