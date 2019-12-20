#!/bin/bash
COLMAP="/is/sg/sdonne/Desktop/research/colmap/build/src/exe/colmap"
WORKPATH=${1:-"/tmp/sdonne/colmap_workspace/"}

array1=$(seq 1 77 | shuf)
array2=$(seq 82 128 | shuf)
array=("${array1[@]}" "${array2[@]}")

BASE_DATA_FOLDER="/is/rg/avg/sdonne/data"

PTPATH_BASE="$BASE_DATA_FOLDER/dtu/Points/colmap/0.25/"
mkdir -p $PTPATH_BASE

for SCANNR in ${array[@]}
do
    IMGPATH="$BASE_DATA_FOLDER/dtu/Rectified_rescaled/0.25/scan$SCANNR/"
    OUTPATH_P="$BASE_DATA_FOLDER/dtu/colmap/photometric/depth/Depth/0.25/scan$SCANNR/"
    OUTPATH_G="$BASE_DATA_FOLDER/dtu/colmap/geometric/depth/Depth/0.25/scan$SCANNR/"
    PTPATH="$PTPATH_BASE/scan${SCANNR}_total.ply"

    if [ -f $PTPATH ]
    then
        echo "$PTPATH already exists -- skipping"
        continue
    fi
    echo ""

    # comment this to not remove the result every time
    rm -rf $WORKPATH

    MIN_DEPTH=40
    MAX_DEPTH=1000

    mkdir -p $WORKPATH
    python data/dtu/_make_dtu_files_colmap.py $WORKPATH
    mkdir -p $WORKPATH/sparse_known
    mkdir -p $WORKPATH/stereo
    mv $WORKPATH/dtu_cameras.txt $WORKPATH/sparse_known/cameras.txt
    mv $WORKPATH/dtu_points3D.txt $WORKPATH/sparse_known/points3D.txt
    mv $WORKPATH/dtu_images.txt $WORKPATH/sparse_known/images.txt

    echo "Scan $SCANNR undistortion"
    
    $COLMAP image_undistorter \
        --image_path $IMGPATH \
        --input_path  $WORKPATH/sparse_known/ \
        --output_path $WORKPATH \
        >& /dev/null

    mv $WORKPATH/dtu_patch-match.cfg $WORKPATH/stereo/patch-match.cfg
    rm -rf $WORKPATH/sparse
    mv $WORKPATH/sparse_known/ $WORKPATH/sparse/

    echo "Scan $SCANNR stereo"
    
    $COLMAP patch_match_stereo \
        --workspace_path $WORKPATH \
        --PatchMatchStereo.depth_min $MIN_DEPTH \
        --PatchMatchStereo.depth_max $MAX_DEPTH \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.filter true \
        >& /dev/null

    echo "Scan $SCANNR fusion"
    
    $COLMAP stereo_fusion \
        --workspace_path $WORKPATH \
        --output_path $WORKPATH/fused.ply \
        --input_type geometric \
        >& /dev/null

    echo "Scan $SCANNR cleanup"
    
    mkdir -p $OUTPATH_P
    mkdir -p $OUTPATH_G
    python data/dtu/2_colmap_depth_to_npy.py "$WORKPATH/stereo/depth_maps/" photometric $OUTPATH_P
    python data/dtu/2_colmap_depth_to_npy.py "$WORKPATH/stereo/depth_maps/" geometric $OUTPATH_G
    mv $WORKPATH/fused.ply $PTPATH
done
