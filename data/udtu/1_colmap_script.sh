#!/bin/bash
COLMAP="/is/sg/sdonne/Desktop/research/colmap/build/src/exe/colmap"
WORKPATH=${1:-"/tmp/sdonne/colmap_workspace/"}

array=$(seq 1 68 | shuf)

BASE_DATA_FOLDER="/is/rg/avg/sdonne/data"
PTPATH_BASE="$BASE_DATA_FOLDER/unrealDTU/Points/colmap/0.25"
mkdir -p $PTPATH_BASE

for SCANNR in ${array[@]}
do
    IMGPATH="$BASE_DATA_FOLDER/unrealDTU/Rectified_rescaled/0.25/scan$SCANNR/"
    OUTPATH_P="$BASE_DATA_FOLDER/unrealDTU/Colmap_Photometric_Depth/0.25/scan$SCANNR/"
    OUTPATH_G="$BASE_DATA_FOLDER/unrealDTU/Colmap_Geometric_Depth/0.25/scan$SCANNR/"
    PTPATH="$PTPATH_BASE/scan${SCANNR}_total.ply"

    if [ -f $PTPATH ]
    then
        echo "$PTPATH already exists -- skipping"
        continue
    fi
    echo ""

    # comment this to not remove the result every time
    rm -rf $WORKPATH

    MIN_DEPTH=4
    MAX_DEPTH=130

    mkdir -p $WORKPATH
    python datasets/dataset_procedures/udtu/make_udtu_files_colmap.py $WORKPATH
    mkdir -p $WORKPATH/sparse_known
    mkdir -p $WORKPATH/stereo
    mv $WORKPATH/udtu_cameras.txt $WORKPATH/sparse_known/cameras.txt
    mv $WORKPATH/udtu_points3D.txt $WORKPATH/sparse_known/points3D.txt
    mv $WORKPATH/udtu_images.txt $WORKPATH/sparse_known/images.txt

    echo "Scan $SCANNR undistortion"
    
    $COLMAP image_undistorter \
        --image_path $IMGPATH \
        --input_path  $WORKPATH/sparse_known/ \
        --output_path $WORKPATH \
        >& /dev/null

    mv $WORKPATH/udtu_patch-match.cfg $WORKPATH/stereo/patch-match.cfg
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
    python datasets/dataset_procedures/udtu/extract_depth.py "$WORKPATH/stereo/depth_maps/" photometric $OUTPATH_P
    python datasets/dataset_procedures/udtu/extract_depth.py "$WORKPATH/stereo/depth_maps/" geometric $OUTPATH_G
    mv $WORKPATH/fused.ply $PTPATH
done
