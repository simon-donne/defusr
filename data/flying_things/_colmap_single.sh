COLMAP=$1
SCANPATH=$2
WORKPATH=$3

set -e

PTPATH="$SCANPATH/colmap/"
if [ -f $PTPATH/cloud_total.ply ]
then
    echo "$PTPATH already exists -- skipping"
    exit 0
fi

rm -rf $WORKPATH

# copy all the input images to a separate folder
IMGPATH="$WORKPATH/tmp_imgs/"
rm -rf $IMGPATH
mkdir -p $IMGPATH
cp $SCANPATH/*.png $IMGPATH
rm -f $IMG_PATH/*_depth.png

OUTPATH_P="$SCANPATH/colmap/photometric/"
OUTPATH_G="$SCANPATH/colmap/geometric/"

MIN_DEPTH=1
MAX_DEPTH=10

python data/flying_things/_make_ft_files_colmap.py "$SCANPATH"
mkdir -p $WORKPATH/sparse_known
mkdir -p $WORKPATH/stereo
mv $SCANPATH/ft_cameras.txt $WORKPATH/sparse_known/cameras.txt
mv $SCANPATH/ft_points3D.txt $WORKPATH/sparse_known/points3D.txt
mv $SCANPATH/ft_images.txt $WORKPATH/sparse_known/images.txt

echo "$SCANPATH undistortion"

$COLMAP image_undistorter \
    --image_path $IMGPATH \
    --input_path  $WORKPATH/sparse_known/ \
    --output_path $WORKPATH \
    >& /dev/null

mv $SCANPATH/ft_patch-match.cfg $WORKPATH/stereo/patch-match.cfg
rm -rf $WORKPATH/sparse
mv $WORKPATH/sparse_known/ $WORKPATH/sparse/

echo "$SCANPATH stereo"

$COLMAP patch_match_stereo \
    --workspace_path $WORKPATH \
    --PatchMatchStereo.depth_min $MIN_DEPTH \
    --PatchMatchStereo.depth_max $MAX_DEPTH \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.filter true \
    &> /dev/null

echo "$SCANPATH fusion"

$COLMAP stereo_fusion \
    --workspace_path $WORKPATH \
    --output_path $WORKPATH/fused.ply \
    --input_type geometric \
    --StereoFusion.min_num_pixels 2 \
    --StereoFusion.max_depth_error 0.01 \
    --StereoFusion.max_normal_error 30 \
    &> /dev/null

echo "$SCANPATH cleanup"

mkdir -p $OUTPATH_P/depth
mkdir -p $OUTPATH_G/depth
mkdir -p $OUTPATH_P/normals
mkdir -p $OUTPATH_G/normals
python data/flying_things/_colmap_extract_depth.py "$WORKPATH/stereo/" $SCANPATH photometric $OUTPATH_P
python data/flying_things/_colmap_extract_depth.py "$WORKPATH/stereo/" $SCANPATH geometric $OUTPATH_G
mkdir -p $PTPATH
cp $WORKPATH/fused.ply $PTPATH/cloud_total.ply