#!/bin/bash
set -e
root_dir=$(dirname $(dirname $(realpath $0)))
scannetpp_dir=$root_dir/datasets/scannetpp
mvc_dir=$root_dir/pixloc/pixlib/datasets/scannetpp
while read scene; do
  src_dir=$scannetpp_dir/data/$scene/dslr/undistorted_images
  dst_path=$src_dir/line_segments.npz
  checkpoint_path=$root_dir/weights/deeplsd_md.tar
  python -m pixloc.pixlib.extract_line_segments --source $src_dir --destination $dst_path --checkpoint $checkpoint_path
done < <(cat $mvc_dir/scenes_train.txt $mvc_dir/scenes_val.txt $mvc_dir/scenes_test.txt)
