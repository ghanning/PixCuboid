#!/bin/bash
set -e
root_dir=$(dirname $(dirname $(realpath $0)))
for area in area_1  area_2  area_3  area_4  area_5a  area_5b  area_6
do
  src_dir=$root_dir/datasets/2d3ds/$area/persp/rgb/
  dst_path=$src_dir/line_segments.npz
  checkpoint_path=$root_dir/weights/deeplsd_md.tar
  python -m pixloc.pixlib.extract_line_segments --source $src_dir --destination $dst_path --checkpoint $checkpoint_path
done
