#!/bin/bash

# takes videos from the input directory in mp4 format
# and saves the first 10 minutes of it in 1080p 420p YUV RAW format
# in the output directory

input_dir=$1 
output_dir=$2 
"/storage/scratch/yuvs/"
for file in ${input_dir}/*;
do
    extname=$(basename -- $file )
    name=${extname%.*}
    
    # parameters
    height=1080
    width=1920
    min=10

    output_video=${output_dir}${name}

    ffmpeg -y -i ${file} -ss 0 -t ${min}:00 \
        -vf scale=${width}:${height} \
        -c:v rawvideo -pix_fmt yuv420p -vsync 0 \
        ${output_video}.yuv
done

