#!/bin/bash

# this script takes every 1080p YUV file that has been generate by ./generate_base_videos.sh
# and resizes it to 480p and encodes it using H.265.
# This 480p video forms the content stream and is upsampled using either SRVC or bicubic upsampling.
# it needs the original (pre-YUV) file to compute the fps and this needs the ORIG_DIR as the third
# argument to find the original file

YUV_dir=$1
resize_dir=$2
orig_dir=$3 

for file in ${YUV_dir}/*;
do
    extname=$(basename -- $file )
    name=${extname%.*}
    
    # parameters
    orig_height=1080
    orig_width=1920
    height=270
    width=480
    
    mp4_name="${orig_dir}${name}*"
    fps=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 \
        -show_entries stream=r_frame_rate ${mp4_name})
    echo ${mp4_name}
    echo ${fps}
    

    for format in 265;
    do
        input_video=${file}
        echo ${input_video}
        output_prefix="${resize_dir}/h${format}/${name}_${width}x${height}"
        
        # crf range
        for crf in 10 20 25 30 35 40;
        do
            output_video=${output_prefix}_crf${crf}
            ffmpeg -y -f rawvideo -s ${orig_width}x${orig_height} -pix_fmt yuv420p -framerate ${fps} \
                -i ${input_video}  \
                -vf scale=${width}:${height} -sws_flags area -vcodec libx${format} -preset slow \
                -vsync 0 -pix_fmt yuv420p\
                -crf ${crf} -an ${output_video}.mp4
        done
    done
done

