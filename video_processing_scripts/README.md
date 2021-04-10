# Scripts to process video files 

## Preparing dataset
### Generating YUV videos
The following script takes videos from an input directory `ORIG_DIR` in whatever size and format and converts them into 1080p videos saved in their 420p RAW YUV format into the directory `YUV_DIR` which is then used by the rest of the pipeline
```bash
mkdir -p YUV_DIR
./generate_base_videos.sh ORIG_DIR YUV_DIR
```

### Downsampling and resizing videos
The following command takes the YUV files generated above and uses area-based downsampling to convert it to 480p videos which are saved in the directory `RESIZE_DIR`. The input directory `ORIG_DIR` must be supplied because the original (pre-YUV) files are needed to extract fps information.
```bash
mkdir -p RESIZE_DIR
./resize.sh YUV_DIR RESIZE_DIR ORIG_DIR
```

## Generating H.264/H.265 baselines
The following command takes the YUV files generated above and reencodes it at the same 1080p size at different Compression Rate Factors (CRFs) using the standard H.264 and H.265 codecs to generate comparison points for PSNR vs. bits-per-pixel values. The reencoded videos are saved in the directory `REENCODE_DIR`. The input directory `ORIG_DIR` must be supplied because the original (pre-YUV) files are needed to extract fps information.
```bash
mkdir -p REENCODE_DIR
./resize.sh YUV_DIR REENCODE_DIR ORIG_DIR
```



