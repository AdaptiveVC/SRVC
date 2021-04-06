# Efficient Video Compression via Content-Adaptive Super-Resolution

## Installation
For installing the required packages using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), use:
```
git clone https://github.com/AdaptiveVC/SRVC.git
cd SRVC
conda env create -f environment.yml
```

## Running SRVC
For running SRVC, check `python srvc.py --help`.

## Videos
### Vimeo Short Films
We use 28 short films with direct download buttons on Vimeo. These videos are in high-resolution, have realistic scene changes from movie-makers, and have 10min+ duration. 

Vimeo `video_id`s are located [here](./datasets/vimeo/vimeo_ids.txt), and you can view/download them at `vimeo.com/video_id`, e.g., [vimeo.com/441417334](https://vimeo.com/441417334). 

### Xiph Full Sequences
We use the following four long video sequences from the [Xiph](https://media.xiph.org/video/derf/) video dataset:
- [Big Buck Bunny](https://media.xiph.org/video/derf/y4m/big_buck_bunny_1080p24.y4m.xz)
- [Elephants Dream](https://media.xiph.org/video/derf/y4m/elephants_dream_1080p24.y4m.xz)
- [Sita Sings the Blues](https://media.xiph.org/video/derf/y4m/sita_sings_the_blues_1080p24.y4m.xz)
- [Meridian](https://media.xiph.org/video/derf/meridian/MERIDIAN_SHR_C_EN-XX_US-NR_51_LTRT_UHD_20160909_OV/)
