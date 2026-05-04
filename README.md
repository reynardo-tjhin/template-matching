# COMP3419: Graphics and Multimedia Lab 10 Template Matching

##  Description

What motivated me was it took me 25+ minutes to get the SSD Map and the SSD Map I got was wrong.
Hence, I wanted to code something that can improve the time so that I can receive the feedback faster.
Recently, I learnt about CUDA and GPUs and would like to use them to make the code faster by using parallel programming.

Final image generated using 2 files (`COMP3419-W10-Lab-Template-Matching.py` and `ssd.cu`)

`COMP3419-W10-Lab-Template-Matching.py`: 
1. Breakdown the image into numbers (0 and 255)
2. Create a file to write the image data into (`template` and `input_image`)
3. Run the `ssd.cu` file to get the ssd_map
4. Get the ssd_map data from `ssd_map`
5. Keep finding the next best possible image with NMS
6. Output the image with the help of matplotlib

`ssd.cu`: A file to generate SSD Map with the help of external files.
- Used RTX 3050 Ti GPU
- nvcc, version: 10.1.243
- number of threads required: 665 per block
- number of blocks required:  946 per grid

NVIDIA-SMI 560.28.03              Driver Version: 560.28.03      CUDA Version: 12.6
NVIDIA Corporation GA107BM [GeForce RTX 3050 Ti Mobile]

Operating System:
- Ubuntu, Version: "20.04.5 LTS (Focal Fossa)"
