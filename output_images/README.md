## Overview

This folder contains the results of each image in [test_images](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/tree/master/test_images) folder after being processed by the pipeline.

For each test image you can find a total of six images:

* colorMask_img: The result of the color thresholding step
* colorThresh_edge: The combination between edge detection and color thresholding.
* regionMasked_img: Region of interest extraction.
* HoughLines: The detected lines (filtered horizontally).
* filteredLines_img: The filtered finals lines.
* final_res: the result of combining lines with the original image.


**Note:** If you uncommented the visualization section in `Lanelines_detection()` function found in [Lanelines_Detection.py](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/Lanelines_Detection.py) script, the resulting images will be automatically saved in this directory
