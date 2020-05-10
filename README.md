# **Finding Lane Lines on the Road** 

---

## Overview

![result](https://user-images.githubusercontent.com/47195928/81497517-5161e380-92bf-11ea-9f25-0e0951489b5a.gif)

The goal of this project is to use OpenCV to detect lane lines in a video stream for different conditions using [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector) and [Hough Line Transform](https://en.wikipedia.org/wiki/Hough_transform).
As a part of of Udacity's 
[Self-Driving Cars ND](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

My pipeline consists of 5 main steps:

1. Color Masking (Based on while and yellow colors).
2. Gaussian filtering / smoothing and edge detection.
3. Selecting area of interest.
4. Performing Hough Line Transform.
5. Filtering lines per lane to maintain only one line per lane.



## Results

**Note** All the output files are included in the project you 

![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/Color_Mask.png) 
![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/Edge_Detection.png)
![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/Region_mask.png)
![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/Filtered_Edges.png)
![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/Line_Segements.JPG)
![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/ref_line.png)
![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/Filtered_lanes.png)
![](https://github.com/HossamKhalil-hub01/CarND-P1-LaneLinesDetection/blob/master/examples/final.png)


---

## 2. Potential Shortcomings my the current pipeline

The assumption of the whole lane filtering is based on the location of the camera relative to the scene, 
so one potential shortcoming would be what would happen when the cammera changes position which will mess up the lane filtering.

Another shortcoming could be the extreme exposure conditions, the detection becomes somewhat jittery.


## 3. Possible improvements

A possible improvement would be to use a real curve fitting techniques for example [numpy.polyfit](https://numpy.org/doc/1.18/reference/generated/numpy.polyfit.html) function can be used to fit each lane in one line
Another potential improvement could be to improve the filtering stages to be more suitable for more real life cases.
