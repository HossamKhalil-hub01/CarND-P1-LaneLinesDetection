# **Finding Lane Lines on the Road** 

---

## Overview

The goal of this project is to use OpenCV to detect lane lines in a video stream for different conditions using [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector) and [Hough Line Transform](https://en.wikipedia.org/wiki/Hough_transform).

My pipeline consists of 5 main steps:


1. Color Masking (Based on while and yellow colors).
2. Gaussian filtering / smoothing and edge detection.
3. Selecting area of interest.
4. Performing Hough Line Transform.
5. Filtering lines per lane to maintain only one line per lane.


## In-Depth

### 1.Color Masking:

The first step is to perform color masking to filter out most of irrelevant data depending on the colors of the lane lines which is specially usefull for the final challenge.

[image1]: ./examples/Color_Mask.png  "Color Mask"


### 2. Gaussian filtering and edge detection:

Now it's much cleaner to perform edge detection on.
 
[image2]: ./examples/Edge_Detection.png "Edge Detection"


**Note** There is no need to bother tunning for the noise on the top of the image as it will be out of the area of interest anyways.


### 3. Selecting Area of Interest:
Using the [cv2.fillpoly](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html) funciton it's possible to cover up all the area needed, the parameters are not specified as hard numbers as it will vary depending on the image,
but rather it's a function of the image shape to be more robust.


Assuming that the camera is centered on the vehicle.

[image3]: ./examples/Region_mask.png "Region Mask"

and here is the result after applying the region mask to the edge detection.


[image4]: ./examples/Filtered_Edges.png "Filtered Edges"


### 4.Hough Line Transform:

The next step is to find the lines using Hough Line Transform and with that we can only draw lines segements as shown.



[image5]: ./examples/Line_Segements.JPG "Line Segements"


### 5. Filtering lines

My simple solution to approach this problem is :

* First draw a ref line which is a line in the middle of image parralel to the x-axis that divides the image into two regions with a dead zone on each side to prevent overlapping.

 
[image6]: ./examples/ref_line.png "Reference line"


* There are two clusters of lines for each region (lane) after dividing the lines into these two groups i can just use any metho for the fitting problem.

* For each cluster I computed the lowest and the highest points and considered them as the begining and the end of the lane.


**Results**


[image7]: ./examples/Filtered_lanes.png "Filtered Lanes"


[image8]: ./examples/final.png "Final Result"


---

## 2. Potential Shortcomings my your current pipeline

As I stated above the assumption of the whole lane filtering is based on the location of the camera relative to the scene, 
so one potential shortcoming would be what would happen when the cammera changes position which will mess up the lane filtering.

Another shortcoming could be the extreme exposure conditions, as in the challenge condition the detection became somewhat jittery.



### 3. Possible improvements

A possible improvement would be to ...
A possible improvement would be to use a real curve fitting techniques for example [numpy.polyfit](https://numpy.org/doc/1.18/reference/generated/numpy.polyfit.html) function can be used to fit each lane in one line
Another potential improvement could be to improve the filtering stages to be more suitable for more real life cases.
