import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im
from moviepy.editor import VideoFileClip


#lists to hold prev frames detected lines
prev_leftlines = []
prev_rightlines = []


#Perform color thresholding in HLS color space
def laneColors_mask(img,thresh=[(0,255),(0,255),(0,255)]):
    #Convert to hls color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    #Color masking using HLS color space

    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]

    #Create gray scale thresholded img
    binary_mask = np.zeros_like(s)

    binary_mask [((h>= thresh[0][0]) & (h<= thresh[0][1])) &\
                ((l >=thresh[1][0]) & (l<=thresh[1][1])) &\
                ((s >=thresh[2][0]) & (s<=thresh[2][1]))] = 1
    return binary_mask

#Canny edge detection + Gaussian smoothing filter
def canny_edge (grayImg,thresh=[0,255],kernel_size = 3):

    # Gaussian filter
    blur = cv2.GaussianBlur(grayImg,(kernel_size,kernel_size),0)

    #Canny Edge Detection
    return cv2.Canny(blur,thresh[0],thresh[1])

# Region masking to extract the region of interest
def regionMask (img,pts):

    region_mask = np.zeros_like(img)

    #Fill black image with the region of interest
    cv2.fillPoly(region_mask,np.array([[pts]]),255)

    return region_mask

# Draw lines into a given image
def drawLines (img,lines,color = 255, thickness = 10):

    #Check if list is empty
    if lines is not None:

        for line in lines:

            for x1,y1,x2,y2 in line:

                cv2.line(img,(x1,y1),(x2,y2),color ,thickness)
        return img

    else : #Nothing to draw
        return img

#Filter Horizontal lines
def filter_horizontal (lines , slope_thresh = 0):


    selected_lines = []
    slopes = []
    intercepts = []

    #Loop over each line in lines
    for line in lines:

        #Unpack the line  points
        x1,y1,x2,y2 = line[0]

        #Compute slope
        m = (y2-y1)/(x2-x1)

        #Filter Horizontal lines
        if abs(m) > slope_thresh:

            selected_lines.append([[x1,y1,x2,y2]])

    return selected_lines

# Sort lines into two grous (left laneline and right laneline)
def sortLines (lines):

    #Lists to hold the averaged lanes
    left_lanelines = []
    right_lanelines = []

    #Using -ve , +ve slopes
    for line in lines :
        for x1,y1,x2,y2 in line:

            #Sort according to the slope
            #Compute slope
            m = (y2-y1)/(x2-x1)
            if m < 0:
                left_lanelines.append([x1,y1,x2,y2])

            elif m >  0:
                right_lanelines.append([x1,y1,x2,y2])

    return [left_lanelines,right_lanelines]

# Filter the sorted lines
def Lanelines_filtering (lanelines,start_y,end_y):

    #Convert points into integers
    start_y = int(start_y)
    end_y = int(end_y)

    #List to hold the averaged lines
    avg_lines = []

    #Lists to carry the slopes and intercepts for each laneline
    slopes = []
    intercepts = []

    for laneline in lanelines :


        #Avg the lines by its slope and intercepts
        # for each line (y = mx +b )

        #Reset lists
        slopes.clear()
        intercepts.clear()

        #Loop over lines in each laneline
        for line  in laneline:

            #Unpack
            x1,y1,x2,y2 = np.array(line).reshape(4,1)

            #Compute slope and intercept

            if x1 != x2: #Avoid div by zero

                m = (y2-y1)/(x2-x1)
                b = y1 - (m*x1)

                slopes.append(m)
                intercepts.append(b)

        #lines were found
        if (len(slopes) > 0):

            #Average all similar lines
            avg_m = sum(slopes)/len(slopes)
            avg_b = sum(intercepts)/len(intercepts)

            #compute starting and ending points for each lane (x = (y-b)/m)
            start_x = int((start_y - avg_b)/avg_m)
            end_x = int((end_y - avg_b)/avg_m)
            avg_lines.append([[start_x,start_y,end_x,end_y]])

        else:
            avg_lines.append([[0,0,0,0]])


    return avg_lines

# The full pipline
def Lanelines_detection(image):


    # #Visualize Original Image if needed
    # plt.figure(1)
    # plt.imshow(image)
    # plt.title('Original Image')


    ####################
    ###1.Color masking##
    ####################

    colorMask = laneColors_mask(image,thresh=[(0,255),(40,100),(120,200)])

    # #Visualize and save results if needed
    # plt.figure(2)
    # plt.imshow(colorMask,cmap='gray')
    # plt.title('Colors Thresholded Image')
    #
    # #Save Image
    # plt.imsave('output_images/colorMask_img.jpg', colorMask,cmap='gray')

    ##############################
    ### 2. Canny Edge Detection###
    ##############################

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #Edge Detection (on the masked image)
    edge = canny_edge(gray,thresh=[50,150],kernel_size = 7)

    #Combine with color masking
    binary_edge = np.zeros_like(edge)
    binary_edge =cv2.bitwise_or(edge,colorMask)


    # #Visualize and save results if needed
    # plt.figure(3)
    # plt.imshow(binary_edge,cmap='gray')
    # plt.title('Color Thresholded Edge Detection')
    #
    # #Save Image
    # plt.imsave('output_images/colorThresh_edge.jpg', binary_edge ,cmap='gray')


    ############################
    #### 3. Region Masking #####
    ############################

    #Shape offsets
    width1 = 0.45
    width2 = 0.05
    region_height = 0.6

    #Define the regions four points
    pt1 = (int(image.shape[1]/2 - width1*image.shape[1]),int(image.shape[0]))
    pt4 = (int(image.shape[1]/2 + width1*image.shape[1]),int(image.shape[0]))
    pt2 = (int(image.shape[1]/2 - width2*image.shape[1]) ,int(region_height*image.shape[0]))
    pt3 = (int(image.shape[1]/2 + width2*image.shape[1]) ,int(region_height*image.shape[0]))


    region_mask = regionMask (binary_edge,[pt1,pt2,pt3,pt4])

    #Apply the mask to the edge detected image
    regionMasked_img = np.copy(binary_edge)
    regionMasked_img [region_mask == 0] = 0


    # #Visualize and save results if needed
    # plt.figure(4)
    # plt.imshow(regionMasked_img,cmap='gray')
    # plt.title('Region Masked Image')
    #
    # #Save Image
    # plt.imsave('output_images/regionMasked_img.jpg', regionMasked_img,cmap='gray')


    ############################
    #### 4. Lines Detection ####
    ############################

    #Hough Lines parameters
    rho = 2
    theta = np.pi/180
    threshold = 70
    min_len = 2
    max_gap = 50

    #Hough lines
    houghLines_img = np.zeros_like(regionMasked_img)
    lines = cv2.HoughLinesP (regionMasked_img, rho, theta, threshold, minLineLength = min_len, maxLineGap = max_gap)

    #####################################
    ### 4.1 Filter Horizontal lines #####
    #####################################
    lines = filter_horizontal (lines , slope_thresh =0.6)

    #Draw Hough Lines
    houghLines_img = drawLines(houghLines_img,lines,color = 255, thickness = 10)

    # # Visualize and save results if needed
    # plt.figure(5)
    # plt.imshow(houghLines_img,cmap='gray')
    # plt.title('Hough Lines')
    #
    # #Save Image
    # plt.imsave('output_images/HoughLines.jpg', hough Lines_img,cmap='gray')

    ########################
    #### 4.2 Sort Lines ####
    ########################

    #Sort lines into two lane
    sortedLines = sortLines(lines)


    ###################################################
    #### 4.3 Filter  Lines into two lanelines only ####
    ###################################################
    global prev_rightlines
    global prev_leftlines

    #Lanelines filtering
    start_y = image.shape[0]
    end_y = region_height*(1.1*image.shape[0])
    filteredLines = Lanelines_filtering(sortedLines,start_y,end_y)


    # Save the left and right lines
    prev_leftlines.append(filteredLines[0])
    prev_rightlines.append(filteredLines[1])



    ###########################################################
    #### 5. Smooth lanelines (in case of video processing) ####
    ##########################################################

    #Remove the oldest readings if the number frames exceeded N
    N  = 8


    if len(prev_leftlines ) > N :
        prev_leftlines.pop(0)
        prev_rightlines.pop(0)


    # Average the last detected lanelines together if previous frames are found
    avg_lanelines = Lanelines_filtering([prev_leftlines,prev_rightlines],start_y,end_y)


    #Draw filtered lines on empty image
    filteredLines_img = image*0
    filteredLines_img = drawLines(filteredLines_img,avg_lanelines,color = [0,0,255] , thickness = 20)


    # #Visualize and save results if needed
    # plt.figure(6)
    # plt.imshow(filteredLines_img)
    # plt.title('Filtered Lanes')
    #
    # #Save Image
    # plt.imsave('output_images/filteredLines_img.jpg', filteredLines_img)


    ##############################################
    ### 8. Draw lanelines on the oriignal image ###
    ##############################################

    #Combine lanelines with the original Image
    final_img = np.copy(image)
    final_img = cv2.addWeighted(image,0.8,filteredLines_img,1,0)

    # #Visualize and save results if needed
    # plt.figure(7)
    # plt.imshow(final_img)
    # plt.title('Final Result')
    #
    # #Save Image
    # plt.imsave('output_images/final_res.jpg', final_img)
    #
    # plt.show()

    return final_img




#################################
### Image lanelines detection ###
#################################

# Load image
#test_img = im.imread("test_images/challenge4.jpg")

# plt.figure(1)
# plt.imshow(test_img)
# plt.title('Original Image')


# Perform detection
#lanelines_img = Lanelines_detection(test_img)

# plt.figure(2)
# plt.imshow(lanelines_img)
# plt.title('Lanelines Detection')
# plt.show()


#################################
### Video lanelines detection ###
#################################

#Load video
in_clip = VideoFileClip('test_vids/challenge.mp4')

#Process video
out_clip = in_clip.fl_image(Lanelines_detection)

#Save the result video
out_clip.write_videofile( 'output_vids/new1.mp4', audio=False)
