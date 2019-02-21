# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:31:48 2019

@author: Daniel
"""
'''reading and displaying images'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('messi.jpg', 0)

def display_image(image):
    cv.imshow('image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()    

display_image(img)
'''resizing images'''
big = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
display_image(big)

'''black and white images via thresholding'''
ret,thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
display_image(thresh)

'''adaptive thresholding can work better with uneven contrast images'''
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,15,2)
display_image(th2)

'''Canny Edge Detection'''
edges = cv.Canny(img,10,100)
display_image(edges)
'''thresholds matter'''
edges = cv.Canny(th2,100,200)
display_image(edges)

'''setting pixel values'''
print(img.item(100,100))
img.itemset((100,100),200)
print(img.item(100,100))

img = cv.imread('messi.jpg', 0)

'''drawing a diagonal line'''
for i in np.arange(50,200):
    img.itemset((i,i),200)
display_image(img)

'''hough line transform'''
sf = cv.imread('sf.jpg')
gray = cv.cvtColor(sf, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,170,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,200)

def plot_hough_line(image, hough_lines):
    for i in np.arange(len(hough_lines)):
        for rho,theta in hough_lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    display_image(image)
plot_hough_line(sf, lines)

'''change thresholding values'''
sf = cv.imread('sf.jpg')
gray = cv.cvtColor(sf, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,170,apertureSize = 3)
lines2 = cv.HoughLines(edges,1,np.pi/18,250)
plot_hough_line(sf, lines2)

'''probabalistic hough line transform'''
'''give it some criteria'''
sf = cv.imread('sf.jpg')
gray = cv.cvtColor(sf, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,170,apertureSize = 3)
minLineLength = 200
maxLineGap = 5
prob_hough_lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for line in prob_hough_lines:    
    for x1,y1,x2,y2 in line:
         cv.line(sf,(x1,y1),(x2,y2),(0,255,0),2)
display_image(sf)
         

'''Hough circle transform'''
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
display_image(cimg)

'''again, but with different thresholding values'''
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,30,
                            param1=50,param2=100,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
display_image(cimg)
'''still doesn't think the soccer ball is a circle?'''

'''machine learning k nearest neighbors'''
# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# Take Blue families and plot them
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

plt.show()

newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
plt.scatter(red[:,0],red[:,1],80,'r','^')
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 5)

print( "result: ", results,"\n")
print( "neighbours: ", neighbours,"\n")
print( "distance: ", dist)

plt.show()
