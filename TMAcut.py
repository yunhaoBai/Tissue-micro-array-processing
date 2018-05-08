#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import cv2
# import numpy as np


def TMAcut(imagename, kernelsize=15, erodecycle=4):

    # transfer into gray scale
    image = cv2.imread(imagename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and binary image
    blurred = cv2.blur(gradient, (15, 15))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    # connect parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelsize, kernelsize))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # cycle of erode and dilate to connect parts
    closed = cv2.erode(closed, None, iterations=erodecycle)
    closed = cv2.dilate(closed, None, iterations=erodecycle)

    # use contours to get all the image of interest
    images, cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnts)):
        # compute the bounding box of contours, mind the orientation
        x, y, w, h = cv2.boundingRect(cnts[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # crop out the rectangle image and save as png
        cropImg = image[y:y + h, x:x + w]
        cv2.imwrite("cropImage"+str(i)+".png", cropImg)


def main():
    args = sys.argv[1:]
    if len(args) >= 1:
        print("Input format: python filename int(kernelsize, default=15) int(erodecycle,default=4)")
        print("bigger kernelsize may make TMA linked together."
              "larger erodecycle remove noisy dots.")
        TMAcut(args[0], args[1], args[2])


if __name__ == '__main__':
    main()
