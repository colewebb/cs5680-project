# Automatic seeded region growing for color image segmentation
# Frank Y. Shih, Shouxian Cheng
# doi:10.1016/j.mavis.2005.05.015
# Implemented in Python by Cole Webb

import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot
from tools import *

class segment():
    def __init__(self, imagePath) -> None:
        self.image = cv.imread(imagePath)                       # open image
        self.columns = len(self.image)                          # get column count
        self.rows = len(self.image[0])                          # get row count
        self.stdDevs = np.zeros((self.columns, self.rows))      # initialize standard deviations array
        self.distances = np.zeros((self.columns, self.rows))    # initialize euclidean distances array
        self.seededPixels = np.zeros((self.columns, self.rows)) # initialize seed pixels array

    def imageConvert(self) -> None:
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2YCR_CB)

    def inverseImageConvert(self) -> None:
        self.image = cv.cvtColor(self.image, cv.COLOR_YCR_CB2BGR)
    
    # Calculates the standard deviation of all three channels in a 3x3 area
    # Takes the location of the center of the area of interest
    # Returns a float
    def stdDev(self, rowLocation, columnLocation) -> float:
        if (rowLocation - 1 < 0) or (rowLocation + 1 > self.rows):                                                  # if we're on an edge
            return 0.0                                                                                              # return zero
        if (columnLocation - 1 < 0) or (columnLocation + 1 > self.columns):                                         #
            return 0.0                                                                                              #
        yStdDev = 0                                                                                                 # set initial standard deviations for all three channels
        cbStdDev = 0                                                                                                #
        crStdDev = 0                                                                                                #
        yMean = np.mean(self.image[columnLocation - 1:columnLocation + 1, rowLocation - 1:rowLocation + 1, 0])      # calculate area means for all three channels
        cbMean = np.mean(self.image[columnLocation - 1:columnLocation + 1, rowLocation - 1:rowLocation + 1, 1])     #
        crMean = np.mean(self.image[columnLocation - 1:columnLocation + 1, rowLocation - 1:rowLocation + 1, 2])     #
        for row in range(rowLocation - 1, rowLocation + 1):                                                         # for each row in the neighborhood
            for column in range(columnLocation - 1, columnLocation + 1):                                            # for each column in the neighborhood
                yStdDev += (self.image[column, row] - yMean)                                                        # update standard deviation sums
                cbStdDev += (self.image[column, row] - cbMean)                                                      #
                crStdDev += (self.image[column, row] - crMean)                                                      #
        yStdDev = np.sqrt(yStdDev * (1/9))                                                                          # divide by 9 and take the square root
        cbStdDev = np.sqrt(cbStdDev * (1/9))                                                                        #
        crStdDev = np.sqrt(crStdDev * (1/9))                                                                        #
        stdDev = yStdDev + cbStdDev + crStdDev                                                                      # sum
        return stdDev                                                                                               # return

    # Calculates the euclidean distance from a pixel to its eight neighbors
    # Takes the location of interest in the original image
    # Returns a float
    def euclideanDistance(self, rowLocation, columnLocation) -> float:
        distances = [0, 0, 0, 0, 0, 0, 0, 0]                                    # initialize distances
        center = self.image[columnLocation, rowLocation]                        # get center
        neighborhood = [self.image[columnLocation - 1, rowLocation + 1],        # get contents of neighborhood, stuff them in an array
                            self.image[columnLocation, rowLocation + 1],        #
                            self.image[columnLocation + 1, rowLocation + 1],    #
                            self.image[columnLocation - 1, rowLocation],        #
                            self.image[columnLocation + 1, rowLocation],        #
                            self.image[columnLocation - 1, rowLocation - 1],    #
                            self.image[columnLocation, rowLocation - 1],        #
                            self.image[columnLocation + 1, rowLocation - 1]]    #
        for i in range(0, 8):
            # The below equation is messy. Here's a nicer version:
            #
            #            sqrt(((pixel[0] - center[0])^2) + ((pixel[1] - center[1])^2) + ((pixel[2] - center[2])^2))
            # distance = ------------------------------------------------------------------------------------------
            #                                 sqrt(pixel[0]^2 + pixel[1]^2 + pixel[2]^2)
            #
            distances[i] = np.sqrt(((neighborhood[i, 0] - center[0])^2) + ((neighborhood[i, 1] - center[1])^2) + ((neighborhood[i, 2] - center[2])^2))/np.sqrt(neighborhood[i, 0]^2 + neighborhood[i, 1]^2 + neighborhood[i, 2]^2)
        return max(distances)

    # Calculates initial seed pixels for the entire image
    # Uses the given image
    # Returns None, just tweaks class variables
    def seedSelection(self) -> None:
        for row in range(1, self.rows - 1):                                             #
            for column in range(1, self.columns - 1):                                   #
                self.stdDevs[column, row] = self.stdDev(row, column)                    # calculate standard deviations
                self.distances[column, row] = self.euclideanDistance(row, column)       # calculate distances
        self.stdDevs = self.stdDevs/max(self.stdDevs)                                   # normalize standard deviations
        thresholdedStdDevs = cv.threshold(self.stdDevs, 0, 1, cv.THRESH_OTSU)           # threshold standard deviations
        thresholdedDistances = cv.threshold(self.distances, 0.05, 1, cv.THRESH_BINARY)  # check conditions
        self.seededPixels = thresholdedStdDevs & thresholdedDistances                   # generate seeded pixels mask and save


    def regionGrowing(self) -> None:
        pass

    def regionMerging(self) -> None:
        pass

    def main(self) -> None:
        self.imageConvert()     # Step 1: Complete
        self.seedSelection()    # Step 2: Complete (needs testing)
        self.regionGrowing()    # Step 3: Planned
        self.regionMerging()    # Step 4: Planned

if __name__ == "__main__":
    imagePath = input(" image path >>> ")
    processor = segment(imagePath)
    if processor.image is None:
        print("This file doesn't exist. Exiting...")
        sys.exit(1)
    processor.main()
    processor.inverseImageConvert()
    pyplot.imshow(processor.image)
