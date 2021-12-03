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
        neighborhood = self.image[columnLocation - 1:columnLocation + 1, rowLocation - 1:rowLocation + 1]   # grab contents of neighborhood
        return np.std(neighborhood)                                                                         # return standard deviation of neighborhood

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
        for i in range(0, 8):                                                   # for each pixel in the neighborhood (except the center)
            distances[i] = np.linalg.norm(neighborhood[i] - center)             # calculate distance between center and pixel
        return max(distances)                                                   # return max distance

    # Calculates initial seed pixels for the entire image
    # Uses the given image
    # Returns None, just tweaks class variables
    def seedSelection(self) -> None:
        for row in range(1, self.rows - 1):                                             #
            for column in range(1, self.columns - 1):                                   #
                self.stdDevs[column, row] = self.stdDev(row, column)                    # calculate standard deviations
                self.distances[column, row] = self.euclideanDistance(row, column)       # calculate distances
        self.stdDevs = self.stdDevs/np.amax(self.stdDevs)                               # normalize standard deviations
        thresholdedStdDevs = cv.threshold(self.stdDevs, 0, 1, cv.THRESH_OTSU)           # threshold standard deviations
        thresholdedDistances = cv.threshold(self.distances, 0.05, 1, cv.THRESH_BINARY)  # check conditions
        self.seededPixels = thresholdedStdDevs & thresholdedDistances                   # generate seeded pixels mask and save


    def regionGrowing(self) -> None:
        pass

    def regionMerging(self) -> None:
        pass

    def main(self) -> None:
        self.imageConvert()     # Step 1: Complete
        self.seedSelection()    # Step 2: Complete (except for picking seed pixels, needs debugging)
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
