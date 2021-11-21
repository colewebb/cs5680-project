import cv2 as cv
import numpy as np
import os, sys
from matplotlib import pyplot
from tools import *

class segment():
    def __init__(self, imagePath) -> None:
        self.image = cv.imread(imagePath)
        self.columns = len(self.image)
        self.rows = len(self.image[0])
        self.stdDevs = np.zeros((self.columns, self.rows))
        self.distances = np.zeros((self.columns, self.rows))

    def imageConvert(self) -> None:
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2YCR_CB)

    def inverseImageConvert(self) -> None:
        self.image = cv.cvtColor(self.image, cv.COLOR_YCR_CB2BGR)
    
    def stdDev(self, rowLocation, columnLocation) -> float:
        if (rowLocation - 1 < 0) or (rowLocation + 1 > self.rows):
            return 0.0
        if (columnLocation - 1 < 0) or (columnLocation + 1 > self.columns):
            return 0.0
        yStdDev = 0
        cbStdDev = 0
        crStdDev = 0
        yMean = np.mean(self.image[columnLocation - 1:columnLocation + 1, rowLocation - 1:rowLocation + 1, 0])
        cbMean = np.mean(self.image[columnLocation - 1:columnLocation + 1, rowLocation - 1:rowLocation + 1, 1])
        crMean = np.mean(self.image[columnLocation - 1:columnLocation + 1, rowLocation - 1:rowLocation + 1, 2])
        for row in range(rowLocation - 1, rowLocation + 1):
            for column in range(columnLocation - 1, columnLocation + 1):
                yStdDev += (self.image[column, row] - yMean)
                cbStdDev += (self.image[column, row] - cbMean)
                crStdDev += (self.image[column, row] - crMean)
        yStdDev = np.sqrt(yStdDev * (1/9))
        cbStdDev = np.sqrt(cbStdDev * (1/9))
        crStdDev = np.sqrt(crStdDev * (1/9))
        stdDev = yStdDev + cbStdDev + crStdDev
        return stdDev

    def euclideanDistance(self, rowLocation, columnLocation) -> float:
        distances = [0, 0, 0, 0, 0, 0, 0, 0]
        center = self.image[columnLocation, rowLocation]
        neighborhood = [self.image[columnLocation - 1, rowLocation + 1],
                            self.image[columnLocation, rowLocation + 1],
                            self.image[columnLocation + 1, rowLocation + 1],
                            self.image[columnLocation - 1, rowLocation],
                            self.image[columnLocation + 1, rowLocation],
                            self.image[columnLocation - 1, rowLocation - 1],
                            self.image[columnLocation, rowLocation - 1],
                            self.image[columnLocation + 1, rowLocation - 1]]
        for i in range(0, 8):
            distances[i] = np.sqrt(((neighborhood[i, 0] - center[0])^2) + ((neighborhood[i, 1] - center[1])^2) + ((neighborhood[i, 2] - center[2])^2))/np.sqrt(neighborhood[i, 0]^2 + neighborhood[i, 1]^2 + neighborhood[i, 2]^2)
        return max(distances)

    def seedSelection(self) -> None:
        for row in range(1, self.rows - 1):
            for column in range(1, self.columns - 1):
                # standard deviations
                # distances
                # threshold standard deviations
                # check conditions
                # generate seeded pixels mask and save
                pass


    def regionGrowing(self) -> None:
        pass

    def regionMerging(self) -> None:
        pass

    def main(self) -> None:
        self.imageConvert()
        self.seedSelection()
        self.regionGrowing()
        self.regionMerging()

if __name__ == "__main__":
    imagePath = input(" image path >>> ")
    processor = segment(imagePath)
    if processor.image is None:
        print("This file doesn't exist. Exiting...")
        sys.exit(1)
    processor.main()
    processor.inverseImageConvert()
    pyplot.imshow(processor.image)
