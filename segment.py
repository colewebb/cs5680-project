# Automatic seeded region growing for color image segmentation
# Frank Y. Shih, Shouxian Cheng
# doi:10.1016/j.mavis.2005.05.015
# Implemented in Python by Cole Webb

import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot
from tools import *
import pandas as pd

class segment():
    def __init__(self, imagePath) -> None:
        self.image = cv.imread(imagePath)                       # open image
        self.columns = len(self.image)                          # get column count
        self.rows = len(self.image[0])                          # get row count
        self.stdDevs = np.zeros((self.columns, self.rows))      # initialize standard deviations array
        self.distances = np.zeros((self.columns, self.rows))    # initialize euclidean distances array
        self.similarities = np.zeros((self.columns, self.rows)) # initialize similarities array
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
        for row in range(1, self.rows - 1):                                                                                                 #
            for column in range(1, self.columns - 1):                                                                                       #
                self.stdDevs[column, row] = self.stdDev(row, column)                                                                        # calculate standard deviations
                self.distances[column, row] = self.euclideanDistance(row, column)                                                           # calculate distances
        self.stdDevs = (self.stdDevs/np.amax(self.stdDevs)) * 255                                                                           # normalize standard deviations
        self.stdDevs = self.stdDevs.astype(np.uint8)                                                                                        #
        self.similarities = 255 - self.stdDevs                                                                                              # calculate similarities
        self.distances = (self.distances/np.amax(self.distances)) * 255                                                                     # normalize distances
        self.distances = self.distances.astype(np.uint8)                                                                                    # 
        self.similarities = cv.threshold(self.similarities, 0, 1, cv.THRESH_OTSU)[1]                                                        # threshold standard deviations
        self.distances = cv.threshold(self.distances, 13, 1, cv.THRESH_BINARY)[1]                                                           # check conditions
        for row in range(1, self.rows - 1):                                                                                                 # 
            for column in range(1, self.columns - 1):                                                                                       # 
                np.put(self.seededPixels, [column, row], float(bool(self.distances[column, row]) & bool(self.similarities[column, row])))   # check conditions for seed pixels

    # Grows the seeds
    # Takes no inputs
    # Returns None, just tweaks class variables
    def regionGrowing(self) -> None:
        seedCount, self.seededPixels, stats, centroids = cv.connectedComponentsWithStats(self.seededPixels, 8)          # label all seeds (using connected components)
        self.distances = self.distances/255                                                                             # return distances to range 0..1
        kernel = [[0, 1, 0],                                                                                            # create kernel for hit operation
                    [1, 0, 1],                                                                                          #
                    [0, 1, 0]]                                                                                          #
        seedNeighbors = np.logical_and(cv.dilate(self.seededPixels, kernel), self.seededPixels)                         # find neighbors of the seeds
        t = pd.DataFrame(columns=['row', 'column', 'distance'])                                                         # stuff all of the neighbors and their distances into a pandas DataFrame
        for row in range(0, self.rows):                                                                                 #
            for column in range(0, self.columns):                                                                       #
                if seedNeighbors[column, row] != 0:                                                                     #
                    t.iloc[len(t)] = [row, column, self.distances[column, row]]                                         #
        t.sort_values('distance')                                                                                       # sort the dataframe on distances
        seededRegionMeans = []                                                                                          # initialize array for means of seeds
        for i in range(1, seedCount):                                                                                   # calculate means of each seeded region
            seededRegionMeans[i] = [np.mean(self.image[self.seededPixels == i, 0]),                                     #
                np.mean(self.image[self.seededPixels == i, 1]),                                                         #
                np.mean(self.image[self.seeded == i, 2])]                                                               #
        while t.empty is False:                                                                                         # while the list of seed neighbors is not empty
            p = t[0]                                                                                                    # grab least distance member
            t = t[1:, :]                                                                                                # delete least distance member from dataframe
            row = p[0]                                                                                                  # set row
            column = p[1]                                                                                               # set column
            neighborhood = [self.seededPixels[column, row - 1],                                                         # find regions of pixel neighborhood
                self.seededPixels[column - 1, row],                                                                     #
                self.seededPixels[column + 1, row],                                                                     #
                self.seededPixels[column, row + 1]]                                                                     #
            minFoundDistance = 2
            minFoundRegion = 0
            if (neighborhood[0] == neighborhood[1] == neighborhood[2] == neighborhood[3]) and neighborhood[0] != 0:     # if all four neighbors are in the same region, and that region isn't the background
                self.seededPixels[column, row] = neighborhood[0]                                                        # set that pixel to that region
                i = neighborhood[0]                                                                                     #
                seededRegionMeans[i] = [np.mean(self.image[self.seededPixels == i, 0]),                                 # update mean
                    np.mean(self.image[self.seededPixels == i, 1]),                                                     #
                    np.mean(self.image[self.seeded == i, 2])]                                                           #
            else:                                                                                                       # if the four neighbors aren't all in the same region
                for i in [[column, row - 1],                                                                            # for each pixel in the 4-neighborhood
                            [column - 1, row],                                                                          #
                            [column + 1, row],                                                                          #
                            [column, row + 1]]:                                                                         #
                    if i != 0:                                                                                          # if it's in a region
                        distance = np.linalg.norm(self.image[i] - self.seededPixels[i])                                 # calculate distance using seededRegionMeans
                        if distance < minFoundDistance:                                                                 # if that distance is less than the lowest found distance
                            minFoundRegion = self.seededPixels[i]                                                       # plan to merge the current pixel with that region
                self.seededPixels[column, row] = minFoundRegion                                                         # 
            if not ((t['row'] == row - 1) & t['column'] == column).any():                                               # if the neighbors aren't already in the sorted list, add them
                t[len(t)] = [row - 1, column, self.distances[column, row - 1]]                                          #
            if not ((t['row'] == row) & t['column'] == column - 1).any():                                               #
                t[len(t)] = [row, column - 1, self.distances[column, row - 1]]                                          #
            if not ((t['row'] == row) & t['column'] == column + 1).any():                                               #
                t[len(t)] = [row, column + 1, self.distances[column, row - 1]]                                          #
            if not ((t['row'] == row + 1) & t['column'] == column).any():                                               #
                t[len(t)] = [row + 1, column, self.distances[column, row - 1]]                                          #
            t.sort_values('distance')                                                                                   #


    def regionMerging(self) -> None:
        pass

    def main(self) -> None:
        self.imageConvert()     # Step 1: Complete
        self.seedSelection()    # Step 2: Complete (needs debugging)
        self.regionGrowing()    # Step 3: Complete (needs debugging)
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
