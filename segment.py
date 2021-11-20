import cv2 as cv
import os, sys
from matplotlib import pyplot
from tools import *

class segment():
    def __init__(self, imagePath) -> None:
        self.image = cv.imread(imagePath)

    def imageConvert(self) -> None:
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2YCR_CB)

    def inverseImageConvert(self) -> None:
        self.image = cv.cvtColor(self.image, cv.COLOR_YCR_CB2BGR)
    
    def seedSelection(self) -> None:
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
