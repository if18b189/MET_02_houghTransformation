"""
useful links and sources:

    https://java2blog.com/cv-imread-python/#cvimread_Method_example # good source to look up basic cv functionalities



"""

import cv2 as cv
import numpy as np
import glob
import tkinter as tk
from PIL import Image, ImageTk
import os
import math

# useful for debugging purposes, making the entire numpy array visible when printed
# # a = np.arange(127 * 127).reshape(127, 127)
np.set_printoptions(edgeitems=127)  # this line sets the amount you want to print

# the resolution in which the images will be displayed
resolutionX = 800
resolutionY = 600


class ImagePaths:
    """
    Finds all images and summarizes their paths.
    """

    def __init__(self, path=os.getcwd() + "\\images", imageType="jpg"):
        """
        Constructor
        """
        self.imagePaths = glob.glob(os.path.join(path, '*.' + imageType))  # searching for all .jpg files

        # print(self.imagePaths)
        # print(os.getcwd())

    def fillListBox(self, listBoxObject):
        """
        Fills the listbox(GUI) with image names.
        """
        for imagePath in self.imagePaths:
            imageName = imagePath.split("\\")[-1]  # splitting all the .pdf up
            listBoxObject.insert('end', imageName)  # inserting each word into tk listbox

    def getPath(self, listBoxIndex):
        """
        Returns image path according to the given index.
        """
        return self.imagePaths[listBoxIndex]


class ImageClass:
    """
    Shows image on a window as a tkinter label. Summarizes operations for images.
    """

    def __init__(self, frame, imageArray, colorType="rgb", title=""):
        """
        Constructor
        """

        self.title = title
        self.imageOriginal = ImageTk.PhotoImage(image=Image.fromarray(imageArray))
        self.imageOriginalArray = imageArray  # remains unchanged

        self.imageArray = imageArray  # stores the results of operations, transformations, ...
        self.colorType = colorType

        self.newSize = (resolutionX, resolutionY)  # default size for all images displayed in the program
        resized = cv.resize(self.imageArray, self.newSize)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.imageLabel = tk.Label(frame, image=self.image, compound="top", text=title)
        self.imageLabel.pack(side="left", padx=10, pady=10)

    def setImage(self, imagePath):
        """
        Changes the current image and updates with updateImage().
        """
        imageArray = cv.imread(imagePath, cv.IMREAD_COLOR)

        convertedImageArray = imageArray  # creating variable convertedImageArray

        if self.colorType == "rgb":
            convertedImageArray = cv.cvtColor(imageArray, cv.COLOR_BGR2RGB)

        if self.colorType == "gray":
            convertedImageArray = cv.cvtColor(imageArray, cv.COLOR_BGR2GRAY)

        # add more if statements here for additional color options

        # print(self.title + ": " + imagePath)

        self.image = ImageTk.PhotoImage(image=Image.fromarray(convertedImageArray))
        self.imageArray = convertedImageArray
        self.imageOriginal = ImageTk.PhotoImage(image=Image.fromarray(convertedImageArray))
        self.imageOriginalArray = convertedImageArray
        self.updateImage()

    def updateImage(self):
        """
        Resizes and updates the currently displayed image with the given image array.
        """

        resized = cv.resize(self.imageArray, self.newSize)  # takes image array and resizes it, returns new image array
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.image = imgtk

        self.imageLabel['image'] = imgtk  # updating the label to show the new image
        self.imageLabel.photo = imgtk

    def reset(self):
        """
        Returns the image and imageArray to their original values (the values, they were initialized with).
        """
        self.imageArray = self.imageOriginalArray
        self.image = self.imageOriginal

    def getImage(self):
        """
        Returns the image currently set in the class object.
        """
        return self.image

    def getImageArray(self):
        """
        Returns the image array currently set in the class object.
        """
        return self.imageArray

    def findLongEdgeRotate(self, lowThreshold=50, houghLineThresh=50, minLineLen=50, maxLineGap=10, rotationOffset=0):
        """
        Applies multiple operations and marks the longest detected edge in the image in yellow.
        Rotates the entire image according to the angle of the longest detected edge

        Important sources:
            canny:
                https://www.docs.opencv.org/master/da/d22/tutorial_py_canny.html
                https://docs.opencv.org/master/da/d5c/tutorial_canny_detector.html

            houghTransformation:
                https://docs.opencv.org/master/d6/d10/tutorial_py_houghlines.html
                https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb

        """

        kernel_size = 3

        highThresold = lowThreshold * 3  # " Canny recommended a upper:lower ratio between 2:1 and 3:1. "

        gray = cv.cvtColor(self.imageArray, cv.COLOR_RGB2GRAY)  # depending on previous conversions (RGB)
        edges = cv.Canny(gray, lowThreshold, highThresold, kernel_size)  # canny edge detection
        lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=houghLineThresh, minLineLength=minLineLen,
                               maxLineGap=maxLineGap)

        longestLineDist = 0.0  # stores longest line distance
        longestLine = lines[0][0]  # stores coordinates ...

        edges = cv.cvtColor(edges, cv.COLOR_GRAY2RGB)  # converting to RGB (to display different line colors)

        # drawing all the lines from the HoughLinesP operation on an image (could also be the original image)
        for line in lines:
            x1, y1, x2, y2 = line[0]  # getting P1 & P2 coordinates

            cv.line(edges, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)  # drawing all lines in red

            dist = np.math.hypot(x2 - x1, y2 - y1)  # calculating distance of each line

            if dist > longestLineDist:  # getting the longest line
                longestLineDist = dist  # length
                longestLine = line[0]  # coordinates

        # marking the longest line in yellow
        cv.line(edges, (longestLine[0], longestLine[1]), (longestLine[2], longestLine[3]), color=(0, 255, 0),
                thickness=3)

        x1, y1, x2, y2 = longestLine  # setting coordinates for P1(x1,y1)/P2(x2,y2) of the longest line for further operations

        # calculating angle of the line in degrees
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # rotating the image according to the calculated angle + rotationOffset from slider
        M = cv.getRotationMatrix2D((resolutionX / 2, resolutionY / 2), rotationOffset + angle, 1)
        rotated = cv.warpAffine(edges, M, (edges.shape[1], edges.shape[0]))

        self.imageArray = rotated
        self.updateImage()


def updateParameter(event):
    """
    Applies the value from the UI(sliders, ...) on following image objects.
    """
    imageObject.reset()
    imageObject.findLongEdgeRotate(int(lowThreshCannySlider.get()),
                                   int(houghLineThreshSlider.get()),
                                   int(minLineLenSlider.get()),
                                   int(maxLineGapSlider.get()),
                                   int(rotationOffsetSlider.get()))


def callbackFileSelection(event):
    """
    Gets called everytime an image is selected from the listbox
    Changes images according to selection.
    Applies additional functions depending on what kind of operation you want to show.

    Note: inefficiency, creating new objects for each image and reapplying operations instead of inheritance ( from previous images/steps).
    """
    selection = event.widget.curselection()
    selectedImagePath = lbImagePaths.getPath(selection[0])

    # updating imageObject
    imageObject.setImage(selectedImagePath)


if __name__ == '__main__':
    # main application window
    master = tk.Tk()  # creating a tk application+
    master.title('houghTransformation')  # title of the program window
    master.geometry("")  # defining the window size, blank means it will "self adjust"

    # subframes, structuring the alignment of GUI objects

    rightFrame = tk.Frame(master)
    rightFrame.pack(side='right', fill=tk.BOTH, expand=True)

    rightTopFrame = tk.Frame(rightFrame)
    rightTopFrame.pack(side='top', fill=tk.BOTH, expand=True)

    rightBottomFrame = tk.Frame(rightFrame)
    rightBottomFrame.pack(side='bottom', fill=tk.BOTH, expand=True)

    # initial image
    initImagePath = '.\\images\\brick.jpg'  # imagepath for the initial image ... when program is started
    initImage = cv.imread(initImagePath, cv.IMREAD_COLOR)

    # initializing the image objects/ different views, used in this program
    imageObject = ImageClass(rightTopFrame, initImage, "rgb", "image one")  # creating image object in rgb(default)

    # initialization of all images, copied from callbackFileSelection function

    # updating imageObject
    imageObject.setImage(initImagePath)
    imageObject.findLongEdgeRotate()

    # initialization of GUI objects

    # Listbox for file selection
    lbFileSelection = tk.Listbox(master, width=30)  # creating a listbox
    lbFileSelection.bind("<<ListboxSelect>>",
                         callbackFileSelection)  # callback function for listbox ... executes when you select an entry
    lbFileSelection.pack(side="top", fill=tk.BOTH, expand=True, padx=10, pady=10,
                         ipady=6)  # outer padding for the listbox/listview

    lbImagePaths = ImagePaths()
    lbImagePaths.fillListBox(lbFileSelection)

    # Low Threshold Canny Slider
    lowThreshCannySlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                                    label="Low Threshold Canny:", command=updateParameter)
    lowThreshCannySlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    lowThreshCannySlider.set(127)  # setting to 127, 127 = start/default value for image objects threshold

    # Hough Lines Threshold Slider
    houghLineThreshSlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                                     label="Hough Lines Threshold:", command=updateParameter)
    houghLineThreshSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    houghLineThreshSlider.set(50)

    # Min Line Length Slider
    minLineLenSlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                                label="Min Line Length:", command=updateParameter)
    minLineLenSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    minLineLenSlider.set(50)

    # Max Line Gap Slider
    # needs lower maxValue
    maxLineGapSlider = tk.Scale(master, from_=0, to=30, orient=tk.HORIZONTAL,
                                label="Max Line Gap:", command=updateParameter)
    maxLineGapSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    maxLineGapSlider.set(10)

    # Manual Rotation Offset Slider
    rotationOffsetSlider = tk.Scale(master, from_=-90, to=90, orient=tk.HORIZONTAL,
                                    label="Rotation Offset:", command=updateParameter)
    rotationOffsetSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    rotationOffsetSlider.set(0)

    master.mainloop()  # window mainloop

    cv.destroyAllWindows()
