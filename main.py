"""
useful links and sources:

    https://java2blog.com/cv-imread-python/#cvimread_Method_example # good source to look up basic cv functionalities



"""
from tkinter import ttk, messagebox

import cv2 as cv
import matplotlib
import numpy as np
import glob
import pathlib
import tkinter as tk
from PIL import Image, ImageTk
import os

# printing bigger np matrix
# # a = np.arange(127 * 127).reshape(127, 127)
np.set_printoptions(edgeitems=127)  # this line sets the amount you want to print


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
        self.originalImage = ImageTk.PhotoImage(image=Image.fromarray(imageArray))
        self.originalImageArray = imageArray

        self.imageArray = imageArray
        self.colorType = colorType

        self.newSize = (800, 600)  # default size for all images displayed in the program
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
        self.originalImage = ImageTk.PhotoImage(image=Image.fromarray(convertedImageArray))
        self.originalImageArray = convertedImageArray
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
        self.imageArray = self.originalImageArray
        self.image = self.originalImage

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

    def findLongEdge(self, lowThreshold=50, houghLineThresh = 50, minLineLen = 50, maxLineGap = 10):
        """
        Applies multiple operations and returns/shows longest edge.

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
        edges = cv.Canny(gray, lowThreshold, highThresold, kernel_size)
        lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=houghLineThresh, minLineLength=minLineLen, maxLineGap=maxLineGap)

        longestLineDist = 0.0
        longestLine = lines[0][0]

        edges = cv.cvtColor(edges, cv.COLOR_GRAY2RGB) # changing back to RGB, displaying line colours

        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv.line(edges, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2) # drawing lines on image

            dist = np.math.hypot(x2 - x1, y2 - y1)  # calculating distance of each line

            if dist > longestLineDist:  # getting the longest line
                longestLineDist = dist
                longestLine = line[0]

        print(longestLineDist)
        print(longestLine)

        cv.line(edges, (longestLine[0], longestLine[1]), (longestLine[2], longestLine[3]), color=(0, 255, 0), thickness= 3)

        x1, y1, x2, y2 = longestLine

        # calculate angle in radian, if you need it in degrees just do angle * 180 / PI
        angle = np.arctan2([y2, y1], [x2, x1]) * 180 / np.pi

        M = cv.getRotationMatrix2D((400, 300), 90-angle[1], 1)
        rotated = cv.warpAffine(edges, M, (edges.shape[1], edges.shape[0]))

        print(angle)

        self.imageArray = rotated
        self.updateImage()


def updateParameter(event):
    """
    Applies the value from the threshold slider on following image objects.
    """
    originalImage.reset()
    originalImage.findLongEdge(int(lowThreshCannySlider.get()),
                               int(houghLineThreshSlider.get()),
                               int(minLineLenSlider.get()),
                               int(maxLineGapSlider.get()))


def callbackFileSelection(event):
    """
    Gets called everytime an image is selected from the listbox
    Changes images according to selection.
    Applies additional functions depending on what kind of operation you want to show.

    Note: inefficiency, creating new objects for each image and reapplying operations instead of inheritance ( from previous images/steps).
    """
    selection = event.widget.curselection()
    selectedImagePath = lbImagePaths.getPath(selection[0])

    # updating originalImage
    originalImage.setImage(selectedImagePath)


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

    # img = cv.imread(cv.samples.findFile('.\\images\\brick.jpg'))
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray, 190, 255, apertureSize=3)
    # cv.imshow("gray", edges)
    # lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv.imshow('houghlines5.jpg', img)

    # initial image
    initImagePath = '.\\images\\brick.jpg'  # imagepath for the initial image ... when program is started
    initImage = cv.imread(initImagePath, cv.IMREAD_COLOR)

    # initializing the image objects/ different views, used in this program
    originalImage = ImageClass(rightTopFrame, initImage, "rgb", "image one")  # creating image object in rgb(default)

    # initialization of all images, copied from callbackFileSelection function

    # updating originalImage
    originalImage.setImage(initImagePath)
    originalImage.findLongEdge()

    # initialization of GUI objects

    lbFileSelection = tk.Listbox(master, width=30)  # creating a listbox
    lbFileSelection.bind("<<ListboxSelect>>",
                         callbackFileSelection)  # callback function for listbox ... executes when you select an entry
    lbFileSelection.pack(side="top", fill=tk.BOTH, expand=True, padx=10, pady=10,
                         ipady=6)  # outer padding for the listbox/listview

    lbImagePaths = ImagePaths()
    lbImagePaths.fillListBox(lbFileSelection)

    # Low Threshold Canny
    lowThreshCannySlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                                    label="Low Threshold Canny:", command=updateParameter)
    lowThreshCannySlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    lowThreshCannySlider.set(127)  # setting to 127, 127 = start/default value for image objects threshold

    # Hough Lines Threshold
    houghLineThreshSlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                                     label="Hough Lines Threshold:", command=updateParameter)
    houghLineThreshSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    houghLineThreshSlider.set(50)

    # Min Line Length
    minLineLenSlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                                     label="Min Line Length:", command=updateParameter)
    minLineLenSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    minLineLenSlider.set(50)

    # Max Line Gap
    # needs lower maxValue
    maxLineGapSlider = tk.Scale(master, from_=0, to=30, orient=tk.HORIZONTAL,
                                     label="Max Line Gap:", command=updateParameter)
    maxLineGapSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    maxLineGapSlider.set(10)

    master.mainloop()  # window mainloop

    cv.destroyAllWindows()
