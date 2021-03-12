"""
useful links and sources:

    https://java2blog.com/cv2-imread-python/#cv2imread_Method_example # good source to look up basic cv2 functionalities

    https://docs.opencv.org/4.5.1/db/d8e/tutorial_threshold.html # documentation and examples for thresholding operations
    https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

    https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html # opencv erosion and dilatation
    https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb # functions



"""
from tkinter import ttk, messagebox

import cv2
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

        self.newSize = (400, 300)  # default size for all images displayed in the program
        resized = cv2.resize(self.imageArray, self.newSize)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.imageLabel = tk.Label(frame, image=self.image, compound="top", text=title)
        self.imageLabel.pack(side="left", padx=10, pady=10)

    def setImage(self, imagePath):
        """
        Changes the current image and updates with updateImage().
        """
        global convertedImageArray
        imageArray = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        if self.colorType == "rgb":
            convertedImageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB)

        if self.colorType == "gray":
            convertedImageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)

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

        resized = cv2.resize(self.imageArray, self.newSize)  # takes image array and resizes it, returns new image array
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
    master.title('countingCoins')  # title of the program window
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
    initImage = cv2.imread(initImagePath, cv2.IMREAD_COLOR)

    # initializing the image objects/ different views, used in this program
    originalImage = ImageClass(rightTopFrame, initImage, "rgb", "ORIGINAL")  # creating image object in rgb(default)

    # initialization of all images, copied from callbackFileSelection function

    # updating originalImage
    originalImage.setImage(initImagePath)

    # initialization of GUI objects

    lbFileSelection = tk.Listbox(master, width=30)  # creating a listbox
    lbFileSelection.bind("<<ListboxSelect>>",
                         callbackFileSelection)  # callback function for listbox ... executes when you select an entry
    lbFileSelection.pack(side="top", fill=tk.BOTH, expand=True, padx=10, pady=10,
                         ipady=6)  # outer padding for the listbox/listview

    lbImagePaths = ImagePaths()
    lbImagePaths.fillListBox(lbFileSelection)

    master.mainloop()  # window mainloop

    cv2.destroyAllWindows()
