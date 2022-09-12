from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
from pathlib import Path
from os import listdir
from os import path
import os
import shutil
import functools

path = "C:/Users/Joshua/Desktop/RPA_Test/testfolder/boxhighlight/"
checkedP = "C:/Users/Joshua/Desktop/RPA_Test/testfolder/checked/"
uncheckedP = "C:/Users/Joshua/Desktop/RPA_Test/testfolder/unchecked/"
unconfidentP = "C:/Users/Joshua/Desktop/RPA_Test/testfolder/unconfident/"
included_extensions = ['png','PNG']

THRESHOLD = 117

allFiles = [f for f in listdir(path) if any(f.endswith(ext) for ext in included_extensions)] # Get all files in current directory

length = len(allFiles)

def check_file(filePath):
    if path.exists(filePath):
        numb = 1
        while True:
            newPath = "{0}_{2}{1}".format(*path.splitext(filePath) + (numb,))
            if path.exists(newPath):
                numb += 1
            else:
                return newPath
    return filePath

for i in range(length):
    img = cv2.imread(path+allFiles[i])
    imgCrop = img[74:1509,104:1626]
    gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
    ret,bw = cv2.threshold(gray,220,255,cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP,1)
    cntLen = 1
    ct = 0 #number of contours
    for cnt in contours:
        if len(cnt) > cntLen: #eliminate the noises
            ct += 1
    print('Total contours: ',ct)

    if ct >= 3:
        avarageWhitePixels = []
        print("Checked!")
        shutil.move(path + allFiles[i], checkedP + allFiles[i])
        img = cv2.imread(checkedP+allFiles[i], cv2.IMREAD_GRAYSCALE)
        imgCrop = img[74:1509,104:1626]
        n_white_pix = np.sum(imgCrop == 255)
        if (n_white_pix > 3747+THRESHOLD):
            new_name = 'unconfident'
            shutil.move(checkedP + allFiles[i], unconfidentP+new_name)
            avarageWhitePixels.append(n_white_pix)

    else:
        print("Not so checked")
        shutil.move(path + allFiles[i], uncheckedP + allFiles[i])
        img = cv2.imread(uncheckedP+allFiles[i], cv2.IMREAD_GRAYSCALE)
        imgCrop = img[74:1509,104:1626]
        n_white_pix = np.sum(imgCrop == 255)
        if (n_white_pix < 3747+THRESHOLD):
            shutil.move(uncheckedP + allFiles[i], unconfidentP+new_name)