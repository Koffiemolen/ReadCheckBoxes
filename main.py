# pip install fitz
# pip install PyMuPDF
# pip install opencv-python numpy Pillow

import fitz
import glob
import cv2
import numpy as np
from PIL import Image
from numpy import *

# Example code:
# https://towardsdatascience.com/convert-pdf-to-image-in-python-using-pymupdf-9cc8f602525b

# You should run pip install fitz followed by pip install PyMuPDF. If you have install PyMuPDF, uninstall it and
# install again.
# https://stackoverflow.com/questions/56467667/how-do-i-resolve-no-module-named-frontend-error-message

# sourceDir: C:/Temp/checkbox/Src
# destDir: C:/Temp/checkbox/Output

# Local folders
sourceDir = 'C:/Temp/checkbox/Src/'
dirPNG = 'C:/Temp/checkbox/PNG/'
destDir = 'C:/Temp/checkbox/BoxHiglight/'

# To get better resolution
zoom_x = 2.0  # horizontal zoom
zoom_y = 2.0  # vertical zoom
mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

# Iterate over files
pdfFiles = glob.glob(sourceDir + "*.pdf")

# Iterating PDF files and export to PNG
for filename in pdfFiles:
    print('Reading file: ', filename)
    doc = fitz.open(filename)  # Open document

    # Getting filename without path and extension
    name = filename.split('.')
    filename = name[0].split('/')
    filename = filename[-1].split('\\')

    for page in doc:  # Iterate through the pages
        print('Processing page: ', page.number)
        pix = page.get_pixmap(matrix=mat)  # Render page to an image

        # Saving PNG
        print('File saved: ', dirPNG + filename[1] + "page-%i.png" % page.number)
        pix.save(dirPNG + filename[1] + "page-%i.png" % page.number)  # Store image as a PNG


# Checking checkboxes

# Variables
pngFiles = glob.glob(dirPNG + "*.png")  # Only select png files
i = 0  # Counter
showImages = False  # Set to true to not save files, instead show files

print(pngFiles)  # Print files that are being processed

# Iterate over files
for filename in pngFiles:
    print('Reading file: ', filename)

    # Read image into array
    inputImage = cv2.imread(filename)

    # Check array type
    type(inputImage)

    # output: numpy.ndarray

    # Prepare a deep copy for results:
    inputImageCopy = inputImage.copy()

    # Converting image to gray scale
    grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Image thresholding
    _, binaryImage = cv2.threshold(grayImage, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binaryImage = 255 - binaryImage

    # Image thresholding v2
    # _, binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Show image
    if showImages:
        Image.fromarray(binaryImage).show()

    # Using morphological operations to identify edges
    # Set min width to detect horizontal lines
    line_min_width = 7

    # Kernel to detect horizontal lines
    kernel_h = np.ones((1, line_min_width), np.uint8)

    # Kernel to detect vertical lines
    kernel_v = np.ones((line_min_width, 1), np.uint8)

    # Horizontal kernel on image
    img_bin_h = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel_h)

    # Vertical kernel on the image
    img_bin_v = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel_v)

    # Show image with horizontal lines only and vertical lines only
    if showImages:
        Image.fromarray(img_bin_h).show()
        Image.fromarray(img_bin_v).show()

    # Combining the image, horizontal + vertical
    img_bin_final = img_bin_h | img_bin_v

    # Show combined image
    if showImages:
        Image.fromarray(img_bin_final).show()

    # Contours Filtering
    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    squares = 0 # Counter for numbering potential checkboxes
    for x, y, w, h, area in stats[2:]:
        print('rectangle : (x , y) = ', x, y)
        print('size: w, h', w, h)
        # Only draw an rectangle on image within a certain size range, width and height
        if (w < 20) and (w > 5) and (h < 20) and (h > 4):
            print('rectangle ', squares, ': (x , y) = ', x, y)
            print('size: w, h', w, h)
            cv2.rectangle(inputImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(inputImage, str(squares), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            squares += 1

    # Show image with found checkboxes highlighted in red
    if showImages:
        Image.fromarray(inputImage).show()

    # saving file
    if not showImages:
        print('File saved: ', destDir + 'contouredImage-%i.png' % i)
        destFileName = destDir + 'contouredImage-%i.png' % i
        Image.fromarray(inputImage).save(destFileName)

    # Increase counter
    i += 1
