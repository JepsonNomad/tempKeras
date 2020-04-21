# Crop and resize imagery en masse

# import numpy as np
import cv2 as cv
import glob
import os
import sys

#### Define parameters ----
# Identify folder of interest
position = sys.argv[1]

# Bounding box of temperature stamp
xmin = int(sys.argv[2])
xmax = int(sys.argv[3])
ymin = int(sys.argv[4])
ymax = int(sys.argv[5])
# xmin = 1090
# xmax = 1400
# ymin = 2250
# ymax = 2340

# Desired output dimensions
width = int(sys.argv[6])
height = int(sys.argv[7])
# width = 120
# height = 30


#### Crop unuseful pixels ----

# Change directory to position folder
os.chdir(position)
# Create an output folder if it doesn't already exist:
if not os.path.exists("tempImg"):
	os.mkdir("tempImg")

# Get images
imgs = glob.glob('*.JPG')
imgs = sorted(imgs)

# For each image crop and save:
for q in range(len(imgs)):
    img = imgs[q]
    outimg = img
    print("*** image " + str(q+1) + " of " + str(len(imgs)) + " ***")
    # Open image
    image = cv.imread(img)
    # Crop using extent parameters from above
    imNewCrop = image[ymin:ymax,xmin:xmax, :]
    # Resize using dimension parameters from above
    imNewRes = cv.resize(imNewCrop, (width, height))
    # Save image
    cv.imwrite("tempImg/"+outimg, imNewRes)
    # Close existing connections
    image = None
    imNewCrop = None
    imNewRes = None
