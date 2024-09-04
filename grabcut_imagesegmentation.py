import cv2
import numpy as np
from matplotlib import pyplot as plt

#information (and most code, maybe plagiat xd) from https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
#todo: takes awfully long -> now used paint magic function which does the same.

# Load the reference image (glass image)
template = cv2.imread('glass.jpg')
# Ensure the image is in BGR format
if template.shape[2] == 4:  # Image with alpha channel (4 channels)
    template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

if template is None:
    print("Error: Could not load template image.")
    exit()

#TODO: let them draw rectangle around glass for easier (see extra file)

height, width = template.shape[:2]
rect = (int(height/10), int(width/4), int(8*height/10), int(5*width/8))

# Initialize the mask
mask = np.zeros((height, width), np.uint8)

# Temporary arrays used by the GrabCut algorithm internally (no idea how)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Apply GrabCut algorithm (5 iterations, rectangular mode)
cv2.grabCut(template, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask so that 0 (definitely background) and 2 (probable background) pixels are converted to background (0),
# and 1 and 3 pixels are converted to foreground (1)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Segment the image using the mask
segmented_image = template * mask2[:, :, np.newaxis]

# Show the segmented image
cv2.imshow('Segmented Glass', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
