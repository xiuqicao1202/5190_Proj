import numpy as np
import pandas as pd
import cv2 as cv
from pathlib import Path
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt


def show_gray(img):
    plt.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


# Local image next to this script (also works when cwd is elsewhere)
image_path = Path(__file__).resolve().parent / "Albert_Einstein_1921_by_F_Schmutzer.jpg"

# Load image and resize it, so that it's easier to see the effects
# of filters/convolutions.
image = io.imread(str(image_path))
gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)


# Convert to grayscale
orig_h = gray_image.shape[0]
orig_w = gray_image.shape[1]
resize_ratio = 0.25 #@param {type:"number"}
new_h = int(orig_h * resize_ratio)
new_w = int(orig_w * resize_ratio)
gray_image = transform.resize(gray_image, (new_h, new_w)) * 256 # Convert it back to the uint8 range of 0-255

# Display it
show_gray(gray_image)








average_filter = np.ones([3, 3]) / 9

warpped_image = cv.filter2D(gray_image, -1, average_filter)
show_gray(warpped_image)
