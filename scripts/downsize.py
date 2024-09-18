import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

image_downsize_factor = 2
folder_to_downsize_images_from = '../new_pictures_ORIGINAL_DONT_TOUCH/'
folder_to_downsize_images_to = '../new_pictures/'

for file in sorted(os.listdir(folder_to_downsize_images_from)):
        if not os.path.isdir(file) and file[-4:] == '.jpg':
            img = cv.imread(os.path.join(folder_to_downsize_images_from, file))
            img_rows = img.shape[1]
            img_cols = img.shape[0]
            new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
            img2 = cv.resize(img, new_img_size, interpolation = cv.INTER_CUBIC)
            cv.imwrite(os.path.join(folder_to_downsize_images_to, file), img2)