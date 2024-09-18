#####################################################################################
#
# MRGCV Unizar - Computer vision - Final Course Project
#
# Title: Compute Camera Calibration and undistort the selected images
#
# Date: 12 January 2024
#
#####################################################################################
#
# Authors: Pedro José Pérez García

# Version: 1.0
#
#####################################################################################

import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

###########################################################################################
# IMPORTANT VARIABLES AND PARAMETERS FOR THIS SCRIPT, LOADED FROM A FILE LATER ON!!!
image_load_folder = '../new_pictures/'
image_load_names = 'torre_belem_*.jpg'
image_save_folder = '../undistorted_pictures/'
image_save_names = 'torre_belem_undistorted_'
checkered_images_load_folder = '../checkerboard_photos_for_calibration/'
checkered_images_load_names = 'calib_*.jpg'
calibration_matrix_save_folder = '../camera_data'
calibration_matrix_save_name = 'calibration_matrix.txt'
rad_dist_save_folder = '../camera_data'
rad_dist_save_name = 'radial_distortion.txt'

# Bool values
do_imshow = False

# Calibration pattern size config
pattern_num_rows = 9
pattern_num_cols = 6
pattern_size= (pattern_num_rows, pattern_num_cols)
############################################################################################
    


def main_camera_calibration():
    #mobile phone cameras can have a very high resolution.
    # It can be reduced to reduce the computing overhead
    image_downsize_factor = 1               # WE DON'T DOWNSIZE (soy masoquista)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_num_rows*pattern_num_cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_num_rows,0:pattern_num_cols].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(os.path.join(checkered_images_load_folder, checkered_images_load_names))
    #cv.namedWindow('img', cv.WINDOW_NORMAL)  # Create window with freedom of dimensions
    #cv.resizeWindow('img', 800, 600)
    for fname in images:
        img = cv.imread(fname)
        img_rows = img.shape[1]
        img_cols = img.shape[0]
        new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
        img = cv.resize(img, new_img_size, interpolation = cv.INTER_CUBIC)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        
        print('Processing caliration image:', fname)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, pattern_size, corners2, ret)
            if do_imshow:
                cv.imshow('img', img)
                cv.waitKey(500)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    #initial_distortion = np.zeros((1, 5))
    #initial_K = np.eye(3)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=(cv.CALIB_ZERO_TANGENT_DIST))

    # reprojection error for the calibration images
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    print('The calibartion matrix is')
    print(mtx)
    print('The radial distortion parameters are')
    print(dist)

    # undistorting the images
    print('Undistoring the images')
    images = sorted(glob.glob(os.path.join(image_load_folder, image_load_names)))
    image_count = 1
    for fname in images:
        img = cv.imread(fname)
        img_rows = img.shape[1]
        img_cols = img.shape[0]
        new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
        img = cv.resize(img,new_img_size, interpolation = cv.INTER_CUBIC)
        undist_image = cv.undistort(img, mtx, dist)
        cv.imwrite(os.path.join(image_save_folder, (image_save_names + str(image_count) + '.jpg')), undist_image)
        print("Saving image ", os.path.join(image_save_folder, (image_save_names + str(image_count) + '.jpg')))
        if do_imshow:
            cv.imshow(os.path.join(image_save_folder, (image_save_names + str(image_count) + '.jpg')), undist_image)
            cv.waitKey(500)
        image_count = image_count + 1
    print('\nDone!')
    print('Writing camera calibration and radial distortion data into txt files...')
    np.savetxt(os.path.join(calibration_matrix_save_folder, calibration_matrix_save_name), np.array(mtx))
    np.savetxt(os.path.join(rad_dist_save_folder, rad_dist_save_name), np.array(dist))
    print('\nDone!')
    return



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_camera_calibration()
    


