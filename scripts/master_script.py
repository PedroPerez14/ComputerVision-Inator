#####################################################################################
#
# MRGCV Unizar - Computer vision - Final Course Project
#
# Title: Master Script that controls the entire Pipeline
#
# Date: 20 June 2024
#
#####################################################################################
#
# Authors: Pedro José Pérez García
#
# Version: 1.0
#
#####################################################################################

# POR QUE x1 EN VEZ DE x3 PARA DLT?????¿? Y NO HE MENCIONADO EL GAUSSIAN BLUR EN LA MEMORIA PARA IMAGE DIFFERENCE! Y CANNY DEBERÏA CALCLARLO SOBRE LA IMAGEN SIN BLUR!!!!

import argparse
import numpy as np
import cv2 as cv
import glob
import math
import os
import random
import scipy.linalg
import scipy.optimize as scOptim
import scipy.io as sio
import matplotlib.pyplot as plt
from configparser import ConfigParser

####################################################################################################

def get_config(conf_file):
    config = ConfigParser()
    config.read(conf_file)
    return config



def euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance



def plot_common_inliers(image_path, inliers, message, style):
    plt.figure()
    image_pers_1 = cv.imread(image_path)
    img1 = cv.cvtColor(image_pers_1, cv.COLOR_BGR2RGB)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    wasd = np.array(inliers)
    plt.plot(wasd[:,0], wasd[:,1 ], style, markersize=10)
    plt.title(message)
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()



def count_common_inliers(inliers_x_old, inliers_x_y, threshold):
    count = 0
    common_x_old = []
    common_x_y = []
    for inlier_x_old in inliers_x_old:
        for inlier_x_y in inliers_x_y:
            if(euclidean_distance(inlier_x_old[0], inlier_x_old[1], inlier_x_y[0], inlier_x_y[1]) <= threshold):
                count = count + 1
                common_x_old.append(inlier_x_old)
                common_x_y.append(inlier_x_y)
                break

    return count, common_x_old, common_x_y



def swap_inliers(inliers_to_invert):
    for inlier in inliers_to_invert:
        inlier[0], inlier[2] = inlier[2], inlier[0]
        inlier[1], inlier[3] = inlier[3], inlier[1]
    return inliers_to_invert



def transferError(match, F):
    ''' In this case, this will be the distance from a point to its computed epipolar line '''
    match1 = np.array((match.item(0), match.item(1), 1))
    match2 = np.array((match.item(2), match.item(3), 1))
    
    l2 = F @ match1     # Epipolar line for p1 in image2
    l1 = F.T @ match2
    
    # Now, compute the distance from the points to the lines
    d_match1_l1 = (match1.T @ F.T @ match2) / np.sqrt(l1[0] ** 2 + l1[1] ** 2)
    d_match2_l2 = (match2.T @ F @ match1) / np.sqrt(l2[0] ** 2 + l2[1] ** 2)
    # Return, for example, the average between both distances
    return abs(d_match1_l1 + d_match2_l2) / 2.0



def computeFundamental(correspondences):
    ''' Copypaste from lab 2 '''
    A = []
    A = np.append(A, np.array([correspondences[0].item(0) * correspondences[0].item(2), correspondences[0].item(1) * correspondences[0].item(2), correspondences[0].item(2), correspondences[0].item(0) * correspondences[0].item(3),
                               correspondences[0].item(1) * correspondences[0].item(3), correspondences[0].item(3), correspondences[0].item(0), correspondences[0].item(1), 1]))
    A = np.reshape(A, (1, 9))
    '''range(1, points1.shape[1] - 1)'''
    for i in range(1, correspondences.shape[0]):
        array = np.array([correspondences[i].item(0) * correspondences[i].item(2), correspondences[i].item(1) * correspondences[i].item(2), correspondences[i].item(2),
                          correspondences[i].item(0) * correspondences[i].item(3), correspondences[i].item(1) * correspondences[i].item(3), correspondences[i].item(3),
                          correspondences[i].item(0), correspondences[i].item(1), 1])
        A = np.append(A, np.reshape(array, (1, 9)), axis=0)

    u, s, vT = np.linalg.svd(A)
    V = vT.T

    sol = V[:, -1]
    F = np.zeros((3, 3))
    index = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F[i, j] = sol[index]
            index = index + 1
    # Now we have to force the rank of F to be 2
    u_F, s_F, vT_F = np.linalg.svd(F)
    V_F = vT_F.T
    # We need the two biggest singular values to convert F to rank 2
    sigma1 = s_F[0]
    sigma2 = s_F[1]

    F_real = u_F @ np.diag((sigma1, sigma2, 0)) @ V_F.T
    return F_real



def RANSAC(correspondences, th, im1, kp1, im2, kp2, dMatchesList, iters, quiet, nRandomItems=8):
    f = None
    inliers = []
    inlier_ids = []

    for i in range(iters):
        tentative_f = None
        tentative_inliers = []
        tentative_inlier_ids = []

        # Get nRandomItems (8 for fundamental) points at random
        sampled_ids = random.sample(range(correspondences.shape[0]), nRandomItems)
        sampled_ids.sort()
        random_set = correspondences[sampled_ids]
      
        # Compute the Fundamental matrix as we did in the last lab session
        tentative_f = computeFundamental(random_set)

        # Now, for the rest of matches not in random-set, compute the transfer error
        # and if t_err < th --> tentative_inliers += match
        # Then, if tentative_inliers.count() > inliers.count():
        #       inliers = tentative_inliers, f = tentative_f and also inlier_ids = tentative_inlier_ids
        inlier_id = 0
        for corr in correspondences:
            if corr not in random_set:
                err = transferError(corr, tentative_f)
                if err < th:
                    tentative_inlier_ids.append(inlier_id)
                    tentative_inliers.append(corr)
            inlier_id = inlier_id + 1

        # Introduce a small probability to plot them
        # Just for visualization
        if not quiet and random.randint(1,iters) <= iters/100:
            #print("Matches: ", dMatchesList)
            mylist = [dMatchesList[id] for id in sampled_ids]
            #mylist = [dMatchesList[sampled_ids[0]], dMatchesList[sampled_ids[1]], dMatchesList[sampled_ids[2]], dMatchesList[sampled_ids[3]]]
            #print("WAAAAAH: ", mylist)
            #plotStuff(im1, im2, kp1, kp2, mylist, 'Matches for creating hypothesis F in iteration '+str(i))
            print('This iteration\'s hypothesis is: \n', tentative_f)
            mylist = [dMatchesList[id] for id in tentative_inlier_ids]
            #plotStuff(im1, im2, kp1, kp2, mylist, 'Inliers obtained with hypothesis F in iteration '+str(i))


        if len(tentative_inliers) > len(inliers):
            inliers = tentative_inliers
            inlier_ids = tentative_inlier_ids
            f = tentative_f

    # for plotting reasons
    inlier_matches_list = [dMatchesList[id] for id in inlier_ids]
    return (f / f[-1, -1]), inliers, inlier_matches_list



def mouse_callback(event, x, y, flags, params):
    """This function will be called whenever the mouse is right-clicked"""
    # right-click event value is 2
    if event == 2:
        global right_clicks
        # store the coordinates of the right-click event
        right_clicks = np.array([x, y])
        right_clicks = np.append(right_clicks, [1])



def plotStuff(image_pers_1, image_pers_2, keypoints_sift_1, keypoints_sift_2, dMatchesList, title):
    img_Matched = cv.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2,
                                         dMatchesList,
                                         None,
                                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.title(title)
    plt.imshow(img_Matched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()



def plot_epipolar_lines(F, img1, img2):
    ''' Plots 2 epipolar lines, estimated epipole and ground truth epipole '''
    scale_width = 640 / img1.shape[1]
    scale_height = 480 / img1.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img1.shape[1] * scale)
    window_height = int(img1.shape[0] * scale)
    cv.namedWindow('image 1 - Click a point', cv.WINDOW_NORMAL)  
    cv.resizeWindow('image 1 - Click a point', window_width, window_height)
    # set mouse callback function for window
    cv.setMouseCallback('image 1 - Click a point', mouse_callback)
    cv.imshow('image 1 - Click a point', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Calcular linea epipolar en image2
    l1 = F @ right_clicks
    #If it was hom. coords. for x1 in c2 cam
    # l0 = F.T @ x1

    # Show epipolar line in image 2
    plt.figure(1)
    plt.imshow(img2)
    plt.title('Image 2 - Epipolar lines')
    plt.plot([0, (-l1[2] / l1[0])], [(-l1[2] / l1[1]), 0],  # en x (0, -c/a) en y (-c/b, 0)
             color='green', linewidth=1)
    
    if(np.linalg.matrix_rank(F.T) == 2):
        u,s,vT = np.linalg.svd(F.T)
        v=vT.T
        e=v[:,-1]
        plt.plot(e[0]/e[2], e[1]/e[2], 'x')
    else:
        print("Can't calculate epipole! F matrix has rank != 2!")

    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    plt.close('Image 2 - Epipolar lines')

right_clicks = np.zeros((2, 1))



def correspondences(points1, points2):
    A = []
    A = np.append(A, np.array([points1[0, 0] * points2[0, 0], points1[1, 0] * points2[0, 0], points2[0, 0], points1[0, 0] * points2[1, 0],
                               points1[1, 0] * points2[1, 0], points2[1, 0], points1[0, 0], points1[1, 0], 1]))
    A = np.reshape(A, (1, 9))
    for i in range(points1.shape[1] - 1):
        array = np.array([points1[0, i + 1] * points2[0, i + 1], points1[1, i + 1] * points2[0, i + 1], points2[0, i + 1],
                          points1[0, i + 1] * points2[1, i + 1], points1[1, i + 1] * points2[1, i + 1], points2[1, i + 1],
                          points1[0, i + 1], points1[1, i + 1], 1])
        A = np.append(A, np.reshape(array, (1, 9)), axis=0)
    return A



def buildCorrespondences(dMatchesList, keypoints_sift_1, keypoints_sift_2):
    correspondenceList = []
    for match in dMatchesList:
        (x1, y1) = keypoints_sift_1[match.queryIdx].pt
        (x2, y2) = keypoints_sift_2[match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])
    corr_s = np.array(correspondenceList)
    return corr_s



def npztomatches(desc1, desc2, keypoints0, keypoints1, match):
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        if match[kDesc1] >= 0:
            dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
            indexSort = np.argsort(dist)
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches



def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList



def loadSuperglueData(path):
    # First, open and load file
    npz = np.load(path)
    _matches = npz['matches']
    key1 = npz['keypoints0']
    key2 = npz['keypoints1']
    desc1 = npz['descriptors0'].T
    desc2 = npz['descriptors1'].T

    # Second, transform the data into an appropiate format
    key_points1 = cv.KeyPoint_convert(key1)
    key_points2 = cv.KeyPoint_convert(key2)

    matches = npztomatches(desc1, desc2, key1, key2, _matches)
    dMatchesList = indexMatrixToMatchesList(matches)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    return dMatchesList, key_points1, key_points2, desc1, desc2



def superglue_ransac(path_im1, path_im2, pathMatches, ransacThreshold, iters, do_interactive):
    #Load the data and the images
    dMatchesList, keypoints_sg_1, keypoints_sg_2, descriptors_1, descriptors_2 = loadSuperglueData(pathMatches)
    image_pers_1 = cv.imread(path_im1)
    image_pers_2 = cv.imread(path_im2)

    # Plot the obtained matches
    if do_interactive:
        imgMatched = cv.drawMatches(image_pers_1, keypoints_sg_1, image_pers_2, keypoints_sg_2, dMatchesList[:100],
                                    None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
        plt.title('First 100 matches with Superglue')
        plt.draw()
        plt.waitforbuttonpress()

    # Now, run the RANSAC
    correspondences = buildCorrespondences(dMatchesList, keypoints_sg_1, keypoints_sg_2)
    fundamental, inliers, id_list_for_plotting = RANSAC(correspondences, ransacThreshold, image_pers_1, keypoints_sg_1, image_pers_2, keypoints_sg_2, dMatchesList, iters, not do_interactive)

    # Plot again the inliers
    if do_interactive:
        plotStuff(image_pers_1, image_pers_2, keypoints_sg_1, keypoints_sg_2, id_list_for_plotting, 'Final '+ str(len(inliers)) + ' inliers obtained after '+str(iters)+' iterations')


    if do_interactive:
        img1 = cv.cvtColor(cv.imread(path_im1), cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(cv.imread(path_im2), cv.COLOR_BGR2RGB)

        # {match1, match2} is an inlier match, and {match3, match4} is another one
        # (we need 2 matched points to plot a line)
        match1 = np.array((inliers[0].item(0), inliers[0].item(1), 1))
        match2 = np.array((inliers[0].item(2), inliers[0].item(3), 1))
        match3 = np.array((inliers[1].item(0), inliers[1].item(1), 1))
        match4 = np.array((inliers[1].item(2), inliers[1].item(3), 1))
        l21 = fundamental.T @ match1     # Epipolar line for match1 in image2
        l11 = fundamental @ match2   # Epipolar line for match2 in image1
        l22 = fundamental.T @ match3     # Epipolar line for match1 in image2
        l12 = fundamental @ match4   # Epipolar line for match2 in image1
        
        plt.close()
        plot_epipolar_lines(fundamental, img1, img2)
        plt.close()
        cv.destroyAllWindows()
    return fundamental, inliers, id_list_for_plotting



def pose_estimation_stage1_calibration(config):
    print("--------------------------------------------------------------------------------------------------------------------------------")
    print("\nStage 1: Camera calibration. K_c and radial distortion will be estimated. Also, undistorted camera pictures will be generated.\n")
    print("--------------------------------------------------------------------------------------------------------------------------------")
    ########################## GET PARAMETERS FROM CONFIG FILE ##########################
    image_load_folder = config.get('stage1', 'image_load_folder')
    image_load_names = config.get('stage1', 'image_load_names')
    image_save_folder = config.get('stage1', 'image_save_folder')
    image_save_names = config.get('stage1', 'image_save_names')
    checkered_images_load_folder = config.get('stage1', 'checkered_images_load_folder')
    checkered_images_load_names = config.get('stage1', 'checkered_images_load_names')
    calibration_matrix_save_folder = config.get('stage1', 'calibration_matrix_save_folder')
    calibration_matrix_save_name = config.get('stage1', 'calibration_matrix_save_name')
    rad_dist_save_folder = config.get('stage1', 'rad_dist_save_folder')
    rad_dist_save_name = config.get('stage1', 'rad_dist_save_name')

    # Bool values
    do_imshow = config.getboolean('stage1', 'do_imshow')

    # Calibration pattern size config
    pattern_num_rows = config.getint('stage1', 'pattern_num_rows')
    pattern_num_cols = config.getint('stage1', 'pattern_num_cols')
    pattern_size = (pattern_num_rows, pattern_num_cols)
    #####################################################################################

    #mobile phone cameras can have a very high resolution.
    # It can be reduced to reduce the computing overhead
    image_downsize_factor = 2           

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

        
        print('Processing calibration image:', fname)
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
        undist_image = cv.undistort(img, mtx, dist)
        cv.imwrite(os.path.join(image_save_folder, (image_save_names + str(image_count) + '.jpg')), undist_image)
        print("Saving image ", os.path.join(image_save_folder, (image_save_names + str(image_count) + '.jpg')))
        if do_imshow:
            cv.imshow(os.path.join(image_save_folder, (image_save_names + str(image_count) + '.jpg')), undist_image)
            cv.waitKey(500)
        image_count = image_count + 1
    print('\nDone!')
    print('Writing camera calibration and radial distortion data into .txt files...')
    np.savetxt(os.path.join(calibration_matrix_save_folder, calibration_matrix_save_name), np.array(mtx))
    np.savetxt(os.path.join(rad_dist_save_folder, rad_dist_save_name), np.array(dist))
    print('\nDone!\n')
    return


def pose_estimation_stage2_matching(config):
    ########################## GET PARAMETERS FROM CONFIG FILE ##########################
    matches_file_folder = config.get('stage2', 'matches_file_folder')
    matches_file_name = config.get('stage2', 'matches_file_name')
    new_images_folder = config.get('stage2', 'new_images_folder')
    old_images_folder = config.get('stage2', 'old_images_folder')
    old_image_name = config.get('stage2', 'old_image_name')
    superglue_matches_output_folder = config.get('stage2', 'matches_output_folder')
    #####################################################################################
    print("--------------------------------------------------------------------------------------------------------------------------------")
    print("\nStage 2: Image matching. This step relies on the Neural Network SuperGLUE, we first create a list of images to match.")
    print("--------------------------------------------------------------------------------------------------------------------------------")
    # Part 1: Create matches.txt
    files = []
    matches = []

    for file in sorted(os.listdir(new_images_folder)):
        if (not os.path.isdir(file) and (file[-4:] == '.png' or file[-4:] == '.jpg')):
            files.append(file)

    for i in range(len(files)):
        for j in range(len(files)):
            if i != j:
                matches.append(files[i]+" "+files[j]+'\n')
        matches.append(files[i] + " " + os.path.join(old_images_folder, old_image_name)+'\n')
        
    if os.path.exists(os.path.join(matches_file_folder, matches_file_name)):
        os.remove(os.path.join(matches_file_folder, matches_file_name))
       
    f = open(os.path.join(matches_file_folder, matches_file_name), "a")
    for match in matches:
        f.write(match)
    f.close()
    
    # Part 2: Run the matching, in this case, SuperGLUE (should I try LightGLUE?)
    os.system("../SuperGluePretrainedNetwork/match_pairs.py --superglue outdoor --max_keypoints 2048 --resize -1 -1 \
              --input_dir " + new_images_folder + " --input_pairs " + os.path.join(matches_file_folder, matches_file_name) + \
                " --output_dir " + superglue_matches_output_folder + " --viz")
    print("Successfully matched the images!")


def pose_estimation_stage3_RANSAC(config):
    ########################## GET PARAMETERS FROM CONFIG FILE ##########################
    matches_list_file_folder = config.get('stage3', 'matches_list_file_folder')
    matches_list_file_name = config.get('stage3', 'matches_list_file_name')
    superglue_matches_folder = config.get('stage3', 'superglue_matches_folder')
    new_images_folder = config.get('stage3', 'new_images_folder')
    old_images_folder = config.get('stage3', 'old_images_folder')
    old_image_name = config.get('stage3', 'old_image_name')
    do_interactive = config.getboolean('stage3', 'do_interactive')
    ransac_iters = config.getint('stage3', 'ransac_iters')
    ransac_threshold = config.getfloat('stage3', 'ransac_threshold')
    common_inliers_threshold = config.getfloat('stage3', 'common_inliers_threshold')
    inliers_save_folder = config.get('stage3', 'inliers_save_folder')
    inliers_im1_old_save_file = config.get('stage3', 'inliers_im1_old_save_file')
    inliers_im1_im2_save_file = config.get('stage3', 'inliers_im1_im2_save_file')
    chosen_imgs_txt_folder = config.get('stage3', 'chosen_imgs_txt_folder')
    chosen_imgs_txt_file = config.get('stage3', 'chosen_imgs_txt_file')
    #####################################################################################
    print("--------------------------------------------------------------------------------------------------------------------------------")
    print("Stage 3: Estimation of the Fundamental matrix using RANSAC (RANdom SAmple Consensus). The obtained inliers will be used later.")
    print("--------------------------------------------------------------------------------------------------------------------------------")

    image_names = []
    
    # Step 1: Read matches.txt list to recover ordered image names and RANSAC them
    f = open(os.path.join(matches_list_file_folder, matches_list_file_name), "r")
    lines = [line.rstrip() for line in f]
    for line in lines:
        img1_name = line.split(' ')[0]
        image_names.append(img1_name)
    image_names = list(dict.fromkeys(image_names))


    # Important variables (se me va a olvidar cómo usarlas en 0.000001 segundos.)
    fundamental_matrices_X_OLD = ['None'] * len(image_names)
    fundamental_matrices_X_Y = [ ['None'] * len(image_names) for i in range(len(image_names)) ]
    inliers_X_with_old_picture = ['None'] * len(image_names)
    inliers_X_with_Y_picture = [ ['None'] * len(image_names) for i in range(len(image_names)) ]

    # Step 2: Brute force search for the OLD-X-Y image combination with the biggest number of common inliers among the 3 images.
    #       Desirably, at least 1 combination should return >=6 2D points (inliers) which are common in all OLD, X, and Y
    max_common_inliers = -1
    best_common_inliers_x_old = 'None'
    inliers_x_y = 'None'
    best_common_inliers_x_y = 'None'
    f_x_old = 'None'
    f_x_y = 'None'
    best_f_x_old = 'None'
    best_f_x_y = 'None'
    best_i = best_j = -1
    inliers_x_old = 'None'
    for i in range(len(image_names)):
        print("Checking image ", image_names[i], " with image ", old_image_name)
        # Compute inliers between i-th image and the old one
        match_npz_name = image_names[i][:-4] + '_' + old_image_name[:-4] + '_matches.npz'
        path_matches_npz = os.path.join(superglue_matches_folder, match_npz_name)

        path_img1 = os.path.join(new_images_folder, image_names[i])
        path_img2 = os.path.join(old_images_folder, old_image_name)

        f_x_old, inliers_x_old, _ = superglue_ransac(path_img1, path_img2, path_matches_npz, ransac_threshold, ransac_iters, do_interactive)
        fundamental_matrices_X_OLD[i] = f_x_old
        inliers_X_with_old_picture[i] = inliers_x_old

        # If num_inliers of image X with the old image <= max_common_inliers, 
        # don't even bother in computing more inliers for that X image
        if len(inliers_x_old) > max_common_inliers:
            for j in range(len(image_names)):
                if False:#j < i:
                    # In this case, we just take the same number of inliers, swap the 
                    # matched points, and transpose the F matrix (F_21 = F_12.T)
                    if fundamental_matrices_X_Y[j][i] != 'None':
                        inliers_x_y = swap_inliers(inliers_X_with_Y_picture[j][i])
                        inliers_X_with_Y_picture[i][j] = inliers_x_y
                    if fundamental_matrices_X_Y[j][i] != 'None':
                        f_x_y = fundamental_matrices_X_Y[j][i].T
                        fundamental_matrices_X_Y[i][j] = f_x_y
                if i != j:#j > i:
                    print("Checking image ", image_names[i], " with image ", image_names[j])
                    match_npz_name = image_names[i][:-4] + '_' + image_names[j][:-4] + '_matches.npz'
                    path_matches_npz = os.path.join(superglue_matches_folder, match_npz_name)
                    path_img1 = os.path.join(new_images_folder, image_names[i])
                    path_img2 = os.path.join(new_images_folder, image_names[j])
                    f_x_y, inliers_x_y, _ = superglue_ransac(path_img1, path_img2, path_matches_npz, ransac_threshold, ransac_iters, do_interactive)
                    inliers_X_with_Y_picture[i][j] = inliers_x_y
                    fundamental_matrices_X_Y[i][j] = f_x_y
                    # Now, check if we have common inliers
                    common_inliers_count, current_common_inliers_x_old, current_common_inliers_x_y = count_common_inliers(inliers_x_old, inliers_x_y, common_inliers_threshold)
                    if common_inliers_count > max_common_inliers:
                        max_common_inliers = common_inliers_count
                        # WARNING: i, j means that the common inliers are i_OLD and i_j
                        best_common_inliers_x_old = current_common_inliers_x_old
                        best_common_inliers_x_y = current_common_inliers_x_y
                        best_f_x_old = f_x_old
                        best_f_x_y = f_x_y
                        best_i = i
                        best_j = j
                        print("New best combination found!: ", old_image_name, " - ", image_names[i], " - ", image_names[j], " with ", max_common_inliers, " inlier matches among the 3 images!")

    print("Overall best combination: ", old_image_name, " - ", image_names[best_i], " - ", image_names[best_j], ", with ", max_common_inliers, " inliers in common!")
    # Small warning, just in case we get the bad ending
    if max_common_inliers >= 6:
        print("At least 6 inliers were found, success!")
    else:
        print("Not enough inliers were found to perform DLT.\nIf you use this result for Bundle Adjustment in stage 4, the results will probably be shit!")

    print("\nSaving the common inliers...")
    np.savetxt(os.path.join(inliers_save_folder, inliers_im1_old_save_file), best_common_inliers_x_old)
    np.savetxt(os.path.join(inliers_save_folder, inliers_im1_im2_save_file), best_common_inliers_x_y)

    print('Saving the fundamental matrices...')
    np.savetxt('../fundamental_matrices/best_f_x_old.txt', best_f_x_old)
    np.savetxt('../fundamental_matrices/best_f_x_y.txt', best_f_x_y)

    print('Saving the names of the 2 chosen images...')
    if os.path.exists(os.path.join(chosen_imgs_txt_folder, chosen_imgs_txt_file)):
        os.remove(os.path.join(chosen_imgs_txt_folder, chosen_imgs_txt_file))

    f = open(os.path.join(chosen_imgs_txt_folder, chosen_imgs_txt_file), "a")
    f.write(os.path.join(new_images_folder, image_names[best_i]))
    f.write('\n')
    f.write(os.path.join(new_images_folder, image_names[best_j]))
    f.close()

    # if do_interactive is enabled, plot matches in image1-OLD and image1-image2 for comparison
    if True:
        print('Plotting the inliers and the common inliers for visualization...')
        plot_common_inliers(os.path.join(new_images_folder, image_names[best_i]), inliers_X_with_old_picture[best_i], 'Image 1, all the inliers with OLD image', 'rx')
        plot_common_inliers(os.path.join(new_images_folder, image_names[best_i]), best_common_inliers_x_old, 'Image 1, all the inliers with OLD image', 'bx')
        plot_common_inliers(os.path.join(new_images_folder, image_names[best_i]), inliers_X_with_Y_picture[best_i][best_j], 'Image 1, all the inliers with image 2', 'rx')
        plot_common_inliers(os.path.join(new_images_folder, image_names[best_i]), best_common_inliers_x_y, 'Image 1, COMMON inliers with image 2', 'bx')

    print("\nDone!")
    return



def matrixA(m1, m2):
    A = np.zeros((2, 9))  # m1.shape[1]

    row1_m1 = np.array([m1[0], m1[1], 1, 0, 0, 0, -m2[0] * m1[0], -m2[0] * m1[1], -m2[0]])
    row2_m1 = np.array([0, 0, 0, m1[0], m1[1], 1, -m2[1] * m1[0], -m2[1] * m1[1], -m2[1]])

    A[0, :] = row1_m1
    A[1, :] = row2_m1
    return A



def triangulate3D(points1, points2, P1, P2):
    #A = np.zeros((4, 4))
    X_w = np.zeros((3, points1.shape[1]))
    for i in range(points1.shape[1]):
        A = np.zeros((4, 4))
        A[0,:] = np.array([P1[2, 0] * points1[0,i] - P1[0, 0], 
                            P1[2, 1] * points1[0,i] - P1[0, 1], 
                            P1[2, 2] * points1[0,i] - P1[0, 2],
                            P1[2, 3] * points1[0,i] - P1[0, 3]])
        
        A[1,:] = np.array([P1[2, 0] * points1[1,i] - P1[1, 0], 
                            P1[2, 1] * points1[1,i] - P1[1, 1], 
                            P1[2, 2] * points1[1,i] - P1[1, 2],
                            P1[2, 3] * points1[1,i] - P1[1, 3]])
        
        A[2,:] = np.array([P2[2, 0] * points2[0,i] - P2[0, 0], 
                            P2[2, 1] * points2[0,i] - P2[0, 1], 
                            P2[2, 2] * points2[0,i] - P2[0, 2],
                            P2[2, 3] * points2[0,i] - P2[0, 3]])
        
        A[3,:] = np.array([P2[2, 0] * points2[1,i] - P2[1, 0], 
                            P2[2, 1] * points2[1,i] - P2[1, 1], 
                            P2[2, 2] * points2[1,i] - P2[1, 2],
                            P2[2, 3] * points2[1,i] - P2[1, 3]])
        u,s,v = np.linalg.svd(A)
        #print("VSHAPE: ", v.shape)
        point = v.T[:, 3]
        point = point[0:3] / point[3]
        X_w[:, i] = point
    return X_w



def z_positive_v2(R, t, X_w):
    val_2 = R[2,:] @ (X_w[:3] - t)
    fr_2 = np.all([val_2 > 0], axis = 0)

    val_1 = np.array([0,0,1]) @ X_w[:3]
    fr_1 = np.all([val_1 > 0], axis = 0)

    res = np.zeros(fr_1.shape)
    for i in range(X_w.shape[1]):
        if fr_1[i] == True and fr_2[i] == True:
            res[i] = 2 
        
    score = np.sum(res == 2)
    return score



def crossMatrixInv(M):
    x = np.array([M[2,1], M[0,2], M[1,0]])
    return x



def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
                [x[2], 0, -x[0]],
                [-x[1], x[0], 0]], dtype="object")
    return M



def recover_R(theta):
    return scipy.linalg.expm(crossMatrix(theta))



def encode_R(R):
    return crossMatrixInv(scipy.linalg.logm(R))



def sphericalToXYZ(Op):
    th  = Op[0]
    phi = Op[1]
    x   = np.sin(th) * np.cos(phi)
    y   = np.sin(th) * np.sin(phi)
    z   = np.cos(th)
    return np.array([x,y,z])



def XYZtoSpherical(x, y, z):
    xy = x**2 + y**2
    if z == 0:
        theta = np.pi / 2.0
    else:
        theta = np.arctan2(np.sqrt(xy), z)
    if x == 0:
        phi = (np.pi / 2.0) * np.sign(y)
    else:
        phi = np.arctan2(y, x)
    return np.array([theta, phi])



# Returns a nparray containing 2*nCameras*nPoints residuals.
# Now we have to consider the translation among cameras!
# Op[0,1] = {phi,th}C1,   Op[2,3,4] = {Rx,Ry,Rz}C1
# Op[5:8] = {tx,ty,tz}C2, Op[8:11] = {Rx,Ry,Rz}C2, ...C3...C<nCameras>
# Op[(6*(nCameras-1)-1) : ((6*nCameras-1)-1) + nPoints*3] = 3DXx, 3DXy, 3DXz
def resBundleProjection_N(Op, xData, K_c, nCameras, nPoints):
    th_ext1 = K_c[0] @ np.append(np.eye(3), np.zeros((3,1)), axis=1)

    #Now we have nCameras cameras, so we should consider one of them as canonical,
    # and the rest will be given from Op (I guess?)
    R = recover_R(Op[2:5])
    t = sphericalToXYZ(Op[0:2])
    th_ext2 = K_c[1] @ np.hstack((R, np.expand_dims(t, axis=1)))    #First camera

    th_ext = [th_ext1, th_ext2]
    for i in range(nCameras - 2):                                   #Rest of cameras
        t = np.expand_dims(Op[5+6*i:8+6*i], axis=1)
        R = recover_R(Op[8+(6*i):11+(6*i)])
        th_ext_i = K_c[i+2] @ np.hstack((R, t))
        th_ext.append(th_ext_i)
    
    # It is more convenient to reorder the xData points by cameras
    xDataPoints = []
    for i in range(nCameras):
        xDataPoints.append(xData[i*2:2+i*2, :]) #all the 2Dpoints for a given cam are 2 rows(x,y), all cols in xData

    # now, residual calculation: for every camera-point combination
    res = []
    for i in range(nCameras):
        for j in range(nPoints):
            X_3D = np.hstack((Op[(6*(nCameras-1)-1)+3*j : (6*(nCameras-1))+j*3+2], np.array([1.0])))
            proj1 = th_ext[i] @ X_3D
            proj1 = proj1[0:2] / proj1[2]
            _res = xDataPoints[i][:,j] - proj1

            res.append(_res[0])
            res.append(_res[1])
    return np.array(res)



def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)



# Returns a nparray containing 2*2*nPoints residuals.
# First 2*nPoints res1, then 2*nPoints res2
# Each res{i} has 2 values, first X then Y
# For example: [res1_0.x, res1_0.y, res1_1.x, res1_1.y, res2_0.x, res2_0.y, res2_1.x, res2_1.y]
def resBundleProjection(Op, x1Data, x2Data, K_c, npoints):

    T = np.expand_dims(sphericalToXYZ(Op[0:2]), axis=1)           # Expand_dims turns the array into a matrix, in this case 1 column
    R = scipy.linalg.expm(crossMatrix(Op[2:5]))


    # Build the Projection matrices for 2 cameras (for now)
    rt1 = np.append(np.eye(3), np.zeros((3,1)), axis=1)     # Camera 1 is in canonical position, we consider it the origin
    P1 = K_c @ rt1                                          # Projection matrix 1 is K_c @ rt1
    
    rt2 = np.append(R, T, axis=1)
    P2 = K_c @ rt2

    # Now, project the points to cam1 and calculate res1
    # Also with cam2
    res1 = []                                                # res to be a col. array, vstack res.x on top of res.y
    res2 = []
    for i in range(npoints):
        X3D_i = np.hstack((Op[5+i*3 : 5+(i+1)*3], np.array([1.0])))
        # For point 1
        projected1 = P1 @ X3D_i
        projected1 = projected1[0:2] / projected1[2]        # Normalize

        # And now for point 2
        projected2 = P2 @ X3D_i
        projected2 = projected2[0:2] / projected2[2]        # Normalize

        #Calculate residuals
        _res1 = x1Data[:, i] - projected1
        #Same for cam2 and xdata2
        _res2 = x2Data[:, i] - projected2

        res1.append(_res1[0])                                # _res1[i].X
        res1.append(_res1[1])                                # _res1[i].Y

        res2.append(_res2[0])                                # _res2[i].X
        res2.append(_res2[1])                                # _res2[i].Y

    res = res1 + res2                                       # First nPoints res1, then nPoints res2
    return np.array(res)



def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)


def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)



def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)



def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)



def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4), dtype=np.float32)
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c



def decomposeP(P):
    # Reference: https://stackoverflow.com/questions/55814640/decomposeprojectionmatrix-gives-unexpected-result
    M = P[0:3,0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2,2]
    A = np.linalg.inv(K) @ M
    l = (1/np.linalg.det(A)) ** (1/3)
    R = l * A
    t = l * np.linalg.inv(K) @ P[0:3,3]
    t = np.append(t, [1])
    return K, R, np.expand_dims(t, axis=-1)



'''
def dlt_sergio(keypoints2D, keypoints3D):
    nMatches = keypoints2D.shape[1]

    # Compute equations
    eqs = []
    for i in range(nMatches):
        x_2D = keypoints2D[:, i]
        x_3D = keypoints3D[:, i]

        eq1 = np.array([-x_3D[0], -x_3D[1], -x_3D[2], -x_3D[3], 0, 0, 0, 0, x_2D[0]*x_3D[0], x_2D[0]*x_3D[1], x_2D[0]*x_3D[2], x_2D[0]*x_3D[3]])
        eq2 = np.array([0, 0, 0, 0, -x_3D[0], -x_3D[1], -x_3D[2], -x_3D[3], x_2D[1]*x_3D[0], x_2D[1]*x_3D[1], x_2D[1]*x_3D[2], x_2D[1]*x_3D[3]])

        eqs.append(eq1)
        eqs.append(eq2)

    eqs_np = np.array(eqs)

    # Solve equations
    u, s, vh = np.linalg.svd(eqs_np)
    P_line = vh[-1, :]
    P = P_line.reshape(3, 4)

    # Decompose projection matrix
    K1, R1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    t1 = t1 / t1[-1]
    t1[:3, -1] *= -1

    K, R, t = decomposeP(P)
    return P, K, R, t
'''



def DLT(points2d,points3d):
    points2d = np.vstack([points2d, np.ones((1, points2d.shape[1]))])
    #points3d = np.vstack([points3d, np.ones((1, points3d.shape[1]))])
    A = np.zeros((2*points2d.shape[1], 12))
    for i in range(points2d.shape[1]):

        A[2*i,:] = np.array([
            -points3d[0, i], -points3d[1, i], -points3d[2, i], -points3d[3, i],
            0, 0, 0, 0,
            points2d[0, i] * points3d[0, i], points2d[0, i] * points3d[1, i], points2d[0, i] * points3d[2, i], points2d[0, i] * points3d[3, i]
        ])

        A[2*i+1] = np.array([
            0, 0, 0, 0,
            -points3d[0, i], -points3d[1, i], -points3d[2, i], -points3d[3, i],
            points2d[1, i] * points3d[0, i], points2d[1, i] * points3d[1, i], points2d[1, i] * points3d[2, i],
            points2d[1, i] * points3d[3, i]
        ])
    print("DLT, Shape of A: ", A.shape)
    u, s, V = np.linalg.svd(A)
    print("V SHAPE: ", V.shape)
    #V = V.T
    res = V[-1, :]
    P = res.reshape((3, 4))
    return P



def BA(path_image_1, path_image_2, path_image_3, do_interactive, path_inliers_1_old, path_inliers_1_2, path_k_c12, path_rad_dist):
    # --------------------------------------- 2.- Bundle adjustment for 2 views --------------------------------------- #
    #Read the images
    
    image_pers_1 = cv.imread(path_image_1)
    image_pers_2 = cv.imread(path_image_2)
    image_pers_3 = cv.imread(path_image_3)
    '''
    image_pers_1 = cv.imread('../../labSession4/image1.png')
    image_pers_2 = cv.imread('../../labSession4/image2.png')
    image_pers_3 = cv.imread('../../labSession4/image3.png')
    '''
    dist_coeffs_k_1_2 = np.loadtxt(path_rad_dist)
    T_w_c1 = np.eye(4, 4)                       # NOTE: FROM W TO C1
    #T_w_c1 = np.loadtxt('../../labSession4/T_w_c1.txt')

    # Compute the Fundamental Matrix as we did in Lab Session 2


    inliers_x1_xold = np.loadtxt(path_inliers_1_old)
    inliers_x1_x2 = np.loadtxt(path_inliers_1_2)

    # Extract the points from the matches
    
    x1 = (inliers_x1_x2[:, 0:2]).T
    x2 = (inliers_x1_x2[:, 2:4]).T
    x3 = (inliers_x1_xold[:, 2:4]).T
    '''
    x1 = np.loadtxt('../../labSession4/x1Data.txt')
    x2 = np.loadtxt('../../labSession4/x2Data.txt')
    x3 = np.loadtxt('../../labSession4/x3Data.txt')
    '''
    num_points_to_use = x1.shape[1]

    K_c_cams_1_2 = np.loadtxt(path_k_c12)
    #K_c_cams_1_2 = np.loadtxt('../../labSession4/K_c.txt')
    F = computeFundamental(inliers_x1_x2)
    
    # Now that we have F, we have to compute E with K_c.T @ F @ K_c
    E = K_c_cams_1_2.T @ F @ K_c_cams_1_2         # NOTE: This E contains t_c2_c1, cam 2 seen from 1: c1 ------> c2
    
    # With E, we have to find the correct rotation to ensemble P and get an initial guess for our points
    U, S, V = np.linalg.svd(E)  #V is already transposed
    W = np.zeros((3,3))
    W[0, 1] = -1
    W[1, 0] = 1
    W[2, 2] = 1

    #We have 4 possible solutions, be careful!
    Rp90 = U @ W @ V
    if(np.linalg.det(Rp90) < 0):
        Rp90 = -Rp90
    
    Rm90 = U @ W.T @ V
    if(np.linalg.det(Rm90) < 0):
        Rm90 = -Rm90

    t = np.reshape(U[:,2], (3, 1))
    I3_3 = np.diag([1,1,1])
    P1 = K_c_cams_1_2 @ np.append(I3_3, np.zeros((3, 1)), axis=1)

    P2_1 = K_c_cams_1_2 @ np.append(Rp90, t, axis=1)
    P2_2 = K_c_cams_1_2 @ np.append(Rp90, -t, axis=1)
    P2_3 = K_c_cams_1_2 @ np.append(Rm90, t, axis=1)
    P2_4 = K_c_cams_1_2 @ np.append(Rm90, -t, axis=1)

    X_w_1 = triangulate3D(x1, x2, P1, P2_1)
    X_w_2 = triangulate3D(x1, x2, P1, P2_2)
    X_w_3 = triangulate3D(x1, x2, P1, P2_3)
    X_w_4 = triangulate3D(x1, x2, P1, P2_4)

    X_w_1_in_c2 = np.append(Rp90, t, axis=1) @ np.vstack((X_w_1, np.ones((1,X_w_1.shape[1]))))
    X_w_2_in_c2 = np.append(Rp90, -t, axis=1) @ np.vstack((X_w_2, np.ones((1,X_w_2.shape[1]))))
    X_w_3_in_c2 = np.append(Rm90, t, axis=1) @ np.vstack((X_w_3, np.ones((1,X_w_3.shape[1]))))
    X_w_4_in_c2 = np.append(Rm90, -t, axis=1) @ np.vstack((X_w_4, np.ones((1,X_w_4.shape[1]))))

    # We have to test which one is the one, which one has more points in front of camera
    '''for i in range(X_w_1.shape[1]):
        print("WASD: ", X_w_1[2,i], " ", X_w_1_in_c2[2,i])'''

    pos_count_1 = sum([X_w_1[2,i] >= 0 and X_w_1_in_c2[2,i] >= 0 for i in range(X_w_1.shape[1])])
    pos_count_2 = sum([X_w_2[2,i] >= 0 and X_w_2_in_c2[2,i] >= 0 for i in range(X_w_2.shape[1])])
    pos_count_3 = sum([X_w_3[2,i] >= 0 and X_w_3_in_c2[2,i] >= 0 for i in range(X_w_3.shape[1])])
    pos_count_4 = sum([X_w_4[2,i] >= 0 and X_w_4_in_c2[2,i] >= 0 for i in range(X_w_4.shape[1])])

    print("First Solution, points in front of both cameras: ", pos_count_1)
    print("Second Solution, points in front of both cameras: ", pos_count_2)
    print("Third Solution, points in front of both cameras: ", pos_count_3)
    print("Fourth Solution, points in front of both cameras: ", pos_count_4)

    pos_count_array = [pos_count_1, pos_count_2, pos_count_3, pos_count_4]
    best_pos_count = max(pos_count_array)

    # Now, pick a winning combination of t and R, 
    # with its corresponding triangulated 3D points, 
    # and compute its residual with our function
    chosen_t = t                        # Initialize to any value, we don't care
    chosen_R = Rp90                     # Same here hehe
    X_3D = X_w_1
    if best_pos_count == pos_count_1:
        chosen_t = t
        chosen_R = Rp90
        X_3D = X_w_1
        print("Initial Guess for camera 2 is +90, t")
    if best_pos_count == pos_count_2:
        chosen_t = -t
        chosen_R = Rp90
        X_3D = X_w_2
        print("Initial Guess for camera 2 is +90, -t")
    if best_pos_count == pos_count_3:
        chosen_t = t
        chosen_R = Rm90
        X_3D = X_w_3
        print("Initial Guess for camera 2 is -90, t")
    if best_pos_count == pos_count_4:          #if best_pos_count == pos_count_4:
        chosen_t = -t
        chosen_R = Rm90
        X_3D = X_w_4
        print("Initial Guess for camera 2 is -90, -t")
    P2 = K_c_cams_1_2 @ np.append(chosen_R, chosen_t, axis=1)
    
    # So far we have computed F, E from F, and then we have triangulated the 3D
    # position of the points from the matches, using F and K_c. We have also recovered the
    # camera 2 position and rotation. This will be our initial guess for the 
    # Bundle Adjustment. We will now plot the error obtained for 2 cameras after
    # performing the 3D triangulation.  

    th_phi = XYZtoSpherical(chosen_t[0], chosen_t[1], chosen_t[2]).flatten()
    RxRyRz = encode_R(chosen_R)
    X_3D_points = X_3D.T.flatten()

    '''print("th_phi shape: ", th_phi.shape)
    print("RxRyRz shape: ", RxRyRz.shape)
    print("X_3D_points shape: ", X_3D_points.shape)'''

    
    Op = np.hstack(( np.hstack((th_phi, RxRyRz)), X_3D_points[0:num_points_to_use*3] ))

    res_ = resBundleProjection(Op, x1, x2, K_c_cams_1_2, num_points_to_use)
    print("ERR LEN: ", len(res_))
    
    # Optimize
    Op_optimized = scOptim.least_squares(resBundleProjection, Op, args=(x1, x2, K_c_cams_1_2, num_points_to_use,), method='lm')

    # Recover 3D points
    X_3D_optimized = np.concatenate((Op_optimized.x[5:8], np.array([1.0])), axis=0)
    for i in range(num_points_to_use-1):
        X_3D_optimized = np.vstack((X_3D_optimized, np.concatenate((Op_optimized.x[8+3*i: 8+3*(i+1)], np.array([1.0])), axis=0)))

    # Recover rotation and translation
    R_c2_c1_optimized = recover_R(Op_optimized.x[2:5])
    t_c2_c1_optimized = sphericalToXYZ(Op_optimized.x[0:2])
    _aux = np.concatenate((R_c2_c1_optimized, np.expand_dims(t_c2_c1_optimized, axis=1)), axis=1)
    P2_optimized = K_c_cams_1_2 @ _aux
    T_c2_c1_optimized = np.vstack((_aux, np.array([0.0, 0.0, 0.0, 1.0])))   #Podría haber usado mi ensemble_T, la verdad



    # Plotting time! :D 
    if do_interactive:
        # 3D plots
        fig3d = plt.figure(2)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        drawRefSystem(ax, (T_w_c1), '-', 'C1')
        #drawRefSystem(ax, wTc1 @ np.linalg.inv(c2Tc1_Op), '-', 'C2_BA')
        drawRefSystem(ax, np.linalg.inv(T_c2_c1_optimized), '-', 'Camera2 optimized after BA')

        X_3D_plot_optimized = (X_3D_optimized).T
        X_3D_plot_optimized[0:3,:] /= X_3D_plot_optimized[3,:]

        ax.scatter(X_3D_plot_optimized[0, :], X_3D_plot_optimized[1, :], X_3D_plot_optimized[2, :], marker='.')
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)
        #plotNumbered3DPoints(ax, X_3D_plot_optimized, 'b', 0.1)
        plt.title('Bundle Adjustment for 2 cameras')
        plt.show()

        # 2D plots after projection
        x1_projected = K_c_cams_1_2 @ np.concatenate((np.identity(3), np.array([[0.0, 0.0, 0.0]]).T), axis=1) @ X_3D_optimized.T
        x2_projected = P2_optimized @ X_3D_optimized.T
        x1_projected /= x1_projected[2, :]
        x2_projected /= x2_projected[2, :]

        plt.figure(3)
        plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
        plt.title('Residuals for Bundle Adjustment in Image1')
        plotResidual(x1.T, x1_projected, 'k-')
        plt.plot(x1[0, :], x1[1, :], 'bo')
        plt.plot(x1_projected[0, :], x1_projected[1, :], 'rx')
        plotNumberedImagePoints(x1[0:2, :], 'r', 4)
        plt.draw()

        plt.show()

        plt.figure(4)
        plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
        plt.title('Residuals for Bundle Adjustment in Image2')
        plotResidual(x2.T, x2_projected, 'k-')
        plt.plot(x2[0, :], x2[1, :], 'bo')
        plt.plot(x2_projected[0, :], x2_projected[1, :], 'rx')
        plotNumberedImagePoints(x2[0:2, :], 'r', 4)
        plt.draw()

        plt.show()

############## 3.- DLT pose estimation of camera three (the old one) ##############
    P3_estimation = DLT(x1, X_3D_optimized.T)
    #K_c_cam_3, _R, _t, _, _, _, _ = cv.decomposeProjectionMatrix(P3_estimation)
    K_c_cam_3, _R, _t = decomposeP(P3_estimation)
    K_c_cam_3 = (K_c_cam_3 / K_c_cam_3[2, 2])

    # P3_estimation = DLT(x3, X_3D_optimized.T)
    # M = P3_estimation[:,:-1]
    # K_c_cam_3, _R, _t = decomposeP(np.sign(np.linalg.det(M)) * P3_estimation)
    # K_c_cam_3 = (K_c_cam_3 / K_c_cam_3[2,2])
    

    print("K_c EST: \n", K_c_cam_3)

    #K_c_cam_3_real = np.loadtxt('../../labSession4/K_c.txt')
    #print("K_c REAL: \n", K_c_cam_3_real)

    K_c_cam_3_ = K_c_cam_3.astype('float32')

    
    X_3D_op_PnP = (X_3D_optimized[:, 0:3]).astype('float32')
    imagePoints = (np.ascontiguousarray(x3[0:2, 0:num_points_to_use].T).reshape((num_points_to_use, 1, 2))).astype('float32')

    print("SHAPE imagePoints (should be Nx2): ", imagePoints.shape)
    print("SHAPE objectPoints (should be Nx3): ", X_3D_op_PnP.shape)

    
    _, rvec, tvec = cv.solvePnP(objectPoints=X_3D_op_PnP, imagePoints=imagePoints, cameraMatrix=K_c_cam_3_, distCoeffs=np.array([]),flags=cv.SOLVEPNP_EPNP)
    
    R_c3_c1 = recover_R(rvec)
    t_c3_c1 = tvec
    
    '''
    R_c3_c1 = _R
    _t /= _t[3, 0]
    t_c3_c1 = -R_c3_c1 @ _t[:3]
    '''
    T_c3_c1_PnP = ensamble_T(R_c3_c1, t_c3_c1.reshape((3,)))
    
    if do_interactive:
        fig = plt.figure(5)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drawRefSystem(ax, (T_w_c1), '-', 'C1')
        drawRefSystem(ax, np.linalg.inv(T_c3_c1_PnP), '-', 'C3_PnP')
        plt.title('3D camera poses cameras 1 and C3 PnP')
        plt.draw()
        plt.show()

############## 4.- Bundle Adjustment from 3 views ##############

# Final part of the lab session: Generalize BA for n cameras and run it for 3 (2, but 3, ok?)

    #initial_guess_SO3_R_c3_c1 = encode_R(R_c3_c1)
    initial_guess_SO3_R_c3_c1 = rvec

    # TODO: Pasar x,y,z porque ya tenemos la escala
    Op = np.hstack((
        np.hstack((th_phi, RxRyRz)), 
        np.hstack(((t_c3_c1[:,0]).flatten(), (initial_guess_SO3_R_c3_c1[:,0]).flatten())), 
        np.array(X_3D_points[0:num_points_to_use*3]),
    ))

    xData = np.vstack((np.vstack((x1, x2)), x3))
    res = resBundleProjection_N(Op, xData, [K_c_cams_1_2, K_c_cams_1_2, K_c_cam_3], 3, num_points_to_use)
    print("Length res BA 3 cams: ", len(res))

    # Now, perform the optimization 
    # TODO: Optimizar solo la tercera cámara!!
    Op_optimized_N = scOptim.least_squares(resBundleProjection_N, Op, args=(xData, [K_c_cams_1_2, K_c_cams_1_2, K_c_cam_3], 3, num_points_to_use,), method='lm')
    
    X_3D_optimized_N = np.concatenate((Op_optimized_N.x[11:14], np.array([1.0])), axis=0)   # Recover the points
    for i in range(num_points_to_use-1):
        X_3D_optimized_N = np.vstack(
            (X_3D_optimized_N, np.concatenate((Op_optimized_N.x[14 + 3 * i: 14 + 3 * i + 3], np.array([1.0])), axis=0)))

    # Recover camera 2 from camera 1 transform matrices after optimization
    R_c2_c1_optimized_N = recover_R(Op_optimized_N.x[2:5])
    t_c2_c1_optimized_n = sphericalToXYZ(Op_optimized_N.x[0:2])
    P2_optimized_N = K_c_cams_1_2 @ np.concatenate((R_c2_c1_optimized_N, np.expand_dims(t_c2_c1_optimized_n, axis=1)), axis=1)
    T_c2_c1_optimized_N = ensamble_T(R_c2_c1_optimized_N, t_c2_c1_optimized_n)

    # Recover camera 3 from camera 1 transform matrices after optimization
    R_c3_c1_optimized_N = recover_R(Op_optimized_N.x[8:11])
    t_c3_c1_optimized_n = Op_optimized_N.x[5:8]
    P3_optimized_N = K_c_cam_3 @ np.concatenate((R_c3_c1_optimized_N, np.expand_dims(t_c3_c1_optimized_n, axis=1)), axis=1) #revisar el concatenate con t_c3_c1_optimized_n
    T_c3_c1_optimized_N = ensamble_T(R_c3_c1_optimized_N, t_c3_c1_optimized_n)

    # Plotting time (again!) :D
    if do_interactive:
        # 3D Plots
        fig = plt.figure(6)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        drawRefSystem(ax, (T_w_c1), '-', 'C1_GT')
        drawRefSystem(ax, np.linalg.inv(T_c2_c1_optimized_N), '-', 'C2_BA')
        drawRefSystem(ax, np.linalg.inv(T_c3_c1_optimized_N), '-', 'C3_BA')

        _3D_plot_optimized_N = (X_3D_optimized_N).T
        _3D_plot_optimized_N[0:3,:] /= _3D_plot_optimized_N[3,:]

        ax.scatter(_3D_plot_optimized_N[0, :], _3D_plot_optimized_N[1, :], _3D_plot_optimized_N[2, :], marker='.')
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)
        #plotNumbered3DPoints(ax, _3D_plot_optimized_N, 'b', 0.1)

        plt.title('3D points after Bundle Adjustment with 3 cameras')
        plt.show()

        # 2D Plots
        x1_projected_n = P1 @ X_3D_optimized_N.T
        x2_projected_n = P2_optimized_N @ X_3D_optimized_N.T
        x3_projected_n = P3_optimized_N @ X_3D_optimized_N.T
        x1_projected_n /= x1_projected_n[2, :]
        x2_projected_n /= x2_projected_n[2, :]
        x3_projected_n /= x3_projected_n[2, :]


        # residuals image 1
        fig_ = plt.figure(7)
        plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
        plt.title('Residuals after Bundle adjustment for 3 cameras in Image 1')
        plotResidual(x1.T, x1_projected_n, 'k-')
        plt.plot(x1[0, :], x1[1, :], 'bo')
        plt.plot(x1_projected_n[0, :], x1_projected_n[1, :], 'rx')
        plotNumberedImagePoints(x1[0:2, :], 'r', 4)
        plt.draw()

        plt.show()

        # residuals image 2
        fig__ = plt.figure(8)
        plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
        plt.title('Residuals after Bundle adjustment for 3 cameras in Image 2')
        plotResidual(x2.T, x2_projected_n, 'k-')
        plt.plot(x2[0, :], x2[1, :], 'bo')
        plt.plot(x2_projected_n[0, :], x2_projected_n[1, :], 'rx')
        plotNumberedImagePoints(x2[0:2, :], 'r', 4)
        plt.draw()

        plt.show()

        # residuals image 3
        fig___ = plt.figure(9)
        plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
        plt.title('Residuals after Bundle adjustment for 3 cameras in Image 3')
        plotResidual(x3.T, x3_projected_n, 'k-')
        plt.plot(x3[0, :], x3[1, :], 'bo')
        plt.plot(x3_projected_n[0, :], x3_projected_n[1, :], 'rx')
        plotNumberedImagePoints(x3[0:2, :], 'r', 4)
        plt.draw()

        plt.show()



def pose_estimation_stage4_bundle_adjustment(config):

    ########################## GET PARAMETERS FROM CONFIG FILE ##########################
    common_inliers_load_folder = config.get('stage4', 'common_inliers_load_folder')
    common_inliers_im1_old_file = config.get('stage4', 'common_inliers_im1_old_file')
    common_inliers_im1_im2_file = config.get('stage4', 'common_inliers_im1_im2_file')
    new_camera_calibration_folder = config.get('stage4', 'new_camera_calibration_folder')
    new_camera_calibration_file = config.get('stage4', 'new_camera_calibration_file')
    new_camera_rad_dist_file = config.get('stage4', 'new_camera_rad_dist_file')
    do_plotting = config.getboolean('stage4', 'do_plotting')
    chosen_imgs_txt_folder = config.get('stage4', 'chosen_imgs_txt_folder')
    chosen_imgs_txt_file = config.get('stage4', 'chosen_imgs_txt_file')
    old_imgs_folder = config.get('stage4', 'old_imgs_folder')
    old_img_file = config.get('stage4', 'old_imgs_file')
    #####################################################################################

    print("--------------------------------------------------------------------------------------------------------------------------------")
    print("Stage 4: Final Stage. Triangulate 3D position of common inlier points to perform DLT to estimate old camera's parameters.")
    print(" After that, perform Bundle Adjustment to estimate the relative position of all the cameras and 3D points. Plots final results.")
    print("--------------------------------------------------------------------------------------------------------------------------------")


    f = open(os.path.join(chosen_imgs_txt_folder, chosen_imgs_txt_file), "r")
    lines = [line.rstrip() for line in f]
    im1 = lines[0]
    im2 = lines[1]

    # path_image_1, path_image_2, path_image_3, do_interactive, path_inliers_1_old, path_inliers_1_2, path_k_c12, path_rad_dist
    BA(im1,
       im2,
       os.path.join(old_imgs_folder, old_img_file),
       do_plotting,
       os.path.join(common_inliers_load_folder, common_inliers_im1_old_file), 
       os.path.join(common_inliers_load_folder, common_inliers_im1_im2_file),
       os.path.join(new_camera_calibration_folder, new_camera_calibration_file),
       os.path.join(new_camera_calibration_folder, new_camera_rad_dist_file)
       )
    
    print("\nDone!")
    return



def pose_estimation_main(args):

    config = get_config(args.config_file)
    stages = sorted(args.stages)
    
    for stage in stages[:4]:
        if int(stage) == 1:
            pose_estimation_stage1_calibration(config)
        elif int(stage) == 2:
            pose_estimation_stage2_matching(config)
        elif int(stage) == 3:
            pose_estimation_stage3_RANSAC(config)
        elif int(stage) == 4:
            pose_estimation_stage4_bundle_adjustment(config)
        else:
            print("ERROR: The provided pipeline stage is not in range [1-4]!\n")
            print(stage," was provided\n")
            print("Aborting...\n")
            return -1
    return



def downsize_img(img, downsize_factor):
    img_rows = img.shape[1]
    img_cols = img.shape[0]
    new_img_size = (int(img_rows / downsize_factor), int(img_cols / downsize_factor))
    im_ret = cv.resize(img, new_img_size, interpolation = cv.INTER_CUBIC)
    return im_ret


# NOTE: https://github.com/bimalka98/Stitch-images-using-SuperGlue-GNN/blob/master/Panorama%20Stitching.ipynb
def loadNPZ(npz_file):    
    npz = np.load(npz_file)
    point_set1 = npz['keypoints0'][npz['matches']>-1]
    matching_indexes =  npz['matches'][npz['matches']>-1] # -1 if the keypoint is unmatched
    point_set2 = npz['keypoints1'][matching_indexes]
    print("Number of matching points for the findHomography algorithm:")
    print("In left  image:", len(point_set1),"\nIn right image:", len(point_set2))
    return point_set1, point_set2



def image_diff_main(args):
    print("--------------------------------------------------------------------------------------------------------------------------------")
    print("Image difference computation: Visual estimation of the differences between two images of a same scene by homography application.")
    print("--------------------------------------------------------------------------------------------------------------------------------")

    # Load images
    im1 = cv.imread(args.img1)
    im2 = cv.imread(args.img2)
    im1 = downsize_img(im1, 2)
    im2 = downsize_img(im2, 2)
    im1_transp = cv.cvtColor(im1, cv.COLOR_RGB2RGBA)
    im2_transp = cv.cvtColor(im2, cv.COLOR_RGB2RGBA)


    # Initiate SIFT detector
    MIN_MATCH_COUNT = 10
    detector = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(im1, None)
    kp2, des2 = detector.detectAndCompute(im2, None)

    FLANN_INDEX_KDTREE = 0 
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50) 

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    M = np.eye(3)

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 3.0)
    else:
        print(f'Not enough matches ({len(good)} / {MIN_MATCH_COUNT}) were found! Resulting homography will be poor')


    im_warped = cv.warpPerspective(im1_transp, np.linalg.inv(M), (im2.shape[1], im2.shape[0]))

    alpha = 0.75
    im_blend = cv.addWeighted(im_warped, alpha, im2_transp, 1-alpha, 0.0)

    # Display images
    cv.imshow("Source Image", im1)
    cv.imshow("Destination Image", im2)
    cv.imshow("Warped Source Image", im_warped)
    cv.imshow("Blended warped image", im_blend)
    cv.waitKey(0)
    plt.figure(2)

    #Ordering matters here so don't fucking touch it
    points = np.array([[
            [0,0],
            [0, im1.shape[0]],
            [im1.shape[1], im1.shape[0]],
            [im1.shape[1], 0]
            ]], dtype='float32')
    points_warped = cv.perspectiveTransform(points, np.linalg.inv(M))

    mask = np.zeros_like(im2)
    cv.fillPoly(mask, pts=[np.int32(points_warped[0])], color=(1, 1, 1))
    warped_cropped = im_warped[int(max(0,min(points_warped[0,:,1]))) : int(min(im_warped.shape[0],max(points_warped[0,:,1]))), \
                        int(max(0,min(points_warped[0,:,0]))) : int(min(im_warped.shape[1],max(points_warped[0,:,0])))]
    old_cropped = im2[int(max(0,min(points_warped[0,:,1]))) : int(min(im_warped.shape[0],max(points_warped[0,:,1]))), \
                        int(max(0,min(points_warped[0,:,0]))) : int(min(im_warped.shape[1],max(points_warped[0,:,0])))]
    mask_cropped = mask[int(max(0,min(points_warped[0,:,1]))) : int(min(im_warped.shape[0],max(points_warped[0,:,1]))), \
                        int(max(0,min(points_warped[0,:,0]))) : int(min(im_warped.shape[1],max(points_warped[0,:,0])))]
    
    old_cropped = old_cropped * mask_cropped
    #cv.imshow("mask", mask)
    warped_cropped_gray = cv.cvtColor(warped_cropped, cv.COLOR_RGBA2GRAY)
    old_cropped_gray = cv.cvtColor(old_cropped, cv.COLOR_RGB2GRAY)
    cv.imshow("IMAGE WARPED CROPPED GRAY", warped_cropped_gray)
    cv.imshow("IMAGE OLD CROPPED GRAY", old_cropped_gray)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Histogram equalization before calculating the image differences

    from skimage import exposure 
    from skimage.exposure import match_histograms 
    old_cropped_gray_aux = match_histograms(old_cropped_gray, warped_cropped_gray)
    old_cropped_gray = old_cropped_gray_aux.astype(np.uint8)

    '''warped_cropped_gray = cv.equalizeHist(warped_cropped_gray)
    old_cropped_gray = cv.equalizeHist(old_cropped_gray)
    '''

    cv.imshow("IMAGE WARPED CROPPED GRAY MATCHED", warped_cropped_gray)
    cv.imshow("IMAGE OLD CROPPED EQUALIZED MATCHED", old_cropped_gray)

    # Image difference
    diff = np.zeros_like(old_cropped_gray,dtype=np.float32)

    diff = np.float32(diff)
    warped_cropped_gray = cv.GaussianBlur(warped_cropped_gray,(3,3),0)
    old_cropped_gray = cv.GaussianBlur(old_cropped_gray,(3,3),0)
    #diff = cv.subtract(np.float32(warped_cropped_gray), np.float32(old_cropped_gray))
    #diff[:,:] = (((diff[:,:] / 255.0) + 1.0) / 2.0) * 255.0
    # diff = cv.absdiff(warped_cropped_gray, old_cropped_gray)
    # diff = np.uint8(diff)
    diff = cv.absdiff(warped_cropped_gray, old_cropped_gray)
    #diff = cv.bitwise_and(warped_cropped_gray, old_cropped_gray)
    #diff = cv.bitwise_xor(diff, warped_cropped_gray)
    #diff = cv.bitwise_xor(diff, old_cropped_gray, mask_cropped)

    

    #diff = cv.absdiff(old_cropped_gray, warped_cropped_gray)
    ## TODO: LERP FROM MIN_PX--> 0 and MAX_PX-->1

    # Threshold the difference
    _,thresholded = cv.threshold(diff, int(4.25*255/10.0),255,cv.THRESH_TOZERO)
    _,mask2 = cv.threshold(thresholded, int(5.75*255/10.0),255,cv.THRESH_BINARY)
    thresholded = cv.bitwise_or(thresholded, mask2)

    #thresholded = cv.adaptiveThreshold(diff,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,21,2)

    cv.imshow("DIFFERENCE 1", diff)
    #cv.imshow("THRESHOLDED", thresholded)

    diff_hm = cv.applyColorMap(diff, cv.COLORMAP_PARULA)
    thresholded_hm = cv.applyColorMap(thresholded, cv.COLORMAP_PARULA)

    canny_old = cv.Canny(old_cropped_gray,90, 200)
    canny_new = cv.Canny(warped_cropped_gray,50, 160)

    rgb_old = cv.cvtColor(canny_old, cv.COLOR_GRAY2BGR)
    rgb_new = cv.cvtColor(canny_new, cv.COLOR_GRAY2BGR)
    rgb_new = np.float32(rgb_new)
    rgb_old = np.float32(rgb_old)
    rgb_old *= np.array((1,0,1))  # Pink-ish, light pink shade
    rgb_new *= np.array((0,1,0))        # Aggressive purple, missing texture level of purple
    rgb_new = np.uint8(rgb_new)
    rgb_old = np.uint8(rgb_old)

    # step 3: compose:

    diff_hm = cv.bitwise_xor(cv.bitwise_or(rgb_old, rgb_new), cv.bitwise_and(diff_hm, cv.bitwise_not(cv.bitwise_or(rgb_old, rgb_new))))             #(diff_hm + (rgb_old + rgb_new))
    thresholded_hm = cv.bitwise_xor(cv.bitwise_or(rgb_old, rgb_new), cv.bitwise_and(thresholded_hm, cv.bitwise_not(cv.bitwise_or(rgb_old, rgb_new))))      #(thresholded_hm + (rgb_old + rgb_new))
 
    cv.imshow("DIFFERENCE 1 HEATMAP", diff_hm)
    cv.imshow("THRESHOLDED HEATMAP", thresholded_hm)

    cv.waitKey(0)


def list_of_integers(arg):
    return arg.split(',')



# Main: Entry point
if __name__ == '__main__':

    # Create an argument parser to enable running this on SuperGlue and NNDR modes
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest='command', help='Commands to run', required=True)

    # First subparser, for Old camera position estimation from new pictures
    pose_estimation_parser = subparsers.add_parser('old_camera_estimation', help='Estimates the pose of a camera from an old photo and 2 or more photos with known calibration')
    pose_estimation_parser.add_argument('--config_file', '-c', type=str, required=False, default='config.ini', help='Config .ini file with options')
    pose_estimation_parser.add_argument('--stages', '-s', type=list_of_integers, default="1,2,3,4", help='Steps of the pipeline to perform')
    pose_estimation_parser.set_defaults(func=pose_estimation_main)

    # Second subparser, for Image Difference Mode
    image_diff_parser = subparsers.add_parser('image_diff', help='Spots differences between two similar images. For example, two images from the same building years apart.')
    image_diff_parser.add_argument('--img1', '-i1', type=str, required=True, help='First image to compute differences with respect to another image of the same scene (new picture).')
    image_diff_parser.add_argument('--img2', '-i2', type=str, required=True, help='The second image, with respect to which the differences will be calculated (old picture here).')
    image_diff_parser.set_defaults(func=image_diff_main)

    args = argparser.parse_args()
    args.func(args)

