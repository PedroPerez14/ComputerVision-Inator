#####################################################################################
#
# MRGCV Unizar - Computer vision - Final Course Project
#
# Title: Fundamental matrix estimation using RANSAC from Lab Session 3 with changes
#
# Date: 12 January 2024
#
#####################################################################################
#
# Authors: Pedro José Pérez García

# Version: 1.0
#
#####################################################################################

import argparse
import random
import cv2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scAlg

##############################################################################################################################

def buildCorrespondences(dMatchesList, keypoints_sift_1, keypoints_sift_2):
    correspondenceList = []
    for match in dMatchesList:
        (x1, y1) = keypoints_sift_1[match.queryIdx].pt
        (x2, y2) = keypoints_sift_2[match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])
    corr_s = np.array(correspondenceList)
    return corr_s



def NNDR(keypoints_sift_1, descriptors_1, keypoints_sift_2, descriptors_2, _dist, _dR):
    """ Perform NNDR matching """
    from SIFTmatching import matchWith2NDRR_v2
    from SIFTmatching import indexMatrixToMatchesList

    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    
    distRatio = _dR
    minDist = _dist

    matchesList = matchWith2NDRR_v2(descriptors_1, descriptors_2, distRatio, minDist)
    
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    return dMatchesList



def plotStuff(image_pers_1, image_pers_2, keypoints_sift_1, keypoints_sift_2, dMatchesList, title):
    img_Matched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2,
                                         dMatchesList,
                                         None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.title(title)
    plt.imshow(img_Matched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()



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
    return abs(d_match1_l1 + d_match2_l2) / 2

def RANSAC(correspondences, th, im1, kp1, im2, kp2, dMatchesList, iters, quiet, nRandomItems=8):
    f = None
    inliers = []
    inlier_ids = []
    print("CORRESPONDENCES[0]: ", correspondences[0])
    for i in range(iters):
        tentative_f = None
        tentative_inliers = []
        tentative_inlier_ids = []

        # Get nRandomItems (8 for fundamental) points at random
        sampled_ids = random.sample(range(correspondences.shape[0]), nRandomItems)
        sampled_ids.sort()
        random_set = correspondences[sampled_ids]
      
        # Compute the Homography as we did in the last lab session
        # tentative_H = computeHomography(random_set)
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
            plotStuff(im1, im2, kp1, kp2, mylist, 'Matches for creating hypothesis F in iteration '+str(i))
            print('This iteration\'s hypothesis is: \n', tentative_f)
            mylist = [dMatchesList[id] for id in tentative_inlier_ids]
            plotStuff(im1, im2, kp1, kp2, mylist, 'Inliers obtained with hypothesis F in iteration '+str(i))


        if len(tentative_inliers) > len(inliers):
            inliers = tentative_inliers
            inlier_ids = tentative_inlier_ids
            f = tentative_f

    # Recompute f using all of the inliers
    # TODO: Should I do this?
    #final_set = correspondences[inlier_ids]
    #f = computeFundamental(final_set)

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



def plot_epipolar_lines(F, img1, img2):
    ''' Plots 2 epipolar lines, estimated epipole and ground truth epipole '''
    scale_width = 640 / img1.shape[1]
    scale_height = 480 / img1.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img1.shape[1] * scale)
    window_height = int(img1.shape[0] * scale)
    cv2.namedWindow('image 1 - Click a point', cv2.WINDOW_NORMAL)  
    cv2.resizeWindow('image 1 - Click a point', window_width, window_height)
    # set mouse callback function for window
    cv2.setMouseCallback('image 1 - Click a point', mouse_callback)
    cv2.imshow('image 1 - Click a point', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

'''
plt.figure()
plt.title(title)
plt.imshow(img)
plt.draw()  # We update the figure display
# Plot 2 epipolar lines
plt.plot([0, (-l1[2] / l1[0])], [(-l1[2] / l1[1]), 0],  # en x (0, -c/a) en y (-c/b, 0)
            color='green', linewidth=1)
plt.plot([0, (-l2[2] / l2[0])], [(-l2[2] / l2[1]), 0],  # en x (0, -c/a) en y (-c/b, 0)
            color='red', linewidth=1)

# Plot epipole in estimated F
if(np.linalg.matrix_rank(F.T) == 2):
    u,s,vT = np.linalg.svd(F.T)
    v=vT.T
    e=v[:,-1]
    plt.plot(e[0]/e[2], e[1]/e[2], 'x', color='blue')
else:
    print("Can't calculate epipole! F matrix has rank != 2!")

# Plot epipole in ground truth F
if(np.linalg.matrix_rank(F_gt.T) == 2):
    u_,s_,vT_ = np.linalg.svd(F_gt.T)
    v_=vT_.T
    e_=v_[:,-1]
    plt.plot(e_[0]/e_[2], e_[1]/e_[2], 'x', color='red')
else:
    print("Can't calculate epipole! F_gt matrix has rank != 2!")

plt.draw()  # We update the figure display
plt.waitforbuttonpress()
plt.close(title)
'''



def fundamentalFromCamera():
    
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')
    K_c = np.loadtxt('K_c.txt')

    t_w_c1 = T_w_c1[0:-1,-1]
    t_w_c2 = T_w_c2[0:-1,-1]
    r_w_c1 = T_w_c1[0:-1,0: -1]
    r_w_c2 = T_w_c2[0:-1,0:-1]
    
    t_c2_c1 = r_w_c2.T @ (t_w_c1 - t_w_c2)
    R_c2_c1 = r_w_c2.T @ r_w_c1

    t_c2_c1_x = np.zeros((3, 3))
    t_c2_c1_x[0, 1] = -t_c2_c1[2]
    t_c2_c1_x[0, 2] = t_c2_c1[1]
    t_c2_c1_x[1, 0] = t_c2_c1[2]
    t_c2_c1_x[1, 2] = -t_c2_c1[0]
    t_c2_c1_x[2, 0] = -t_c2_c1[1]
    t_c2_c1_x[2, 1] = t_c2_c1[0]
    E = t_c2_c1_x @ R_c2_c1
    F = np.linalg.inv(K_c).T @ E @ np.linalg.inv(K_c)
    
    #F = np.loadtxt('F_21_test.txt')
    return F



def nndr_args(args):
    """ Nearest Neighbor Distance Ratio matching method """
    
    # Read images
    path_image_1 = args.img1
    path_image_2 = args.img2
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    # Perform NNDR
    dMatchesList = NNDR(keypoints_sift_1, descriptors_1, keypoints_sift_2, descriptors_2, args.minDist, args.distRatio)

    # Plot the obtained matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.title('First 100 matches with NNDR')
    plt.draw()
    plt.waitforbuttonpress()

    # Now, run the RANSAC
    correspondences = buildCorrespondences(dMatchesList, keypoints_sift_1, keypoints_sift_2)
    fundamental, inliers, _ = RANSAC(correspondences, args.ransacThreshold, image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList, args.iters, args.quiet)

    # Final plots for inliers in image 1 and image 2 by transforming with our F
    print('The estimated Fundamental Matrix by RANSAC is ', fundamental)
    print("Final inliers count: ", len(inliers))

    img1 = cv2.cvtColor(cv2.imread(path_image_1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(path_image_2), cv2.COLOR_BGR2RGB)

    # {match1, match2} is an inlier match, and {match3, match4} is another
    match1 = np.array((inliers[0].item(0), inliers[0].item(1), 1))
    match2 = np.array((inliers[0].item(2), inliers[0].item(3), 1))
    match3 = np.array((inliers[1].item(0), inliers[1].item(1), 1))
    match4 = np.array((inliers[1].item(2), inliers[1].item(3), 1))
    l21 = fundamental.T @ match1     # Epipolar line for match1 in image2
    l11 = fundamental @ match2   # Epipolar line for match2 in image1
    l22 = fundamental.T @ match3     # Epipolar line for match1 in image2
    l12 = fundamental @ match4   # Epipolar line for match2 in image1

    # Load the camera parameters to get the "ground truth" F and epipole
    #plot_epipolar_lines(fundamental.T, F_gt.T, l11, l12, img1, 'Epipolar lines and epicenters (Red= GT) for 2 inliers in image 2 seen in camera 1')
    plot_epipolar_lines(fundamental, img1, img2)
    plt.close()
    #plot_epipolar_lines(fundamental, F_gt, l21, l22, img2, 'Epipolar lines and epicenters (Red = GT) for 2 inliers in image 1 seen in camera 2')
    cv2.destroyAllWindows()



def npztomatches(desc1, desc2, keypoints0, keypoints1, match):
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        if match[kDesc1] >= 0:
            dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
            indexSort = np.argsort(dist)
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches



def loadSuperglueData(path):
    from SIFTmatching import indexMatrixToMatchesList

    # First, open and load file
    npz = np.load(args.pathMatches)
    _matches = npz['matches']
    key1 = npz['keypoints0']
    key2 = npz['keypoints1']
    desc1 = npz['descriptors0'].T
    desc2 = npz['descriptors1'].T

    # Second, transform the data into an appropiate format
    key_points1 = cv2.KeyPoint_convert(key1)
    key_points2 = cv2.KeyPoint_convert(key2)

    matches = npztomatches(desc1, desc2, key1, key2, _matches)
    dMatchesList = indexMatrixToMatchesList(matches)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    return dMatchesList, key_points1, key_points2, desc1, desc2



def superglue_args(args):
    #Load the data and the images
    dMatchesList, keypoints_sg_1, keypoints_sg_2, descriptors_1, descriptors_2 = loadSuperglueData(args.pathMatches)
    image_pers_1 = cv2.imread(args.img1)
    image_pers_2 = cv2.imread(args.img2)
    '''
    img_rows = image_pers_1.shape[1]
    img_cols = image_pers_1.shape[0]
    img_rows_2 = image_pers_2.shape[1]
    img_cols_2 = image_pers_2.shape[0]
    image_downsize_factor = 2
    new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
    new_img_size_2 = (int(img_rows_2 / image_downsize_factor), int(img_cols_2 / image_downsize_factor))
    
    image_pers_1 = cv2.resize(image_pers_1, new_img_size, interpolation = cv2.INTER_CUBIC)
    image_pers_2 = cv2.resize(image_pers_2, new_img_size_2, interpolation = cv2.INTER_CUBIC)
    '''

    # Plot the obtained matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sg_1, image_pers_2, keypoints_sg_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.title('First 100 matches with Superglue')
    plt.draw()
    plt.waitforbuttonpress()

    # Now, run the RANSAC
    correspondences = buildCorrespondences(dMatchesList, keypoints_sg_1, keypoints_sg_2)
    fundamental, inliers, id_list_for_plotting = RANSAC(correspondences, args.ransacThreshold, image_pers_1, keypoints_sg_1, image_pers_2, keypoints_sg_2, dMatchesList, args.iters, args.quiet)

    # Final plots for inliers in image 1 and image 2 by transforming with our F
    print('The estimated Fundamental Matrix by RANSAC is ', fundamental)
    print("Final inliers count: ", len(inliers))

    # Plot again the inliers
    plotStuff(image_pers_1, image_pers_2, keypoints_sg_1, keypoints_sg_2, id_list_for_plotting, 'Final '+ str(len(inliers)) + ' inliers obtained after '+str(args.iters)+' iterations')

    img1 = cv2.cvtColor(cv2.imread(args.img1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(args.img2), cv2.COLOR_BGR2RGB)

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
    #plot_epipolar_lines(fundamental.T, F_gt.T, l11, l12, img1, 'Epipolar lines and epicenters (Red= GT) for 2 inliers in image 2 seen in camera 1')
    plot_epipolar_lines(fundamental, img1, img2)
    plot_epipolar_lines(fundamental.T, img2, img1)
    plt.close()
    #plot_epipolar_lines(fundamental, F_gt, l21, l22, img2, 'Epipolar lines and epicenters (Red = GT) for 2 inliers in image 1 seen in camera 2')
    cv2.destroyAllWindows()


# Main: Entry point
if __name__ == '__main__':
    # Create an argument parser to enable running this on SuperGlue and NNDR modes
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest='command', help='Commands to run', required=True)

    # First subparser, for nndr mode
    nndr_parser = subparsers.add_parser('nndr', help='Estimates the Fundamental matrix with matches computed by using Nearest Neighbor Distance Ratio')
    nndr_parser.add_argument('--img1', '-i1', type=str, required=False, default='image1.png', help='First image to use')
    nndr_parser.add_argument('--img2', '-i2', type=str, required=False, default='image2.png', help='Second image to use')
    nndr_parser.add_argument('--iters', '-i', type=int, required=False, default=100, help='RANSAC iterations')
    nndr_parser.add_argument('--minDist', '-d', type=float, required=False, default=500, help='Distance between descriptors to be used for matching')
    nndr_parser.add_argument('--distRatio', '-r', type=float, required=False, default=0.6, help='Threshold ratio between the distances in the first and second matches')
    nndr_parser.add_argument('--ransacThreshold', '-t', type=float, required=False, default=3.0, help='Threshold (pixels) to accept or reject a hypothesis based on transfer error')
    nndr_parser.add_argument('--quiet', '-q', action='store_true', help='Do not output tentative matches or inliers during ransac, only results', default=False)
    nndr_parser.set_defaults(func=nndr_args)

    # Second subparser, for SuperGlue mode
    superglue_parser = subparsers.add_parser('superglue', help='Estimates the Fundamental matrix loading matches generated with SuperGlue')
    superglue_parser.add_argument('--img1', '-i1', type=str, required=False, default='image1.png', help='First image to use')
    superglue_parser.add_argument('--img2', '-i2', type=str, required=False, default='image2.png', help='Second image to use')
    superglue_parser.add_argument('--iters', '-i', type=int, required=False, default=2000, help='RANSAC iterations')
    superglue_parser.add_argument('--pathMatches', '-m', type=str, required=False, default='image1_image2_matches.npz', help='File to load matches from')
    superglue_parser.add_argument('--ransacThreshold', '-t', type=float, required=False, default=12.0, help='Threshold (pixels) to accept or reject a hypothesis based on transfer error')
    superglue_parser.add_argument('--quiet', '-q', action='store_true', help='Do not output tentative matches or inliers during ransac, only results', default=False)
    superglue_parser.set_defaults(func=superglue_args)

    args = argparser.parse_args()
    args.func(args)