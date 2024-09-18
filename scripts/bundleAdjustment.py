import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize as scOptim
import scipy.io as sio
import cv2



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
    th_ext2 = K_c[1] @ np.hstack((R, np.expand_dims(t, axis=1)))   #First camera

    th_ext = [th_ext1, th_ext2]
    for i in range(nCameras - 2):                               #Rest of cameras
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



# Returns a nparray containing 2*2*nPoints residuals.
# First 2*nPoints res1, then 2*nPoints res2
# Each res{i} has 2 values, first X then Y
# For example: [res1_0.x, res1_0.y, res1_1.x, res1_1.y, res2_0.x, res2_0.y, res2_1.x, res2_1.y]
def resBundleProjection(Op, x1Data, x2Data, K_c, npoints):
    '''
    TODO:
    Poner T_c1_c2 en op en la forma que dice la slide 46 de structure from motion
    sacar de op la T y la rot
    con K_c armar P
    Sacar de Op los puntos 3D
    Reproyectar los puntos a imagen con esa P
    calcular error con x1Data: res_x1 = [repr[i].x - x1Data[i].x, repr[i].y - x1Data[i].y . . .].T
    lo mismo con x2Data: res_x2
    Devolver las dos matrices columna res_x1 y res_x2
    '''

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



def matrixA(m1, m2, P1, P2):
    """Creates the matrix of equations for each pair of matches"""
    A = np.zeros((2 * m1.shape[0], 4))

    row1_m1 = np.array([P1[2, 0] * m1[0] - P1[0, 0], P1[2, 1] * m1[0] - P1[0, 1], P1[2, 2] * m1[0] - P1[0, 2],
                        P1[2, 3] * m1[0] - P1[0, 3]])
    row2_m1 = np.array([P1[2, 0] * m1[1] - P1[1, 0], P1[2, 1] * m1[1] - P1[1, 1], P1[2, 2] * m1[1] - P1[1, 2],
                        P1[2, 3] * m1[1] - P1[1, 3]])
    row1_m2 = np.array([P2[2, 0] * m2[0] - P2[0, 0], P2[2, 1] * m2[0] - P2[0, 1], P2[2, 2] * m2[0] - P2[0, 2],
                        P2[2, 3] * m2[0] - P2[0, 3]])
    row2_m2 = np.array([P2[2, 0] * m2[1] - P2[1, 0], P2[2, 1] * m2[1] - P2[1, 1], P2[2, 2] * m2[1] - P2[1, 2],
                        P2[2, 3] * m2[1] - P2[1, 3]])

    A[0, :] = row1_m1
    A[1, :] = row2_m1
    A[2, :] = row1_m2
    A[3, :] = row2_m2
    return A



def triangulate3D(points1, points2, Pose1, Pose2):
    X_w_my_a = np.zeros((3, points1.shape[1]))
    for ia in range(points1.shape[1]):
        A1a = matrixA(points1[:, ia], points2[:, ia], Pose1, Pose2)
        ua, sa, vaT = np.linalg.svd(A1a)
        Va = vaT.T
        sol_a = Va[:, 3]
        sol_a = sol_a[0:3] / sol_a[3]
        X_w_my_a[:, ia] = sol_a
    return X_w_my_a



def z_positive_v2(R, t, X_w):
    val_2 = R[2,:] @ (X_w[:3] - t)
    fr_2 = np.all([val_2 >= 0], axis = 0)

    val_1 = np.array([0,0,1]) @ X_w[:3]
    fr_1 = np.all([val_1 >= 0], axis = 0)

    res = np.zeros(fr_1.shape)
    for i in range(X_w.shape[1]):
        if fr_1[i] == True and fr_2[i] == True:
            res[i] = 2 
        
    score = np.sum(res == 2)
    return score



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
    T_w_c = np.zeros((4, 4), dtype=np.float32) #TODO da problemas en P4
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c



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



def DLT(points2d,points3d):
    points2d = np.vstack([points2d, np.ones((1, points2d.shape[1]))])
    points3d = np.vstack([points3d, np.ones((1, points3d.shape[1]))])
    A = []
    for i in range(points2d.shape[1]):

        A.append([
            -points3d[0, i], -points3d[1, i], -points3d[2, i], -points3d[3, i],
            0, 0, 0, 0,
            points2d[0, i] * points3d[0, i], points2d[0, i] * points3d[1, i], points2d[0, i] * points3d[2, i], points2d[0, i] * points3d[3, i]
        ])

        A.append([
            0, 0, 0, 0,
            -points3d[0, i], -points3d[1, i], -points3d[2, i], -points3d[3, i],
            points2d[1, i] * points3d[0, i], points2d[1, i] * points3d[1, i], points2d[1, i] * points3d[2, i],
            points2d[1, i] * points3d[3, i]
        ])
    A = np.array(A)
    u, s, V = np.linalg.svd(A)
    res = V.T[:, -1]
    P = res.reshape((3, 4))
    return P



if __name__ == '__main__':
# --------------------------------------- 2.- Bundle adjustment for 2 views --------------------------------------- #
    #Read the images
    path_image_1 = '../new_pictures/torre_belem_5.jpg'
    path_image_2 = '../new_pictures/torre_belem_8.jpg'
    path_image_3 = '../old_pictures/torre_de_belem_1930_1980.jpg'
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)
    do_interactive = True                       # FIXME CAMBIA ESTO TROZO DE MIERDA INFECTA
    T_w_c1 = np.eye(4, 4)                       # We don't know its ground truth pose

    # Compute the Fundamental Matrix as we did in Lab Session 2


    inliers_x1_xold = np.loadtxt('../computed_inliers/common_inliers_im1_OLD.txt')
    inliers_x1_x2 = np.loadtxt('../computed_inliers/common_inliers_im1_im2.txt')

    # Extract the points from the matches
    x1 = (inliers_x1_x2[:, 0:2]).T
    x2 = (inliers_x1_x2[:, 2:4]).T
    x3 = (inliers_x1_xold[:, 2:4]).T
    num_points_to_use = x1.shape[1]

    print("x1 shape: ", x1.shape)

    K_c_cams_1_2 = np.loadtxt('../camera_data/calibration_matrix.txt')
    
    A = correspondences(x1, x2)     # NOTE: correspondences(x1, x2) --> F_21 (2 seen/in base 1 or 1 ------> 2)
    F = computeFundamental(A)
    # TODO: If this behaves weirdly, load F_21.txt y palante
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
        Rp90 = -U @ W @ V
    
    Rm90 = U @ W.T @ V
    if(np.linalg.det(Rm90) < 0):
        Rm90 = -U @ W.T @ V

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

    # We have to test which one is the one, which one has more points in front of camera
    pos_count_1 = z_positive_v2(Rp90, t, X_w_1)
    pos_count_2 = z_positive_v2(Rp90, -t, X_w_2)
    pos_count_3 = z_positive_v2(Rm90, t,  X_w_3)
    pos_count_4 = z_positive_v2(Rm90, -t,  X_w_4)

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
    if best_pos_count == pos_count_1:
        chosen_t = t
        chosen_R = Rp90
        X_3D = X_w_1
        print("Initial Guess for camera 2 is +90, t")
    elif best_pos_count == pos_count_2:
        chosen_t = -t
        chosen_R = Rp90
        X_3D = X_w_2
        print("Initial Guess for camera 2 is +90, -t")
    elif best_pos_count == pos_count_3:
        chosen_t = t
        chosen_R = Rm90
        X_3D = X_w_3
        print("Initial Guess for camera 2 is -90, t")
    else:          #if best_pos_count == pos_count_4:
        chosen_t = -t
        chosen_R = Rm90
        X_3D = X_w_4
        print("Initial Guess for camera 2 is -90, -t")

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
        drawRefSystem(ax, T_w_c1, '-', 'C1')
        #drawRefSystem(ax, wTc1 @ np.linalg.inv(c2Tc1_Op), '-', 'C2_BA')
        drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1_optimized), '-', 'Camera2 optimized after BA')

        X_3D_plot_optimized = T_w_c1 @ (X_3D_optimized).T

        ax.scatter(X_3D_plot_optimized[0, :], X_3D_plot_optimized[1, :], X_3D_plot_optimized[2, :], marker='.')
        plotNumbered3DPoints(ax, X_3D_plot_optimized, 'b', 0.1)
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
    print("X_3D SHAPE: \n", X_3D.shape)

    P3_estimation = DLT(x3, X_3D)
    M = P3_estimation[:,:-1]
    K_c_cam_3, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(np.sign(np.linalg.det(M)) * P3_estimation)
    
    
    X_3D_op_PnP = (X_3D_optimized[:, 0:3]).astype('float32')
    imagePoints = (np.ascontiguousarray(x3[0:2, 0:num_points_to_use].T).reshape((num_points_to_use, 1, 2))).astype('float32')
    K_c_cams_1_2_ = K_c_cams_1_2.astype('float32')

    print("SHAPE imagePoints (should be Nx2): ", imagePoints.shape)
    print("SHAPE objectPoints (should be Nx3): ", X_3D_op_PnP.shape)

    coeffs = []
    _, rvec, tvec = cv2.solvePnP(objectPoints=X_3D_op_PnP, 
        imagePoints=imagePoints, cameraMatrix=K_c_cam_3, distCoeffs=np.array(coeffs),flags=cv2.SOLVEPNP_EPNP)

    R_c3_c1 = recover_R(rvec)
    t_c3_c1 = tvec
    T_c3_c1_PnP = ensamble_T(R_c3_c1, t_c3_c1.reshape((3,)))

    if do_interactive:
        fig = plt.figure(5)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drawRefSystem(ax, T_w_c1, '-', 'C1_GT')
        drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c3_c1_PnP), '-', 'C3_PnP')
        plt.title('3D camera poses cameras 1 and C3 PnP')
        plt.draw()
        plt.show()

############## 4.- Bundle Adjustment from 3 views ##############

# Final part of the lab session: Generalize BA for n cameras and run it for 3 (2, but 3, ok?)

    initial_guess_SO3_R_c3_c1 = rvec

    Op = np.hstack((
        np.hstack((th_phi, RxRyRz)), 
        np.hstack(((t_c3_c1[:,0]).flatten(), (initial_guess_SO3_R_c3_c1[:,0]).flatten())), 
        np.array(X_3D_points[0:num_points_to_use*3])
    ))

    xData = np.vstack((np.vstack((x1, x2)), x3))
    res = resBundleProjection_N(Op, xData, [K_c_cams_1_2, K_c_cams_1_2, K_c_cam_3], 3, num_points_to_use)
    print("Length res BA 3 cams: ", len(res))

    # Now, perform the optimization 
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
    P3_optimized_N = K_c_cams_1_2 @ np.concatenate((R_c3_c1_optimized_N, np.expand_dims(t_c3_c1_optimized_n, axis=1)), axis=1) #revisar el concatenate con t_c3_c1_optimized_n
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
        drawRefSystem(ax, T_w_c1, '-', 'C1_GT')
        drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1_optimized_N), '-', 'C2_BA')
        drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c3_c1_optimized_N), '-', 'C3_BA')

        _3D_plot_optimized_N = T_w_c1 @ (X_3D_optimized_N).T

        ax.scatter(_3D_plot_optimized_N[0, :], _3D_plot_optimized_N[1, :], _3D_plot_optimized_N[2, :], marker='.')
        plotNumbered3DPoints(ax, _3D_plot_optimized_N, 'b', 0.1)

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