import numpy as np
import glob
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def camera_calibration(img_size,
                       calibration_filenames='camera_calibration/calibration*.jpg',
                       nx=9,                                                             # X number of corners in the checkboard
                       ny=6,                                                             # Y number of corners in the checkboard
                       verbose=False):

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Make a list of calibration images
    images = glob.glob(calibration_filenames)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if verbose:
            print('Calibration image ' + fname + ': ' + str(ret))

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
                
    # Compute calibration coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    if verbose:
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            ax1.imshow(img)
            img_undistorted = undistort_image(img, mtx, dist)
            ax2.imshow(img_undistorted)
    
    # Save for future use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera_calibration/calibration.p", "wb"))
    
    return mtx, dist


def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def warp_image(img, src, dst):
               
    img_size = (img.shape[1], img.shape[0])
    h = img.shape[0]
    w = img.shape[1]
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv


def region_of_interest(img):
    
    # Parameters of the region of interest
    height = img.shape[0]
    width = img.shape[1]
    roi_h_center = width / 2
    roi_v_center = 420
    roi_flat_size = width / 20
    roi_v_side_left = 0
    roi_v_side_right = roi_v_side_left
    roi_bottom = height
    
    # Vertices of the polygon describing the region of interest
    vertices = np.array([[(0,roi_bottom),
                          (0, height - roi_v_side_left),
                          (roi_h_center-roi_flat_size/2, roi_v_center),
                          (roi_h_center+roi_flat_size/2, roi_v_center),
                          (width, height - roi_v_side_right),
                          (width,roi_bottom)]],
                        dtype=np.int32)
    
    # Defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img, mask)
