import numpy as np
import cv2
import os
from os.path import basename
import os.path
import glob
import time
import pickle
from collections import deque
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.measurements as scipymeas


"""
Object to calculate the stereo calibration necessary to match images using the BM object
"""
class Calibration():
    def __init__(self, path, left_template, right_template, square_size_in_mm=100, toskip=[]):
        
        # Path
        self.path = path

        # Load filenames for images
        self.imagesL = glob.glob(self.path + left_template)
        self.imagesR = glob.glob(self.path + right_template)
        self.imagesL.sort()
        self.imagesR.sort()
        assert len(self.imagesL) == len(self.imagesR)
        n_before = len(self.imagesL)

        # Removing the ones manually specified as unsuitable
        # This won't work if the template is not the standard one. Modify manually in case.
        for i in toskip:
            sL = self.path + "left_{:03d}.png".format(i)
            sR = self.path + "right_{:03d}.png".format(i)
            if sL in self.imagesL:
                self.imagesL.remove(sL)
                self.imagesR.remove(sR)
        self.imagesL.sort()
        self.imagesR.sort()
        assert len(self.imagesL) == len(self.imagesR)
        n_after = len(self.imagesL)
        print("{} images loaded, {} remaining after manual selection".format(n_before, n_after))

        # Termination criteria
        self.termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS +
                                            cv2.TERM_CRITERIA_MAX_ITER,
                                            30,
                                            0.001)
        self.termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS +
                                                cv2.TERM_CRITERIA_MAX_ITER,
                                                100,
                                                0.001)      # chubby guy: 1e-5

        # Pattern specs
        self.patternX = 6
        self.patternY = 9
        self.square_size_in_mm = square_size_in_mm

        # Arrays to store object points and image points from all the images
        self.objpoints = [] # 3d point in real world space
        self.imgpointsR = [] # 2d points in image plane
        self.imgpointsL = [] # 2d points in image plane
        self.image_size = None

        # Filenames
        self.calibration = None
        self.calibration_filename = "calibration.p"

        # Windows
        self.windowNameL = "LEFT Camera"
        self.windowNameR = "RIGHT Camera"
        self.ratio = 1920/1208
        self.wsize = 800
        cv2.namedWindow(self.windowNameL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowNameL, self.wsize, int(self.wsize/self.ratio))
        cv2.moveWindow(self.windowNameL, 0, 0)
        cv2.namedWindow(self.windowNameR, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowNameR, self.wsize, int(self.wsize/self.ratio))
        cv2.moveWindow(self.windowNameR, self.wsize+100, 0)


    def calibrate(self, visual=False, window_timeout=100, save=False):

        # Save or not
        self.save = save

        """
        Intrinsic parameters
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.patternX*self.patternY,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.patternX,0:self.patternY].T.reshape(-1,2)
        objp = objp * self.square_size_in_mm

        # count number of chessboard detection (across both images)
        chessboard_pattern_detections = 0
        chessboard_pattern_detections_accepted = 0

        for i in range(len(self.imagesL)):
            imgL = cv2.imread(self.imagesL[i])
            imgR = cv2.imread(self.imagesR[i])
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            self.image_size = grayL.shape[::-1]

            # Find the chess board corners
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE #| cv2.CALIB_CB_FAST_CHECK
            retR, cornersL = cv2.findChessboardCorners(imgL, (self.patternX, self.patternY), flags)
            retL, cornersR = cv2.findChessboardCorners(imgR, (self.patternX, self.patternY), flags)

            if retR and retL:
                chessboard_pattern_detections += 1

                # refine corner locations to sub-pixel accuracy and then
                corners_sp_L = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), self.termination_criteria_subpix)
                corners_sp_R = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), self.termination_criteria_subpix)

                # Draw and display the corners
                drawboardL = cv2.drawChessboardCorners(imgL, (self.patternX, self.patternY), corners_sp_L, retL)
                drawboardR = cv2.drawChessboardCorners(imgR, (self.patternX, self.patternY), corners_sp_R, retR)

                if visual:
                    text = 'Image {}: detecting chessboard pattern'.format(self.imagesL[i])
                    cv2.putText(drawboardL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)
                    text = 'Image {}: detecting chessboard pattern'.format(self.imagesR[i])
                    cv2.putText(drawboardR, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)
                    cv2.imshow(self.windowNameL, drawboardL)
                    cv2.imshow(self.windowNameR, drawboardR)
                    key = cv2.waitKey(window_timeout)

                # Add to global list
                self.imgpointsL.append(corners_sp_L)
                self.imgpointsR.append(corners_sp_R)
                self.objpoints.append(objp)
                chessboard_pattern_detections_accepted += 1

        print("{} images were accepted, now using them to calibrate each camera separately...".format(chessboard_pattern_detections_accepted))

        # Perform calibration on both cameras - uses [Zhang, 2000]
        ret, self.mtxL, self.distL, self.rvecsL, self.tvecsL = cv2.calibrateCamera(self.objpoints, self.imgpointsL, self.image_size, None, None)
        ret, self.mtxR, self.distR, self.rvecsR, self.tvecsR = cv2.calibrateCamera(self.objpoints, self.imgpointsR, self.image_size, None, None)

        # Check results
        if visual:
            print("Now showing the same images undistorted to check if everything makes sense...")
            for i in range(len(self.imagesL)):
                imgL = cv2.imread(self.imagesL[i])
                imgR = cv2.imread(self.imagesR[i])
                undistortedL = cv2.undistort(imgL, self.mtxL, self.distL, None, None)
                undistortedR = cv2.undistort(imgR, self.mtxR, self.distR, None, None)
                text = 'Image {}: this should be undistorted'.format(self.imagesL[i])
                cv2.putText(undistortedL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)
                text = 'Image {}: this should be undistorted'.format(self.imagesR[i])
                cv2.putText(undistortedR, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)
                cv2.imshow(self.windowNameL, undistortedL);
                cv2.imshow(self.windowNameR, undistortedR);
                key = cv2.waitKey(window_timeout)
            
        # Show mean re-projection error of the object points onto the image(s)
        tot_errorL = 0
        for i in range(len(self.objpoints)):
            imgpointsL2, _ = cv2.projectPoints(self.objpoints[i], self.rvecsL[i], self.tvecsL[i], self.mtxL, self.distL)
            errorL = cv2.norm(self.imgpointsL[i], imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
            tot_errorL += errorL
        print("Left re-projection error: ", tot_errorL/len(self.objpoints))

        tot_errorR = 0
        for i in range(len(self.objpoints)):
            imgpointsR2, _ = cv2.projectPoints(self.objpoints[i], self.rvecsR[i], self.tvecsR[i], self.mtxR, self.distR)
            errorR = cv2.norm(self.imgpointsR[i],imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
            tot_errorR += errorR
        print("Right re-projection error: ", tot_errorR/len(self.objpoints))

        """
        Extrinsic parameters
        """
        # this takes the existing calibration parameters used to undistort the individual images as
        # well as calculated the relative camera positions - represented via the fundamental matrix, F
        # alter termination criteria to (perhaps) improve solution - ?
        
        # Set flags
        # Documentation here: https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1, InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2, Size imageSize, OutputArray R, OutputArray T, OutputArray E, OutputArray F, int flags,TermCriteria criteria)
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        
        rms_stereo, \
        self.camera_matrix_l, \
        self.dist_coeffs_l, \
        self.camera_matrix_r, \
        self.dist_coeffs_r, \
        self.R, \
        self.T, \
        self.E, \
        self.F = cv2.stereoCalibrate(self.objpoints, self.imgpointsL, self.imgpointsR,
                                     self.mtxL,
                                     self.distL,
                                     self.mtxR,
                                     self.distR,
                                     self.image_size,
                                     criteria=self.termination_criteria_extrinsics,
                                     flags=flags)
        
        print("Stereo RMS left to right re-projection error: {}".format(rms_stereo))

        """
        Rectification
        """        
        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(self.camera_matrix_l,
                                                    self.dist_coeffs_l,
                                                    self.camera_matrix_r,
                                                    self.dist_coeffs_r,
                                                    self.image_size,
                                                    self.R,
                                                    self.T)

        # compute the pixel mappings to the rectified versions of the images
        self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(self.camera_matrix_l, self.dist_coeffs_l, RL, PL, self.image_size, cv2.CV_32FC1)
        self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(self.camera_matrix_r, self.dist_coeffs_r, RR, PR, self.image_size, cv2.CV_32FC1)

        # Check rectification
        if visual:
            print("Now showing the same images undistorted and recified to check if everything makes sense...")
            for i in range(len(self.imagesL)):
                imgL = cv2.imread(self.imagesL[i])
                imgR = cv2.imread(self.imagesR[i])
                undistorted_rectifiedL = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
                undistorted_rectifiedR = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)
                text = 'Image {}: this should be undistorted and rectified'.format(self.imagesL[i])
                cv2.putText(undistorted_rectifiedL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)
                text = 'Image {}: this should be undistorted and rectified'.format(self.imagesR[i])
                cv2.putText(undistorted_rectifiedR, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)
                cv2.imshow(self.windowNameL, undistorted_rectifiedL)
                cv2.imshow(self.windowNameR, undistorted_rectifiedR)
                key = cv2.waitKey(window_timeout)

        cv2.destroyAllWindows()

        # Save
        if self.save:
            self.calibration = { "mapL1": self.mapL1,
                                 "mapL2": self.mapL2,
                                 "mapR1": self.mapR1,
                                 "mapR2": self.mapR2,
                                 "Q": Q}
            pickle.dump(self.calibration, open(self.path + self.calibration_filename, "wb"))
https://nerian.com/support/resources/calculator/
        return
        

if __name__ == '__main__':        
    print("OpenCV version: {}".format(cv2.__version__))
    
    # calibration_folder = '../videos/20171220_stereo_calibration_120deg_2/calibration_frames/'
    # calibration_folder = '../videos/20171220_stereo_calibration_120deg_2/calibration_frames_small/'
    # calibration_folder = '../videos/20180109_stereo_calibration_60deg_250mm/calibration_frames/'
    calibration_folder = '../videos/20180111_stereo_calibration_60deg_120mm/calibration_frames/'
    toskip = []

    # square_size_in_mm = 40 when using the A3 checkerboard, 100 when using the A0 checkerboard
    cameras = Calibration(calibration_folder,
                          toskip=toskip,
                          #left_template='calibration_left_*_cropped.png',
                          #right_template='calibration_right_*_cropped.png')
                          left_template='calibration_left_*.png',
                          right_template='calibration_right_*.png',
                          square_size_in_mm=100)
    cameras.calibrate(visual=True, save=True)
