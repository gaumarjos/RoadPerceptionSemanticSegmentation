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

# Import file in semantic segmentation to automatize labeling
# sys.path.append("../semantic_segmentation")
# import mapillary_labels


"""
DOCUMENTATION

Initial
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
https://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
MAYBE https://github.com/opencv/opencv/blob/master/samples/python/morphology.py
https://github.com/julienr/cvscripts/tree/master/rectification

Great references
https://github.com/tobybreckon/python-examples-cv/blob/master/stereo_sgbm.py
https://github.com/erget/StereoVision
https://erget.wordpress.com/2014/02/01/calibrating-a-stereo-camera-with-opencv/

Filtering
https://docs.opencv.org/3.1.0/d3/d14/tutorial_ximgproc_disparity_filtering.html

ADCensus
https://www.youtube.com/watch?v=MZsSTpS-XGI
"""


"""
Object to match two images (corrected accordingly to the calibration obatined by the "Calibration" object) and compute the disparity map.
"""
class BM():

    """
    Initialization
    """
    def __init__(self,
                 calibration,
                 disparity_crop=[0, -1, 0, -1],
                 distance_calibration_poly=np.asarray([0, 1, 0]),
                 distance_calibration_invert=0):
        # Matcher
        window_size = 5
        self.minDisparity = 0
        self.numDisparities = 128
        self.blockSize = window_size  # the old SADWindowSize
        self.P1 = 8 * 3 * (window_size+2)**2  # used to be window_size**2 as suggested in the documentation
        self.P2 = 32 * 3 * (window_size+2)**2
        self.disp12MaxDiff = 1
        self.uniquenessRatio = 10
        self.speckleWindowSize = 100
        self.speckleRange = 32
        self.preFilterCap = 31
        
        # Filter in use
        self.use_wls_filter = 1

        # Speckle Filter
        self.speckle_maxSpeckleSize = 10
        self.speckle_maxDiff = 4

        # WLS Filter
        self.wls_lambda = 500000
        self.wls_sigma = 1.2

        # Disparity crop
        self.crop_left = disparity_crop[0]
        self.crop_right = disparity_crop[1]
        self.crop_top = disparity_crop[2]
        self.crop_bottom = disparity_crop[3]

        # Distance calibration polynomial
        self.distance_calibration_poly = distance_calibration_poly
        self.distance_calibration_invert = distance_calibration_invert

        # Create matchers
        self._create_matchers()
        
        # Calibration
        """
        Old default
        h, w = imgL.shape[:2]
        f = 0.8*w                          # guess for focal length
        self.Q = Q = np.float32([[1, 0, 0, -0.5*w],
                                 [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                                 [0, 0, 0,     -f], # so that y-axis looks up
                                 [0, 0, 1,      0]])
        """
        self.Q = calibration["Q"]
        self.mapL1 = calibration["mapL1"]
        self.mapL2 = calibration["mapL2"]
        self.mapR1 = calibration["mapR1"]
        self.mapR2 = calibration["mapR2"]


    """
    Create matchers
    Internal method called once at initialization and multiple times by the tuner
    """
    def _create_matchers(self):
        self.matcherL = cv2.StereoSGBM_create(minDisparity=self.minDisparity,
                                              numDisparities=self.numDisparities,
                                              blockSize=self.blockSize,
                                              P1=self.P1,
                                              P2=self.P2,
                                              disp12MaxDiff=self.disp12MaxDiff,
                                              uniquenessRatio=self.uniquenessRatio,
                                              speckleWindowSize=self.speckleWindowSize,
                                              speckleRange=self.speckleRange,
                                              preFilterCap=self.preFilterCap,
                                              #mode=cv2.StereoSGBM_MODE_HH,
                                              )

        if self.use_wls_filter:
            # Create right matcher
            self.matcherR = cv2.ximgproc.createRightMatcher(self.matcherL)
            # Filter parameters
            visual_multiplier = 1.0
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.matcherL)
            self.wls_filter.setLambda(self.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma)


    """
    Calculate the disparity image.
    This method should be as light as possible as it's supposed to work standalone without the tuner.
    """
    def calculate_disparity(self, imgL, imgR):
        # Pre-process input images
        undistorted_rectifiedL = cv2.remap(imgL, self.mapL1, self.mapL2, interpolation=cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(imgR, self.mapR1, self.mapR2, interpolation=cv2.INTER_LINEAR)
        grayL = cv2.cvtColor(undistorted_rectifiedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(undistorted_rectifiedR, cv2.COLOR_BGR2GRAY)

        # Calculate disparity (return a fixed-point disparity map, where disparity values are multiplied by 16)
        disparityL = self.matcherL.compute(grayL, grayR)

        if self.use_wls_filter:
            # Filtered
            disparityR = self.matcherR.compute(grayR, grayL)

            # Experiment: filter the smaller speckles before using the WLS filter to avoid creating artifacts
            # cv2.filterSpeckles(disparityL, 0, self.speckle_maxSpeckleSize, self.speckle_maxDiff)
            # cv2.filterSpeckles(disparityR, 0, self.speckle_maxSpeckleSize, self.speckle_maxDiff)

            disparity_filtered = self.wls_filter.filter(np.int16(disparityL / 16.), undistorted_rectifiedL, None, np.int16(disparityR / 16.))
            disparity_filtered = cv2.normalize(src=disparity_filtered, dst=disparity_filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            disparity_filtered = np.uint8(disparity_filtered)
            # EXPERIMENTAL: try final filtering to make the image less noisy
            # disparity_filtered, _ = cv2.filterSpeckles(disparity_filtered, 0, self.speckle_maxSpeckleSize, self.speckle_maxDiff)

            self.disparity_scaled = disparity_filtered
        else:
            # Unfiltered
            #cv2.filterSpeckles(disparityL, 0, self.speckle_maxSpeckleSize, self.speckle_maxDiff)
            self.disparity_scaled = (disparityL / 16.).astype(np.uint8) + abs(disparityL.min())

        #B = 200  # distance between images, in mm
        #f = 10  # focal length, in px?
        #z = B * f / self.disparity

        # Crop
        self.disparity_scaled = self.disparity_scaled[self.crop_top:self.crop_bottom,self.crop_left:self.crop_right]
        self.disparity_size = self.disparity_scaled.shape

        return self.disparity_scaled


    def calculate_depth_mm(self, disparity):
        localQ = self.Q.copy()
        #print(localQ)
        #print("Estimated distance between the two cameras: {:4.1f}mm".format(1/localQ[3,2]))

        # Rotate the image upside down for a clearer view in MeshLab
        localQ[1,:] = -1 * localQ[1,:]
        localQ[2,:] = -1 * localQ[2,:]

        # Calculate reprojected points
        points = cv2.reprojectImageTo3D(disparity, localQ)

        # Estimate distance
        # x_points_mm = np.polyval(self.distance_calibration_poly, -points[:,:,0])
        # y_points_mm = np.polyval(self.distance_calibration_poly, -points[:,:,1])
        if self.distance_calibration_invert:
            z_points_mm = np.polyval(self.distance_calibration_poly, -1/points[:,:,2])
        else:
            z_points_mm = np.polyval(self.distance_calibration_poly, -points[:,:,2])

        # Prepare infinity mask to avoid MeshLab issues
        # infinity_mask = disparity > disparity.min() # TODO check is this is always 0 or can change
        infinity_mask = disparity > 0
        # print("Minimum value: {}".format(disparity.min()))

        # Make global variable to be used by the tuner
        self.points = points
        #self.points = np.dstack((x_points_mm, y_points_mm, z_points_mm))
        self.infinity_mask = infinity_mask

        return points, infinity_mask, z_points_mm


    """
    Internal method called every time a previes needs to be generated by the tuner
    """
    def _refresh_preview(self, imgL, imgR):

        self._create_matchers()
        _ = self.calculate_disparity(imgL, imgR)

        # Display disparity
        preview = self.disparity_scaled.copy()
        interline_px = 35
        text = 'minDisparity: {}'.format(self.minDisparity)
        cv2.putText(preview, text, (10,25+0*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'numDisparities: {}'.format(self.numDisparities)
        cv2.putText(preview, text, (10,25+1*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'blockSize: {}'.format(self.blockSize)
        cv2.putText(preview, text, (10,25+2*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'P1: {}'.format(self.P1)
        cv2.putText(preview, text, (10,25+3*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'P2: {}'.format(self.P2)
        cv2.putText(preview, text, (10,25+4*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'disp12MaxDiff: {}'.format(self.disp12MaxDiff)
        cv2.putText(preview, text, (10,25+5*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'uniquenessRatio: {}'.format(self.uniquenessRatio)
        cv2.putText(preview, text, (10,25+6*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'speckleWindowSize: {}'.format(self.speckleWindowSize)
        cv2.putText(preview, text, (10,25+7*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'speckleRange: {}'.format(self.speckleRange)
        cv2.putText(preview, text, (10,25+8*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'preFilterCap: {}'.format(self.preFilterCap)
        cv2.putText(preview, text, (10,25+9*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'wls_lambda: {}'.format(self.wls_lambda)
        cv2.putText(preview, text, (10,25+10*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        text = 'wls_sigma: {}'.format(self.wls_sigma)
        cv2.putText(preview, text, (10,25+11*interline_px), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 8)
        cv2.imshow(self.windowNameD, preview)

        # Display depth
        fig = plt.figure(1)
        plt.ion()
        plt.show()
        _, _, depth_preview = self.calculate_depth_mm(self.disparity_scaled)
        plt.imshow(depth_preview, cmap='hot', interpolation='nearest')
        plt.pause(0.001)


    """
    Tuner used to tune stereo matching parameters
    """
    def tuner(self, imgL, imgR, imgB, meas_distance=None):

        # Pre-process the image that will be used by "_save_cloud_function"
        self.imgL = imgL
        self.imgR = imgR
        self.imgB = imgB
        
        # Create window
        if meas_distance is None:    
            self.windowNameD = 'SGBM Stereo Disparity'
        else:
            self.windowNameD = 'SGBM Stereo Disparity (distance: {})'.format(meas_distance)
        cv2.namedWindow(self.windowNameD, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowNameD, 1200, 1000)
        cv2.moveWindow(self.windowNameD, 0, 0)
        cv2.createTrackbar("minDisparity", self.windowNameD, self.minDisparity, 100, self._change_minDisparity)
        cv2.createTrackbar("numDisparities (*16)", self.windowNameD, int(self.numDisparities/16), 64, self._change_numDisparities)
        cv2.createTrackbar("blockSize", self.windowNameD, self.blockSize, 21, self._change_blockSize)
        cv2.createTrackbar("P1", self.windowNameD, self.P1, 10000, self._change_P1)
        cv2.createTrackbar("P2", self.windowNameD, self.P2, 10000, self._change_P2)
        cv2.createTrackbar("disp12MaxDiff", self.windowNameD, self.disp12MaxDiff, 100, self._change_disp12MaxDiff)
        cv2.createTrackbar("uniquenessRatio", self.windowNameD, self.uniquenessRatio, 100, self._change_uniquenessRatio)
        cv2.createTrackbar("speckleWindowSize", self.windowNameD, self.speckleWindowSize, 1000, self._change_speckleWindowSize)
        cv2.createTrackbar("speckleRange", self.windowNameD, self.speckleRange, 100, self._change_speckleRange)
        cv2.createTrackbar("preFilterCap", self.windowNameD, self.preFilterCap, 100, self._change_preFilterCap)
        cv2.createTrackbar("speckle_maxSpeckleSize", self.windowNameD, self.speckle_maxSpeckleSize, 10000, self._change_speckle_maxSpeckleSize)
        cv2.createTrackbar("speckle_maxDiff", self.windowNameD, self.speckle_maxDiff, 256, self._change_speckle_maxDiff)
        cv2.createTrackbar("wls_lambda (/1000)", self.windowNameD, int(self.wls_lambda/1000), 1000, self._change_wls_lambda)
        cv2.createTrackbar("wls_sigma (/10)", self.windowNameD, int(self.wls_sigma*10), 40, self._change_wls_sigma)
        cv2.setMouseCallback(self.windowNameD, self._save_cloud_function)  # Right click on the image to save the point cloud

        # Run the first time
        self._refresh_preview(self.imgL, self.imgR)

        while 1:
            k = cv2.waitKey()
            if k == 27:
                break

    """
    Internal methods used to change single matcher parameters
    """
    def _change_minDisparity(self, value):
        self.minDisparity = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_numDisparities(self, value):
        self.numDisparities = value * 16
        self._refresh_preview(self.imgL, self.imgR)

    def _change_blockSize(self, value):
        if value > 2 and value < 22 and value%2 == 1:
            self.blockSize = value
            self._refresh_preview(self.imgL, self.imgR)

    def _change_P1(self, value):
        self.P1 = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_P2(self, value):
        self.P2 = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_disp12MaxDiff(self, value):
        self.disp12MaxDiff = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_uniquenessRatio(self, value):
        self.uniquenessRatio = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_speckleWindowSize(self, value):
        self.speckleWindowSize = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_speckleRange(self, value):
        self.speckleRange = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_preFilterCap(self, value):
        self.preFilterCap = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_speckle_maxSpeckleSize(self, value):
        self.speckle_maxSpeckleSize = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_speckle_maxDiff(self, value):
        self.speckle_maxDiff = value
        self._refresh_preview(self.imgL, self.imgR)

    def _change_wls_lambda(self, value):
        self.wls_lambda = value * 1000
        self._refresh_preview(self.imgL, self.imgR)

    def _change_wls_sigma(self, value):
        self.wls_sigma = value / 10
        self._refresh_preview(self.imgL, self.imgR)

    def _save_cloud_function(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.generate3Dimage(self.points, self.infinity_mask, self.imgB)
        return


    """
    Method to save the point cloud in a format that an be opened by MeshLab
    """
    def generate3Dimage(self, points, infinity_mask, imgB, mask_type=0, mesh_filename=None):

        def color2mask(image, color):
            return np.array(cv2.inRange(image, color, color), dtype=bool)

        def write_ply(fn, verts, colors):
            ply_header = '''ply
                            format ascii 1.0
                            element vertex %(vert_num)d
                            property float x
                            property float y
                            property float z
                            property uchar red
                            property uchar green
                            property uchar blue
                            end_header
                            '''
            verts = verts.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            verts = np.hstack([verts, colors])
            with open(fn, 'wb') as f:
                f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
                np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

        # Translate and rectify the background image to fit the depth map
        Tmatrix = np.float32([[1,0,0],[0,1,0]])
        rows, cols, _ = imgB.shape
        imgB = cv2.warpAffine(imgB, Tmatrix, (cols,rows))
        undistorted_rectified_background = cv2.remap(imgB, self.mapR1, self.mapR2, interpolation=cv2.INTER_LINEAR)
        # Output 3-channel floating-point image of the same size as disparity . Each element of _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map.
        #colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)   # I don't think it makes sense, as the disparity is calculated on the remapped image not on imgL
        colors = cv2.cvtColor(undistorted_rectified_background,
                              cv2.COLOR_BGR2RGB)
        colors = colors[self.crop_top:self.crop_bottom,self.crop_left:self.crop_right]

        # Mask based on specific labels (color values)
        if mask_type == 1:
            object_mask = np.full_like(colors[:,:,0], False, dtype=bool)
            object_mask |= color2mask(colors, np.array([0, 192, 0])) # ground animal
            object_mask |= color2mask(colors, np.array([220, 20, 60])) # person
            object_mask |= color2mask(colors, np.array([255, 0, 0])) # bicyclist
            object_mask |= color2mask(colors, np.array([119, 11, 32])) # bicycle
            object_mask |= color2mask(colors, np.array([0, 0, 142])) # boat
            object_mask |= color2mask(colors, np.array([0, 60, 100])) # bus
            object_mask |= color2mask(colors, np.array([0, 0, 142])) # car
            object_mask |= color2mask(colors, np.array([0, 0, 90])) # caravan
            object_mask |= color2mask(colors, np.array([0, 0, 230])) # motorcycle
            object_mask |= color2mask(colors, np.array([0, 80, 100])) # onrails
            object_mask |= color2mask(colors, np.array([128, 64, 64])) # othervehicle
            object_mask |= color2mask(colors, np.array([0, 0, 110])) # trailer
            object_mask |= color2mask(colors, np.array([0, 0, 70])) # truck
            object_mask |= color2mask(colors, np.array([0, 0, 192])) # wheeledslow
        elif mask_type == -1:
            object_mask = np.full_like(colors[:,:,0], True, dtype=bool)
            object_mask &= np.logical_not(color2mask(colors, np.array([70, 130, 180]))) # sky
            object_mask &= np.logical_not(color2mask(colors, np.array([196, 196, 196]))) # curb
            object_mask &= np.logical_not(color2mask(colors, np.array([170, 170, 170]))) # curb cut
            object_mask &= np.logical_not(color2mask(colors, np.array([96, 96, 96]))) # pedestrianarea
            object_mask &= np.logical_not(color2mask(colors, np.array([128, 64, 128]))) # road
            object_mask &= np.logical_not(color2mask(colors, np.array([110, 110, 110]))) # servicelane
            object_mask &= np.logical_not(color2mask(colors, np.array([244, 35, 232]))) # sidewalk
            object_mask &= np.logical_not(color2mask(colors, np.array([200, 128, 128]))) # linescrosswalk
            object_mask &= np.logical_not(color2mask(colors, np.array([255, 255, 255]))) # linesgeneral
            object_mask &= np.logical_not(color2mask(colors, np.array([152, 251, 152]))) # Terrain
            object_mask &= np.logical_not(color2mask(colors, np.array([0, 170, 30]))) # Water
            object_mask &= np.logical_not(color2mask(colors, np.array([32, 32, 32]))) # Car Mount
            object_mask &= np.logical_not(color2mask(colors, np.array([0, 0, 0]))) # unlabeled
        else:
            object_mask = np.full_like(colors[:,:,0], True, dtype=bool)
        mask = np.logical_and(infinity_mask, object_mask)

        """
        # Experiment with labelling
        person_points = points[np.logical_and(infinity_mask, mask_person)]
        person_labels = scipymeas.label(person_points)
        print(person_labels)
        """

        out_points = self.points[mask]
        out_colors = colors[mask]

        # Save mesh
        if mesh_filename is None:
            mesh_filename = 'disparity'
            while os.path.isfile(mesh_filename + '.ply'):
                mesh_filename = mesh_filename + '_'
        print("Saving point cloud as {}...".format(mesh_filename + '.ply'), end="")
        write_ply(mesh_filename + '.ply', out_points, out_colors)
        print(" done!")
        return


if __name__ == '__main__':
    print("OpenCV version: {}".format(cv2.__version__))

    TUNE = 0
    MASK_TYPE = 1
    SEGMENTED_BACKGROUND = 1
    SHOW_DEPTH_MAP = 1

    # for n in range(0,6):
    # for n in [5]:
    # for n in range(0,22):
    for n in [12]:
    
        # 120deg cameras, 390mm
        if 0:
            calibration_folder = '../videos/20171220_stereo_calibration_120deg_390mm/calibration_frames_small/'
            test_folder = '../videos/20171220_stereo_calibration_120deg_390mm/test_frames/'
            test_segmented_folder = '../videos/20171220_stereo_calibration_120deg_390mm/test_frames_segmented/'
            fileL = test_folder + 'test_left_{:03d}_cropped.png'.format(n)
            fileR = test_folder + 'test_right_{:03d}_cropped.png'.format(n)
            if SEGMENTED_BACKGROUND:
                fileB = test_segmented_folder + 'test_left_{:03d}_cropped.png'.format(n)
            else:
                fileB = fileL
            disparity_crop = [200, 1410, 100, 660]
            distance_calibration_poly = np.asarray([2.57345412e-04, -6.24761506e-01, 3.30567462e+03])
            distance_calibration_invert = 0

        # 60deg camera, 120mm
        if 1:
            calibration_folder = '../videos/20180111_stereo_calibration_60deg_120mm/calibration_frames/'
            test_folder = '../videos/20180111_stereo_calibration_60deg_120mm/test_frames/'
            test_segmented_folder = '../videos/20180111_stereo_calibration_60deg_120mm/test_frames_segmented/'
            fileL = test_folder + 'test_left_{:03d}.png'.format(n)
            fileR = test_folder + 'test_right_{:03d}.png'.format(n)
            if SEGMENTED_BACKGROUND:
                fileB = test_segmented_folder + 'test_left_{:03d}.png'.format(n)  # to use the segmented image
            else:
                fileB = fileL
            disparity_crop = [130, 1890, 67, 1170]
            #distance_calibration_poly = np.asarray([9.18628104e-04, -1.74423770e+00, 3.32211320e+03])
            distance_calibration_poly = np.asarray([5.89577503e-13, 1.41653543e-07, -5.70583838e-04, 2.93477758e+00, -9.62542808e+02])
            distance_calibration_invert = 0

        imgL = cv2.imread(fileL)
        imgR = cv2.imread(fileR)
        imgB = cv2.imread(fileB)
        imgB = cv2.resize(imgB, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_NEAREST)
        mesh_filename = fileL[:-4]
        depthmap_filename = fileL[:-4] + '_depthmap.png'

        mycal = pickle.load(open(calibration_folder + "calibration.p", "rb"))
        block_matcher = BM(calibration=mycal,
                           disparity_crop=disparity_crop,
                           distance_calibration_poly=distance_calibration_poly,
                           distance_calibration_invert=distance_calibration_invert,
                           )
        if TUNE:
            block_matcher.tuner(imgL, imgR, imgB)

        if not TUNE:
            # Calculate
            disparity = block_matcher.calculate_disparity(imgL, imgR)
            points, infinity_mask, z_points_mm = block_matcher.calculate_depth_mm(disparity)
            block_matcher.generate3Dimage(points, infinity_mask, imgB, MASK_TYPE, mesh_filename)

            # Depth map
            fig = plt.figure()
            plt.imshow(z_points_mm/1000, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('Depth map (m)')
            plt.savefig(depthmap_filename, bbox_inches='tight')
            if SHOW_DEPTH_MAP:
                plt.show()
