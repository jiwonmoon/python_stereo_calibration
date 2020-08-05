import numpy as np
import cv2
from matplotlib import pyplot as plt

k_data = np.load("C://Users//mjw31//Desktop//data_DP//PNC_data//calibraion//k_data.npy")
dist_data = np.load("C://Users//mjw31//Desktop//data_DP//PNC_data//calibraion//dist_data.npy")
print(k_data)
print(dist_data)

front_image_extension_L = "cameraL_"
front_image_extension_R = "cameraR_"
back_image_extension = ".png"
image_seq_num = 0

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
checkSize = (7, 5)
check_real_l = 0.022 # 2.2cm
objp = np.zeros((5 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
objp =  objp * check_real_l
print("objp:\n", objp)

# Arrays to store object points and image points from all the images    .
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

n_images = 18
input_path = "C://Users//mjw31//Desktop//data_DP//stereo_test//output//stereo_calibration_image//"
imgL_set = []
imgR_set = []

for idx in range(n_images):
    inputL_name = input_path + front_image_extension_L + str(idx) + back_image_extension
    inputR_name = input_path + front_image_extension_R + str(idx) + back_image_extension
    imgL = cv2.imread(inputL_name, 1)
    imgR = cv2.imread(inputR_name, 1)

    imgL_set.append(imgL)
    imgR_set.append(imgR)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
imagePointsL = []  # 2d points in image plane.
imagePointsR = []  # 2d points in image plane.

for idx in range(n_images):

    ret_L, cornersL = cv2.findChessboardCorners(imgL_set[idx], checkSize, None)
    ret_R, cornersR = cv2.findChessboardCorners(imgR_set[idx], checkSize, None)

    if (ret_L):
        # cv2.cornerSubPix(imgL_set[idx], cornersL, (11, 11), (-1, -1),criteria)
        cv2.drawChessboardCorners(imgL_set[idx], checkSize, cornersL, ret_L)
    if (ret_R):
        # cv2.cornerSubPix(imgR_set[idx], cornersR, (11, 11), (-1, -1),criteria)
        cv2.drawChessboardCorners(imgR_set[idx], checkSize, cornersR, ret_R)

    if (ret_L != 0 and ret_R != 0):
        imagePointsL.append(cornersL)
        imagePointsR.append(cornersR)
        object_points.append(objp)

    # cv2.imshow('imgL', imgL_set[idx])
    # cv2.imshow('imgR', imgR_set[idx])
    # cv2.waitKey(0)

print("Starting Calibration\n")
height, width, c = imgL_set[0].shape
R = np.zeros((3, 3), dtype=np.float64)
T = np.zeros((3, 1), dtype=np.float64)
E = np.zeros((3, 3), dtype=np.float64)
F = np.zeros((3, 3), dtype=np.float64)

cv2.stereoCalibrate(object_points, imagePointsL, imagePointsR, k_data, dist_data, k_data, dist_data, (width, height), R, T, E, F)

# retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points, imagePointsL, imagePointsR, (width, height))

print("R: \n", R)
print("T: \n", T, "\nNorm t:", cv2.norm(T) * 0.022)
print("E: \n", E)
print("F: \n", F)

output_path = "C://Users//mjw31//Desktop//data_DP//PNC_data//calibraion//"
np.save(output_path + "rvecs", R)
np.save(output_path + "tvecs", T)

print ("Done Calibration\n")
'''
print ("Starting Rectification\n")
R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,3))
P2 = np.zeros(shape=(3,3))

cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))

print ("Done Rectification\n")
print ("Applying Undistort\n")
'''



