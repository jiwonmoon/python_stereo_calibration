import numpy as np
import cv2
from matplotlib import pyplot as plt


R = np.load("C://Users//mjw31//Desktop//data_DP//stereo_test//imgs//rvecs.npy")
T = np.load("C://Users//mjw31//Desktop//data_DP//stereo_test//imgs//tvecs.npy")
k_data = np.load("C://Users//mjw31//Desktop//data_DP//stereo_test//imgs//cam_m.npy")
dist_data = np.load("C://Users//mjw31//Desktop//data_DP//stereo_test//imgs//distortion.npy")

print(R, R.dtype)
print(T, T.dtype, cv2.norm(T))
print(k_data)
print(dist_data)

RTL = np.eye(4, dtype=np.float64)[:3]
RTR = np.hstack([R, T])

print(RTL, " ", RTL.dtype, RTL.shape)
print(RTR, " ", RTR.dtype, RTR.shape)

checkSize = (7, 5)

# termination criteria for corner sub pixel
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

capL = cv2.VideoCapture("C://Users//mjw31//Desktop//data_DP//stereo_test//imgs//output_L.mp4")
capL.set(cv2.CAP_PROP_POS_FRAMES, 220)
capR = cv2.VideoCapture("C://Users//mjw31//Desktop//data_DP//stereo_test//imgs//output_R.mp4")
capR.set(cv2.CAP_PROP_POS_FRAMES, 220)

while (True):
    retc_L, image_L = capL.read()
    retc_R, image_R = capR.read()

    if (retc_L and retc_R):

        ret_L, cornersL = cv2.findChessboardCorners(image_L, checkSize, None)
        ret_R, cornersR = cv2.findChessboardCorners(image_R, checkSize, None)

        if (ret_L):
            # cv2.cornerSubPix(imgL_set[idx], cornersL, (11, 11), (-1, -1),criteria)
            cv2.drawChessboardCorners(image_L, checkSize, cornersL, ret_L)
            undistort_L = cv2.undistortPoints(cornersL, k_data, dist_data)
        if (ret_R):
            # cv2.cornerSubPix(imgR_set[idx], cornersR, (11, 11), (-1, -1),criteria)
            cv2.drawChessboardCorners(image_R, checkSize, cornersR, ret_R)
            undistort_R = cv2.undistortPoints(cornersR, k_data, dist_data)

        if (ret_L and ret_R):
            pts4D = cv2.triangulatePoints(RTL, RTR, undistort_L, undistort_R)

            # convert from homogeneous coordinates to 3D
            pts3D = []
            for pt_idx in range(pts4D.shape[1]):
                pts3D_each = (pts4D[:, pt_idx: pt_idx + 1] / pts4D[3:4, pt_idx: pt_idx + 1])[:3, :]
                pts3D.append(pts3D_each)

            # print(np.array(pts3D).shape)
            # print(np.array(pts3D)[0])
            print(cv2.norm(np.array(pts3D)[0]))

        cv2.imshow('image_L', image_L)
        cv2.imshow('image_R', image_R)
        if cv2.waitKey(35) == ord('q'):
            break

cv2.destroyAllWindows()



