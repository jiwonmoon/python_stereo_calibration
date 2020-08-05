import numpy as np
import cv2

image_out_path = "C://Users//mjw31//Desktop//data_DP//stereo_test//output//stereo_calibration_image//"
front_image_extension_L = "cameraL_"
front_image_extension_R = "cameraR_"
back_image_extension = ".png"
image_seq_num = 0

capL = cv2.VideoCapture("C://Users//mjw31//Desktop//data_DP//stereo_test//PNC//Camera6//record_2020-08-02-14.25.45.mp4")
capL.set(cv2.CAP_PROP_POS_FRAMES, 280+60)
capR = cv2.VideoCapture("C://Users//mjw31//Desktop//data_DP//stereo_test//PNC//Camera7//record_2020-08-02-14.25.47.mp4")
capR.set(cv2.CAP_PROP_POS_FRAMES, 280+5)

while(True):
    retc_L, image_L = capL.read()
    retc_R, image_R = capR.read()

    if(retc_L and retc_R):

        L_resize = cv2.resize(image_L, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        R_resize = cv2.resize(image_R, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('image_L', L_resize)
        cv2.imshow('image_R', R_resize)

        image_out_name_L = image_out_path + front_image_extension_L + str(image_seq_num) + back_image_extension
        image_out_name_R = image_out_path + front_image_extension_R + str(image_seq_num) + back_image_extension

        key_value = cv2.waitKey(30)
        if key_value == ord('c'):
            print(image_out_name_L)
            print(image_out_name_R)
            cv2.imwrite(image_out_name_L, image_L)
            cv2.imwrite(image_out_name_R, image_R)
            image_seq_num = image_seq_num + 1

        if key_value == ord('q'):
            break

cv2.destroyAllWindows()