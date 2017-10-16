import numpy as np
import cv2
import glob
import pickle

num_row = 6
num_col = 9

# prepare object points
obj_points = np.zeros((num_row * num_col, 3), np.float32)
obj_points[:, :2] = np.mgrid[0:num_col, 0:num_row].T.reshape(-1, 2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
img_points = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, img_filename in enumerate(images):
    img = cv2.imread(img_filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_img, (num_col, num_row), None)

    # If found, add object points, image points
    if ret:
        print("working on ", img_filename)

        # corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (num_col, num_row), corners, ret)
        write_name = "corners_found" + str(idx) + ".jpg"
        cv2.imwrite("./camera_cal/" + write_name, img)
    else:
        print("unable to find corners of ", img_filename)

# Load Image for reference
img = cv2.imread('./camera_cal/calibration01.jpg')
img_size = (img.shape[1], img.shape[0])
cv2.imshow("image", img)
# cv2.waitKey(0)

# Do camera calibration given object points and image points
ret, mtx, dist, _, _ = cv2.calibrateCamera([obj_points]*len(img_points), img_points, img_size, None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("calibration.p", "wb" ) )
