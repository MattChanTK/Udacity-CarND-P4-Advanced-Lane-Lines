import cv2
import glob
import pickle

# Load the camera calibration file
dist_pickle = pickle.load(open("calibration.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Load the test images
images = glob.glob("./test_images/test*.jpg")

for idx, img_filename in enumerate(images):
    raw_img = cv2.imread(img_filename)
    img = cv2.undistort(raw_img, mtx, dist, None, mtx)

    write_name = "./result_images/undistorted" + str(idx) + ".jpg"
    cv2.imwrite(write_name, img)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)