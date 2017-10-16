import numpy as np
import cv2
import glob
from edge_detection import area_of_interest

# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def perspective_transform(img):

    img_shape = img.shape
    src = np.float32(area_of_interest(img_shape[0], img_shape[1]).vertices)
    width  = img_shape[1]
    height =  img_shape[0]
    left_bottom = (width * 0, height * 1)
    right_bottom = (width * 1, height * 1)
    left_top = (width * 0, height * 0)
    right_top = (width * 1, height * 0)
    dst = np.float32((left_top, right_top, right_bottom, left_bottom))

    # Given src and dst points, calculate the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src, dst)

    # Warp the image
    warped_img = cv2.warpPerspective(img, transform_matrix, (img_shape[1], img_shape[0]))

    # Return the resulting image and matrix
    return warped_img

if __name__ == "__main__":

    # images = glob.glob("./result_images/undistorted*.jpg")
    images = glob.glob("./result_images/binary*.jpg")

    for idx, img_filename in enumerate(images):

        undistorted_img = cv2.imread(img_filename)
        processed_img = perspective_transform(undistorted_img)
        #cv2.imshow("image", undistorted_img)
        #cv2.waitKey(0)
        #cv2.imshow("image", processed_img)
        #cv2.waitKey(0)

        write_name = "./result_images/perspective" + str(idx) + ".jpg"
        cv2.imwrite(write_name, processed_img)
