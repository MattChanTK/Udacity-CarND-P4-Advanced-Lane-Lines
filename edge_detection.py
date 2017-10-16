import numpy as np
import cv2
import glob


class area_of_interest:

    def __init__(self, height, width):
        self.left_bottom = (width*0.15, height*0.95)
        self.right_bottom = (width*0.87, height*0.95)
        self.left_top = (width*0.45, height*0.62)
        self.right_top = (width*0.57, height*0.62)

    @property
    def vertices(self):
        return np.array([[self.left_top, self.right_top, self.right_bottom, self.left_bottom]], dtype=np.int32)

# Black out region of the image that is outside a polygon defined by "vertices"
def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, [vertices], 1)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_sobel_filtered_imgs(gray_img, sobel_kernel=7):
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    return sobel_x, sobel_y


# Calculate directional gradient
def abs_thresh(sobel, thresh=(0, 255)):
    # Take the absolute values
    sobel = np.absolute(sobel)

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

    # Return the result
    return binary_output


# Calculate gradient magnitude
def mag_thresh(sobel_x, sobel_y, mag_thresh=(0, 255)):
    # Calculate the gradient magnitude
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1

    # Return the result
    return binary_output


# Calculate gradient direction
def dir_threshold(sobel_x, sobel_y, thresh=(0, np.pi / 2)):
    # Error statement to ignore division and invalid errors
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobel_y / sobel_x))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
    # Return the result
    return dir_binary


def edge_detection_pipline(img):
    # Gaussian Blur
    kernel_size = 5
    processed_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(processed_img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    # Grayscale image
    gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)

    # Define sobel kernel size
    ksize = 7

    # Apply each of the thresholding functions
    sobel_x, sobel_y = get_sobel_filtered_imgs(gray, sobel_kernel=ksize)
    gradient_x = abs_thresh(sobel_x, thresh=(55, 255))
    gradient_y = abs_thresh(sobel_y, thresh=(75, 255))
    mag_binary = mag_thresh(sobel_x, sobel_y, mag_thresh=(50, 255))
    dir_binary = dir_threshold(sobel_x, sobel_y, thresh=(np.pi / 4, np.pi / 3))

    # Combine all the filters into one mask
    combined = np.zeros_like(dir_binary)
    combined[((gradient_x == 1) & (gradient_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Black out pixel outside the given color range
    s_binary = np.zeros_like(combined)
    s_binary[(s > 160) & (s < 255)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1

    # Defining vertices for marked area
    img_shape = img.shape
    vertices = area_of_interest(img_shape[0], img_shape[1]).vertices

    # Masked area
    color_binary = region_of_interest(color_binary, vertices)

    # convert back to grayscale
    color_binary *= 255

    return color_binary


if __name__ == "__main__":
    images = glob.glob("./result_images/undistorted*.jpg")

    for idx, img_filename in enumerate(images):

        undistorted_img = cv2.imread(img_filename)
        processed_img = edge_detection_pipline(undistorted_img)
        #cv2.imshow("image", undistorted_img)
        #cv2.waitKey(0)
        # cv2.imshow("image", processed_img)
        # cv2.waitKey(0)

        write_name = "./result_images/binary" + str(idx) + ".jpg"
        cv2.imwrite(write_name, processed_img)

