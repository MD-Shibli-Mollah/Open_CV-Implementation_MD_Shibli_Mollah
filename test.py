import cv2
import numpy as np

import edge_canny_get_orientation_sector as ors
import edge_canny_is_local_max as locmax
import edge_canny_trace_and_threshold as tt

# Image Read
image = cv2.imread("Boss_Family.jpg")

# Implementing Canny Edge Detection with Open cv package
canny = cv2.Canny(image, 80, 100)
cv2.imwrite("images/my_canny_Boss_family.jpg", canny)

canny = cv2.Canny(image, 100, 200)  # Canny edge detection in CV2
cv2.imwrite("images/my_canny_Boss_family_2.jpg", canny)

# pixel information
print("Dimension of Canny: ", canny.shape)
print("Dimension of Image: ", image.shape)
print("Dtype of Canny: ", canny.dtype)
print("Dtype of Image: ", image.dtype)
print("Pixel [0, 0] of Canny: ", canny[0, 0])
print("Pixel [0, 0] of Image: ", image[0, 0])

# modification of image
'''for r in range(0, image.shape[0]):
    for c in range(0, image.shape[1]):
        image[r, c] = [100, 0, 100]
cv2.imwrite("pix_change1.jpg", image)
print(image)
print("new Image", image.shape)
print("Dimension of Image[r]:  ", image.shape[0])
print("Dimension of Image[c]: ", image.shape[1])'''

# Blurry effect using Gausian Filter
gaussianBlur = cv2.GaussianBlur(image, (5, 5), 1)
cv2.imwrite("images/Boss_Family_gaussian_blur.jpg", gaussianBlur)



#Colvolution
def convolve(image, kernel):
    output = np.zeros(image.shape, image.dtype)
    for r in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            value = 0
            for a in range(0, 3):
                for b in range(0, 3):
                    value = int(int(value) + int(kernel[a][b]) * int(image[r + a - 1, c + b - 1]))
            value = int(value / 8)
            if value < 0:
                value = 0
            if value > 255:
                value = 255
            output[r, c] = value
    return output


#Implementation of Canny Edge detection algorithm my Shibli

# Step 1: Gaussian Filter using raw code
image = cv2.imread("Boss_Family.jpg", cv2.IMREAD_GRAYSCALE)  # Converting to grayscale
cv2.imwrite("images/GrayScale_image_Boss_family.jpg", image)

height = image.shape[0]
width = image.shape[1]

gauss = (1.0 / 57) * np.array(
    [[0, 1, 2, 1, 0],
     [1, 3, 5, 3, 1],
     [2, 5, 9, 5, 2],
     [1, 3, 5, 3, 1],
     [0, 1, 2, 1, 0]])
sum(sum(gauss))

for i in np.arange(2, height - 2):
    for j in np.arange(2, width - 2):
        sum = 0
        for k in np.arange(-2, 3):
            for l in np.arange(-2, 3):
                a = image.item(i + k, j + l)
                p = gauss[2 + k, 2 + l]
                sum = sum + (p * a)
        b = sum
        image.itemset((i, j), b)
cv2.imwrite("images/filter_gaussian_Boss_family.jpg", image)

print("Gaussian Filter done")
# step 2: compute gradient magnitude

Hx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Hy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])

image_x = convolve(image, Hx) / 8.0
image_y = convolve(image, Hy) / 8.0

magnitude = np.sqrt(np.power(image_x, 2)) + np.power(image_y, 2)
magnitude = (magnitude / np.max(magnitude)) * 255
cv2.imwrite("images/Sobel_Boss_Family.jpg", magnitude)
print("sobel is done")

# step 3: non-maximum suppression
t_low = 4
E_nms = np.zeros((height, width))
for i in np.arange(1, height - 1):
    for j in np.arange(1, width - 1):
        dx = image_x[i, j]
        dy = image_y[i, j]
        s_theta = ors.get_orientation_sector(dx, dy)

        if locmax.is_local_max(magnitude, i, j, s_theta, t_low):
            E_nms[i, j] = magnitude[i, j]

# step 4: edge tracing and hysteresis thresholding
t_high = 7
E_bin = np.zeros((height, width))
for i in np.arange(1, height - 1):
    for j in np.arange(1, width - 1):
        if E_nms[i, j] >= t_high and E_bin[i, j] == 0:
            tt.trace_and_threshold(E_nms, E_bin, i, j, t_low)

cv2.imwrite('images/Boss_Family_own_canny.jpg', E_bin)
