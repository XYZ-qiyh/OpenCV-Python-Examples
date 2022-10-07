import cv2
import numpy as np

# source image (deep-learning-with-pytorch-1.png)
image_path2 = "./images/deep-learning-with-pytorch-1.png"
image2 = cv2.imread(image_path2)
h, w = image2.shape[0], image2.shape[1]
# print(h, w)
pts_image2 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])


# destination image (book1.jpg)
image_path1 = "./images/book1.jpg"
image1 = cv2.imread(image_path1)
pts_image1 = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

h, status = cv2.findHomography(pts_image2, pts_image1)
warped_image = cv2.warpPerspective(image2, h, (image1.shape[1], image1.shape[0]))

image1_mask = cv2.fillConvexPoly(image1, pts_image1, color=(0,0,0))

cv2.imshow("book2", image2)
cv2.imshow("book1", image1)
cv2.imshow("warped", warped_image)
cv2.imshow("final", image1_mask+warped_image)
cv2.waitKey(0)

cv2.imwrite("./blended.jpg", (image1_mask+warped_image))