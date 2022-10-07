import cv2
import numpy as np

# source image (book2.jpg)
image_path2 = "./images/book2.jpg"
image2 = cv2.imread(image_path2)
pts_image2 = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])
# print(pts_image2.shape)
# for idx in range(pts_image2.shape[0]):
#     pts = pts_image2[idx]
#     x, y = int(pts[0]), int(pts[1])
#     image2 = cv2.circle(image2, (x, y), radius=5, color=(0, 0, 255), thickness=2)
# image_path2 = "./images/deep-learning-with-pytorch-1.png"
# image2 = cv2.imread(image_path2)
# h, w = image2.shape[0], image2.shape[1]
# print(h, w)
# pts_image2 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])


# destination image (book1.jpg)
image_path1 = "./images/book1.jpg"
image1 = cv2.imread(image_path1)
pts_image1 = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
# for idx in range(pts_image1.shape[0]):
#     pts = pts_image1[idx]
#     x, y = int(pts[0]), int(pts[1])
#     image1 = cv2.circle(image1, (x, y), radius=5, color=(0, 0, 255), thickness=2)

h, status = cv2.findHomography(pts_image2, pts_image1)
warped_image = cv2.warpPerspective(image2, h, (image1.shape[1], image1.shape[0]))

# image1_mask = cv2.fillConvexPoly(image1, pts_image1, color=(0,0,0))


cv2.imshow("book2", image2)
cv2.imshow("book1", image1)
cv2.imshow("warped", warped_image)
# cv2.imshow("masked", image1_mask+warped_image)
cv2.waitKey(0)