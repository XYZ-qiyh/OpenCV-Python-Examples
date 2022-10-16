# Simple OpenCV imread function test
import cv2

image_path = "./lena.jpg"
image = cv2.imread(image_path)

# out_image = image[:, :, ::-1]
# cv2.imwrite("lena_BGR.jpg", out_image)

cv2.imshow("Display", image)
cv2.waitKey(0)