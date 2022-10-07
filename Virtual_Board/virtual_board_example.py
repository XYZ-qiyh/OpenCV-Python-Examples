import os
import cv2
import numpy as np

# imread background picture
bg_image_path = "./media/Background.jpg"
bg_image = cv2.imread(bg_image_path)
print(bg_image.shape)

# load input video
in_video_path = "./media/iKun.mp4"
video = cv2.VideoCapture(in_video_path)

# h, w = 720, 1280
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
print(height, width)
pts_src = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
pts_dst = np.array([[1200, 548], [1488, 460], [1498, 758], [1200, 782]])

h, status = cv2.findHomography(pts_src, pts_dst)
print("h, status: {}, {}".format(h.shape, status.shape))

'''
Test code
image = cv2.imread("E:/Projects/opencv-python-test/lena.jpg")
image2 = cv2.resize(image, (width, height))
cv2.imshow("lena", image2)

im_dst = cv2.warpPerspective(image2, h, (bg_image.shape[1], bg_image.shape[0]))
bg_image = cv2.fillConvexPoly(bg_image, pts_dst, color=(0,0,0))
out_blend_image = im_dst + bg_image
cv2.imwrite("out.jpg", out_blend_image)
#cv2.imshow("warp", im_dst+bg_image)
#cv2.waitKey(0)
print(aaa)
'''
# out_video = cv2.VideoWriter(
#     "output.avi", cv2.VideoWriter_fourcc(*'MPEG'), fps, (height, width))

idx = 0
while(True):
    ret, frame = video.read()
    if not ret:
        break

    warped_frame = cv2.warpPerspective(frame, h, (bg_image.shape[1], bg_image.shape[0]))
    bg_image = cv2.fillConvexPoly(bg_image, pts_dst, color=(0,0,0))
    blended_frame = warped_frame + bg_image

    # writing the blended frame in output video
    # out_video.write(blended_frame)
    cv2.imwrite("./outputs/{:0>4}.jpg".format(idx), blended_frame)
    idx = idx + 1
    cv2.imshow('frame', blended_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
# out_video.release()
cv2.destroyAllWindows()

print('-'*20)
print("END")