import numpy as np
import cv2

import sys
# sys.path.append("D:\\研一下\\！NIPS\\My-MVSNet\\2020-05-20")
from datasets.preprocess import read_pfm, read_cam


def depth2normal(depth_path, K):
    # step1. read_pfm()
    # reproject to 3D space
    
    # depth_image [h, w]
    # point_image [h, w, 3]
    depth_image, _ = read_pfm(depth_path) # [H, W]
    print("data: {}".format(depth_image[8, 10]))
    print("===============")

    
    height, width = depth_image.shape
    #print("H: {} W: {}".format(height, width))
    
    x = np.linspace(0, (width-1), width, dtype=np.float32)[np.newaxis, :].repeat(height, axis=0) # [H, W]
    y = np.linspace(0, (height-1), height, dtype=np.float32)[:, np.newaxis].repeat(width, axis=1) # [H, W]
    x, y = x.reshape(height*width), y.reshape(height*width) # [H*W]
    z = np.ones_like(x)
    xyz = np.stack((x, y, z), axis=0) #[3, H*W]

    # xyz means uv1 (齐次坐标)
    
    #print("data: {}".format(xyz[:, 0]))

    # https://stackoverflow.com/questions/21638895/inverse-of-a-matrix-using-numpy    
    K_inv = np.linalg.inv(K) # TEST pass OK
    print("============================")
    print("xyz: {}".format(K_inv.dtype))
    
    xyz_tmp = np.matmul(K_inv, xyz)
    xyz_tmp = xyz_tmp.reshape(3, height, width)
    print("xyz_tmp: {}".format(xyz_tmp.shape))
    print("data: {}".format(xyz_tmp[:, 8, 10]))
    
    point_image = depth_image * xyz_tmp
    point_image = point_image.transpose([1, 2, 0])
    print("point_image: {}".format(point_image.shape))
    print("data: {}".format(point_image[8, 10, :]))
    print(point_image.dtype)

    
    
    # step2. 分块
    n = 5 # 7 # 9
    shape = ((height - n + 1), (width - n + 1), n, n, 3)
    print("=======================")
    print("shape: {}".format(shape))
    
    s = point_image.strides
    print("strides: {}".format(s))

    
    point_blocks = np.lib.stride_tricks.as_strided(point_image, shape=shape,
                                                    strides=(s[:2] + s)) # [new_h, new_w, n, n, 3]
    
    print("shape: {}".format(point_blocks.shape)) 
    print("data: {}".format(point_blocks[8, 10, :, :, :]))
    #print("data2： {}".format(point_image[8:8+5, 10:10+5, :]))
    #print(point_blocks[8, 10, :, :, :] - point_image[8:8+5, 10:10+5, :])
    
    
    new_h, new_w = point_blocks.shape[0], point_blocks.shape[1]
    print("new_h: {}, new_w: {}".format(new_h, new_w))

    point_blocks = point_blocks.reshape(new_h*new_w, n*n, 3) # [new_h*new_w, n^2, 3]
    print("new point_blocks: {}".format(point_blocks.shape))

    #point_blocks = point_blocks.reshape(new_h, new_w, n, n, 3)
    #print("data: {}".format(point_blocks[8, 10, :, :, :]))
    #print(point_blocks.reshape)
    
    print("data: {}".format(point_blocks[8*new_w+10, :, :]))
    
    point_t = point_blocks.transpose([0, 2, 1])
    
    # Perform KNN
    dist_xx = np.sum(point_blocks * point_blocks, axis=2, keepdims=True) # [B, n^2]
    dist_inner = np.matmul(point_blocks, point_t) # [B, n^2, n^2]
    dist = dist_xx - 2 * dist_inner + dist_xx.transpose([0, 2, 1]) # [B, n^2, n^2]
    print("===================================")
    print("dist_xx: {}".format(dist_xx.shape))
    print("dist_inner: {}".format(dist_inner.shape))
    print("dist: {}".format(dist.shape))
    
    


    
    # step3. calculate normal 
    # Least Square Method
    point_t = point_blocks.transpose([0, 2, 1])
    print("data: {}".format(point_t[8*new_w+10, :, :]))
    
    tmp = np.matmul(point_t, point_blocks) # [B, 3, 3]
    print("tmp: {}".format(tmp.shape))
    
    # https://stackoverflow.com/questions/21828202/fast-inverse-and-transpose-matrix-in-python
    tmp_inv = np.linalg.inv(tmp) # [B, 3, 3]
    #print("tmp_inv: {}".format(tmp_inv.shape))
    
    tmp = np.matmul(tmp_inv, point_t) # [B, 3, K]
    #print("tmp: {}".format(tmp.shape))
    
    vector_one = np.ones((1, n*n, 1), dtype=np.float32).repeat(new_h*new_w, axis=0) # [B, K, 1]
    print("vector_one: {}".format(vector_one.shape))
    
    normal_vector = np.matmul(tmp, vector_one) # [B, 3, 1]
    
    normal_vector = np.squeeze(normal_vector, axis=2) # [B, 3]
    print("normal_vector: {}".format(normal_vector.shape))    
        
    # normalize
    normal_vector = normal_vector / np.linalg.norm(normal_vector, axis=1)[:, np.newaxis]
    #normal_length = np.linalg.norm()
    
    normal_image = normal_vector.reshape(new_h, new_w, 3)
    print("normal_image: {}".format(normal_image.shape))

    normal_image_2333 = np.zeros((height, width, 3), dtype=np.float32)
    normal_image_2333[2:2+new_h, 2:2+new_w, :] = normal_image
    print("normal_image 2333: {}".format(normal_image_2333.shape))
    
    cv2.imshow("normal_map", normal_image_2333)
    cv2.waitKey(0)
    
    # step last. multiply mask(valid mask)

    #print(aaa)


    #write_gipuma_dmb()


if __name__ == "__main__":
    in_depth_path = "./data_for_test/00000000_init.pfm"
    in_cam_path = "./data_for_test/00000000_cam.txt"
    cam = read_cam(in_cam_path)
    cam = np.float32(cam)
    K = cam[1, :3, :3]
    K[:2, :3] /= 2  # scale
    K_inv = np.linalg.inv(K)
    #print("K: {}".format(K))
    #print("K_inv: {}".format(K_inv))
    depth2normal(in_depth_path, K)