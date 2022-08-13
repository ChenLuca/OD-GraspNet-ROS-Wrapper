#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
from PIL import Image
from numpy import asarray
import numpy as np
import rospy
import cv2 as cv
from skimage.feature import peak_local_max

from pcl_utils.srv import sample_q_image, sample_q_imageResponse

def q_img_sample(q_img, ang_img):

    def normalize_data(ori_data):

        sum_data = sum(ori_data)
        norm_data = ori_data/sum_data

        return norm_data

    # Step1: sample n local_max point
    local_max = peak_local_max(q_img, min_distance=8, threshold_abs=0.2, num_peaks=10)

    # Step2: get gray_scale value and normalized to 1
    val_ori = []
    for i in range(len(local_max)):
        val = q_img[local_max[i][0],local_max[i][1]]
        print(local_max[i][0],local_max[i][1], val)
        val_ori.append(val)

    val_normalized = normalize_data(val_ori)
    # print('val_normalized', val_normalized)

    # Step3: sample again from n local_max point,
    #        the quality(0-255) of image is the possibility to be sampled
    point_idx = [i for i in range(len(local_max))]

    sampled_point_idx = np.random.choice(point_idx, 1, p=val_normalized)

    # output: sampled pixel position
    # print("pixel coord: ", local_max[sampled_point_idx][0], "grasp quality: ", val_ori[sampled_point_idx[0]])

    print("tuple(local_max[sampled_point_idx][0]) ", tuple(local_max[sampled_point_idx][0]))
    grasp_point = local_max[sampled_point_idx][0]
    print("grasp_point ", grasp_point)
    return local_max[sampled_point_idx][0], ang_img[tuple(grasp_point)]

def handle_sample_q_image(req):
    
    print("in service get_q_image, req:", req.call)
    
    # input: gray_scale image 0-255
    q_img = Image.open('/home/ur5/datasets/GraspPointDataset/training/q_img_{}.png'.format(req.call))
    q_img = asarray(q_img)

    # input: gray_scale image 0-255
    ang_img = np.load('/home/ur5/datasets/GraspPointDataset/training/ang_img_{}.npy'.format(req.call))
    ang_img = asarray(ang_img)

    local_max , rotation= q_img_sample(q_img, ang_img)
    print("local_max, rotation ", local_max, rotation)
    res = sample_q_imageResponse()
    res.x = local_max[1] + 190
    res.y = local_max[0] + 110
    res.rotation = -1 * rotation

    return res

if __name__ == '__main__':

    rospy.init_node('sample_q_image_server', anonymous=True)

    s = rospy.Service('/sample_q_image', sample_q_image, handle_sample_q_image)

    rospy.spin()