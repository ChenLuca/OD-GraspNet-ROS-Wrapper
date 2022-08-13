# input: gray_scale image 0-255
# Step1: sample n local_max point
# Step2: get gray_scale value and normalized to 1
# Step3: sample again from n local_max point,
#        the quality(0-255) of image is the possibility to be sampled
# output: pixel position
#!/usr/bin/env python3
import sys
import rospy

from sklearn import preprocessing
from skimage.draw import polygon
from skimage.feature import peak_local_max

from PIL import Image

from numpy import asarray
import numpy as np

def normalize_data(ori_data):

    sum_data = sum(ori_data)
    norm_data = ori_data/sum_data

    return norm_data

# input: gray_scale image 0-255
image = Image.open('q_gray_img_1.png')
data = asarray(image)

# Step1: sample n local_max point
local_max = peak_local_max(data, min_distance=20, threshold_abs=0.2, num_peaks=5)

# Step2: get gray_scale value and normalized to 1
val_ori = []
for i in range(len(local_max)):
    val = data[local_max[i][0],local_max[i][1]]/255
    print(local_max[i][0],local_max[i][1], val)
    val_ori.append(val)

val_normalized = normalize_data(val_ori)
print('val_normalized', val_normalized)

# Step3: sample again from n local_max point,
#        the quality(0-255) of image is the possibility to be sampled
point_idx = [i for i in range(len(local_max))]
sampled_point_idx = np.random.choice(point_idx, 1, p=val_normalized)

# output: sampled pixel position
print("pixel coord: ", local_max[sampled_point_idx][0], "grasp quality: ", val_ori[sampled_point_idx[0]])


if __name__=="__main__":
    ''''
    TESTING!!!
    '''
    print("========TESTING===========")
    testing_times = 100000
    sampled_list = []
    for cnt in range(testing_times):
        sampled_tmp = np.random.choice(point_idx, 1, p=val_normalized)
        sampled_list.append(sampled_tmp)

    idx_counter = [[i for _ in range(2)] for i in range(len(local_max))]
    for idx1 in idx_counter:
        idx1[1] = 0
        # print(idx1)

    for idx2 in sampled_list:
        # print(idx2)
        idx_counter[idx2[0]][1] += 1

    for idx1 in idx_counter:
        idx1[1] /= testing_times
        print(idx1)
    