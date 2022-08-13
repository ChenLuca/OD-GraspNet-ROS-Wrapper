#!/usr/bin/env python3
import rospy
import rospkg
import cv2
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import os
from skimage.feature import peak_local_max

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import detect_grasps

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from od_graspnet.msg import dl_grasp_result
from od_graspnet.msg import AngleAxis_rotation_msg


logging.basicConfig(level=logging.INFO)

rgb_bridge = CvBridge()
depth_bridge = CvBridge()

global_q_img = np.zeros((0,0,1), np.uint8)
global_ang_img = np.zeros((0,0,1), np.uint8)

rgb_image = np.zeros((0,0,3), np.uint8)
depth_image = np.zeros((0,0,1), np.uint8)

save_img_counter = 0
no_grasps = 1

model_path = os.path.join(rospkg.RosPack().get_path('od_graspnet'), 'scripts/trained-models/220524_0030_odc_shuffle_v2_4_ds-shuffle_epoch300_NoDataAugment_jacquard', 'epoch_85_iou_0.94')

print("model_path ", model_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default=model_path,
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=5,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

def rgb_callback(image):
    global rgb_image
    try:
        rgb_image = rgb_bridge.imgmsg_to_cv2(image, "rgb8")
    except CvBridgeError as e:
        print(e)

def depth_callback(image):
    global depth_image
    try:
        depth_image = depth_bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        print(e)

if __name__ == '__main__':

    args = parse_args()

    rospy.init_node('grcnn_inference', anonymous=True)

    pub_AngleAxisRotation = rospy.Publisher('/2D_Predict/AngleAxis_rotation', AngleAxis_rotation_msg, queue_size=10)

    pub_osa_result = rospy.Publisher('/dl_grasp/result', dl_grasp_result, queue_size=10)

    rospy.Subscriber("/rgb/image_raw", Image, rgb_callback)

    rospy.Subscriber("/depth_to_rgb/image_raw", Image, depth_callback)

    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    while not rospy.is_shutdown():
        if (depth_image.shape[0]!=0):
            try:
                fig = plt.figure(figsize=(10, 10))
                while 1:
                    rgb = rgb_image
                    depth = np.expand_dims(depth_image, axis=2)
                    x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

                    with torch.no_grad():
                        xc = x.to(device)
                        pred = net.predict(xc)

                        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                        
                        global_q_img = q_img
                        global_ang_img = ang_img

                        gs = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=no_grasps)

                        if gs is not None:
                            for g in gs:
                                osa_result_msg = dl_grasp_result()
                                osa_result_msg.y = g.center[0] + 110
                                osa_result_msg.x = g.center[1] + 190
                                osa_result_msg.angle = g.angle
                                osa_result_msg.length = g.length
                                osa_result_msg.width = g.width
                                pub_osa_result.publish(osa_result_msg)

                                rotation = AngleAxis_rotation_msg()
                                rotation.x = 0
                                rotation.y = 0
                                rotation.z = -1* g.angle 
                                pub_AngleAxisRotation.publish(rotation)
                                
                                print("center(y, x):{}, angle:{}, length:{}, width:{} ".format(g.center, g.angle, g.length, g.width))

                        plot_results(fig=fig,
                                    rgb_img=cam_data.get_rgb(rgb, False),
                                    depth_img=np.squeeze(cam_data.get_depth(depth)),
                                    grasp_q_img=q_img,
                                    grasp_angle_img=ang_img,
                                    no_grasps=args.n_grasps,
                                    grasp_width_img=width_img)
            finally:
                print('bye grcnn_inference!')
