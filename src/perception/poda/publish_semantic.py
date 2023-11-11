#!/usr/bin/env python3.6

import rospy
import cv2
import numpy as np
import torch
import torch.nn as nn
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from network import modeling
from torchvision import transforms as T
from PIL import Image as PILImage
from collections import namedtuple

# Define the CityscapesClass namedtuple
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])

# Define the classes using the CityscapesClass
classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

# Create a mapping from train IDs to colors
train_id_to_color = {c.train_id: c.color for c in classes if c.train_id != 255}

class DeepLabV3:
    def __init__(self):
        rospy.init_node('deeplabv3_node', anonymous=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = '/root/EdgeAda-ROS/src/perception/PODA/ckpt/CS_source.pth'
        self.num_classes = 19

        # Load the model
        self.model = modeling.__dict__['deeplabv3plus_resnet_clip'](num_classes=self.num_classes, BB='RN50')
        self.model.backbone.attnpool = nn.Identity()
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # Set up CV bridge
        self.bridge = CvBridge()

        # Set up the publisher for the semantic segmentation results
        self.semantic_pub = rospy.Publisher('/semantic', Image, queue_size=1)
        
        # Set up the publisher for the input RGB images
        self.image_pub = rospy.Publisher('/image', Image, queue_size=1)


        # Set up the transform for input images
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def colorize_mask(self, mask):
        # Create an RGB image from the mask using the mapping
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for train_id, color in train_id_to_color.items():
            color_mask[mask == train_id] = color
        return color_mask
    
    def process_frame(self, frame):
        # Perform inference
        input_image = PILImage.fromarray(frame)
        input_tensor = self.transform(input_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.detach().max(1)[1].cpu().numpy()[0]  # HW


        # Colorize the mask
        colorized_mask = self.colorize_mask(pred)

        return colorized_mask
    

    def run(self):
        # Open the camera device
        cap = cv2.VideoCapture(0)  # Use 0 for the default camera
        if not cap.isOpened():
            rospy.logerr("Could not open camera.")
            return

        try:
            while not rospy.is_shutdown():
                ret, frame = cap.read()
                if not ret:
                    rospy.logerr("Failed to read frame from camera.")
                    continue

                # Publish the raw camera image
                try:
                    image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    self.image_pub.publish(image_msg)
                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

                # Get the semantic segmentation result
                colorized_mask = self.process_frame(frame)

                # Publish the semantic segmentation result
                try:
                    semantic_msg = self.bridge.cv2_to_imgmsg(colorized_mask, "bgr8")
                    self.semantic_pub.publish(semantic_msg)
                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

        finally:
            cap.release()


if __name__ == '__main__':
    try:
        deep_lab_node = DeepLabV3()
        deep_lab_node.run()
    except rospy.ROSInterruptException:
        pass
