#!/usr/bin/env python3.6
import cv2
import sys
import rospy
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image as RosImage
from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform
import threading
# main
class Omnidata:
    def __init__(self):
        self.root_dir = '/root/Edge_Ada/src/perception/omnidata/omnidata_tools/torch/tools/pretrained_models/'
        self.trans_topil = transforms.ToPILImage()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
        self.trans_totensor = None
        self.model_depth = None
        self.model_normal = None

        self.initialize_depth_model()
        self.initialize_normal_model()

        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', RosImage, self.image_callback)
        
        self.rgb_pub = rospy.Publisher('/camera/rgb', RosImage, queue_size=1)

        self.depth_pub = rospy.Publisher('/camera/depth', RosImage, queue_size=1)
        self.normal_pub = rospy.Publisher('/camera/normal', RosImage, queue_size=1)

    def initialize_normal_model(self):
        image_size = 384
        pretrained_weights_path = self.root_dir + 'omnidata_dpt_normal_v2.ckpt'
        self.model_normal = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)

        checkpoint = torch.load(pretrained_weights_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        self.model_normal.load_state_dict(state_dict)
        self.model_normal.to(self.device)

        self.trans_totensor_normal = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            get_transform('rgb', image_size=None)
        ])

    def initialize_depth_model(self):
        image_size = 384
        pretrained_weights_path = self.root_dir + 'omnidata_dpt_depth_v2.ckpt'
        self.model_depth = DPTDepthModel(backbone='vitb_rn50_384')

        checkpoint = torch.load(pretrained_weights_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        self.model_depth.load_state_dict(state_dict)
        self.model_depth.to(self.device)
        self.trans_totensor_depth = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

    def process_rgb(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rgb_msg = RosImage()
        rgb_msg.height = rgb_img.shape[0]
        rgb_msg.width = rgb_img.shape[1]
        rgb_msg.encoding = "bgr8"
        rgb_msg.is_bigendian = False
        rgb_msg.step = rgb_img.shape[1] * 3  # Number of bytes per row: width * num_channels
        rgb_msg.data = np.array(rgb_img, dtype=np.uint8).tobytes()

        self.rgb_pub.publish(rgb_msg)

    # depth [0-1]
    # def process_depth(self, input_tensor):
    #     # Perform the forward pass
    #     output = self.model_depth(input_tensor).clamp(min=0, max=1)
    #     output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)

    #     # Convert the tensor to a numpy array
    #     depth_image = output.squeeze().cpu().numpy()

    #     # Standardize and rescale the depth map
    #     depth_image = 1 - depth_image
    #     depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))

    #     # Create a ROS image message
    #     depth_image_msg = RosImage()
    #     depth_image_msg.header.stamp = rospy.Time.now()
    #     depth_image_msg.height = depth_image.shape[0]
    #     depth_image_msg.width = depth_image.shape[1]
    #     depth_image_msg.encoding = "32FC1"
    #     depth_image_msg.is_bigendian = False
    #     depth_image_msg.step = depth_image_msg.width * 4  # Width * sizeof(float)
    #     depth_image_msg.data = depth_image.astype(np.float32).tobytes()

    #     # Publish the depth image message
    #     self.depth_pub.publish(depth_image_msg)

    def process_depth(self, input_tensor):
        # Perform the forward pass
        output = self.model_depth(input_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)

        # Convert the tensor to a numpy array
        depth_image = output.squeeze().cpu().numpy()

        # Standardize and rescale the depth map
        depth_image = 1 - depth_image
        depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))

        # Convert depth to color using a colormap
        depth_color = plt.cm.viridis(depth_image)[:, :, :3]
        depth_color = (depth_color * 255).astype(np.uint8)

        # Convert to PIL.Image
        depth_color_pil = Image.fromarray(depth_color)

        # Create a ROS image message
        depth_image_msg = RosImage()
        depth_image_msg.header.stamp = rospy.Time.now()
        depth_image_msg.height = depth_color.shape[0]
        depth_image_msg.width = depth_color.shape[1]
        depth_image_msg.encoding = "rgb8"
        depth_image_msg.is_bigendian = False
        depth_image_msg.step = depth_color.shape[1] * 3  # Width * sizeof(uint8) * num_channels
        depth_image_msg.data = depth_color_pil.tobytes()

        # Publish the depth image message
        self.depth_pub.publish(depth_image_msg)
    
    
    def process_normal(self, img_tensor):
        output = self.model_normal(img_tensor).clamp(min=0, max=1)
        normal_img = self.trans_topil(output[0]).convert("RGB")
        normal_img = np.array(normal_img)
        normal_msg = RosImage()
        normal_msg.height = normal_img.shape[0]
        normal_msg.width = normal_img.shape[1]
        normal_msg.encoding = "bgr8"
        normal_msg.is_bigendian = False
        normal_msg.step = normal_img.shape[1] * 3  # Number of bytes per row: width * num_channels
        normal_msg.data = np.array(normal_img, dtype=np.uint8).tobytes()

        self.normal_pub.publish(normal_msg)

    # def image_callback(self, frame):
    #     # Convert the captured frame to a PIL.Image object
    #     cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     pil_image = PIL.Image.fromarray(cv_image)

    #     # Resize the image to fit the model
    #     pil_image_resized = pil_image.resize((640, 480), PIL.Image.BICUBIC)

    #     # Convert the image to a tensor
    #     input_tensor_depth = self.trans_totensor_depth(pil_image_resized).unsqueeze(0).to(self.device)
    #     input_tensor_normal = self.trans_totensor_normal(pil_image_resized).unsqueeze(0).to(self.device)
        
    #     # Perform inference using the model
    #     with torch.no_grad():
    #         self.process_depth(input_tensor_depth)
    #         self.process_normal(input_tensor_normal)
    #         self.process_rgb(cv_image)
    
    def image_callback(self, frame):
        # Convert the captured frame to a PIL.Image object
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(cv_image)

        # Resize, center crop, and normalize the image
        image_size = 384
        trans_totensor = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        # Convert the image to a tensor
        input_tensor_depth = trans_totensor(pil_image).unsqueeze(0).to(self.device)
        
        # Resize the image to fit the model
        pil_image_resized = pil_image.resize((640, 480), PIL.Image.BICUBIC)
        input_tensor_normal = self.trans_totensor_normal(pil_image_resized).unsqueeze(0).to(self.device)
        
        # Perform inference using the model
        with torch.no_grad():
            self.process_depth(input_tensor_depth)
            self.process_normal(input_tensor_normal)
            self.process_rgb(cv_image)

def main():
    rospy.init_node('omnidata_node', anonymous=True)
    omnidata = Omnidata()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("Error: Unable to open camera")
        sys.exit(1)

    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if ret:
                omnidata.image_callback(frame)
            else:
                rospy.logerr("Error: Unable to read frame from camera")
                break
    except rospy.ROSInterruptException:
        pass
    finally:
        cap.release()

if __name__ == '__main__':
    main()
