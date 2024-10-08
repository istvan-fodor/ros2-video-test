import sys
print("Curent Python: ", sys.executable)


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import numpy as np
from PIL import Image as PILImage


class CameraReceiver(Node):
    def __init__(self):
        super().__init__('camera_receiver')

        self.subscription = self.create_subscription(
            CompressedImage,
            'camera/image',
            self.image_callback,
            1
        )

        self.sensor_publisher = self.create_publisher(CompressedImage, 'sensor/image/segmented/compressed', 10)

        self.bridge = CvBridge()

        self.latest_image = None
        self.processing = False
    

    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 


    def overlay_segments(self, image, anns, borders=True):
    # Convert the input NumPy image to a PIL image
        base_img = PILImage.fromarray(image)

        if len(anns) == 0:
            return base_img  # Return the original image if there are no annotations

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        # Create a transparent RGBA image for the overlay
        overlay_img = PILImage.new('RGBA', base_img.size, (0, 0, 0, 0))  # Fully transparent image

        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = tuple((np.random.random(3) * 255).astype(np.uint8).tolist()) + (128,)  # Random color with alpha 128
            mask_img = PILImage.new('RGBA', base_img.size, color_mask)

            # Create the mask as a PIL image from the binary mask (use it to apply the overlay)
            mask = PILImage.fromarray(m.astype(np.uint8) * 255, mode='L')  # Binary mask in 'L' (grayscale)

            # Apply the mask to the overlay image
            overlay_img.paste(mask_img, (0, 0), mask)

            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(image, contours, -1, (0, 0, 255), thickness=1)  # Blue borders

        # Convert the overlay image to RGB to match the base image
        overlay_img = overlay_img.convert('RGB')
        print(type(overlay_img))
        # Blend the base image and overlay using PIL's blend function (alpha blending)
        blended_img = PILImage.blend(base_img, overlay_img, alpha=0.5)  # Adjust alpha as needed (0.0-1.0)

        return blended_img


    def image_callback(self, msg):
        if self.processing:
            self.get_logger().info('Still processing the previous image, skipping new message...')
            return

        self.processing = True

        try:
            self.get_logger().info('Received a new compressed image, processing...')
            self.latest_image = msg

            np_arr = np.frombuffer(msg.data, np.uint8)
    
            # Decode the image using OpenCV
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image = PILImage.fromarray(frame)
            image = np.array(image.convert("RGB"))

            masks = mask_generator.generate(image)
            overlaid_image = self.overlay_segments(image, masks, True)

            if overlaid_image != None:
                self.get_logger().info('Segments detected! Publishing segmented image...')
                overlaid_image = np.array(overlaid_image)
                overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
                sensor_msg = self.bridge.cv2_to_compressed_imgmsg(overlaid_image)
                self.sensor_publisher.publish(sensor_msg)


            
            # sensor_image = ai_model_segment_image(image, labels=['face'], threshold=0.4)
            
            # if sensor_image is not None:
            #     self.get_logger().info('Sensor detected! Publishing segmented image...')
                
            #     sensor_msg = self.bridge.cv2_to_compressed_imgmsg(sensor_image)

            #     self.sensor_publisher.publish(sensor_msg)
            # else:
            #     self.get_logger().info('No sensor detected.')

        except Exception as e:
            self.get_logger().error(f'Error during image processing: {e}')

        finally:
            self.processing = False


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
import os
cwd = os.getcwd()
print(cwd)
sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)


def main(args=None):
    rclpy.init(args=args)
    camera_receiver = CameraReceiver()
    rclpy.spin(camera_receiver)
    camera_receiver.destroy_node()
    rclpy.shutdown()
