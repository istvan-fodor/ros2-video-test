import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import torch
from transformers import SamModel, SamProcessor
import numpy as np

# Load the SAM2 model and processor
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")


class CameraReceiver(Node):
    def __init__(self):
        super().__init__('camera_receiver')

        # Create a subscription to the 'camera/image' topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            10
        )

        # Create a publisher to send the segmented part to a new topic 'sensor/image'
        self.sensor_publisher = self.create_publisher(Image, 'sensor/image', 10)

        # Initialize CvBridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()

        # Store the latest image in a class attribute
        self.latest_image = None

        # Create a timer that fires every 1 second (1000 ms)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def image_callback(self, msg):
        # Store the latest image message when it arrives
        self.latest_image = msg

    def timer_callback(self):
        # Only process the latest image if it exists
        if self.latest_image is not None:
            self.get_logger().info('Processing the latest image...')

            # Convert the ROS Image message to an OpenCV image
            frame = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')
            
            # Call the AI stub to simulate detecting a sensor
            sensor_image = ai_model_segment_image(frame)
            
            if sensor_image is not None:
                self.get_logger().info('Sensor detected! Publishing segmented image...')
                
                # Convert the cropped sensor image back to a ROS Image message
                sensor_msg = self.bridge.cv2_to_imgmsg(sensor_image, encoding='bgr8')

                # Publish the segmented sensor image to the 'sensor/image' topic
                self.sensor_publisher.publish(sensor_msg)
            else:
                self.get_logger().info('No sensor detected.')


def ai_model_segment_image(image):
    # Convert OpenCV image (BGR) to RGB since models usually expect RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare the image for the SAM2 model
    inputs = sam_processor(images=image_rgb, return_tensors="pt").to(device)
    
    # Run inference to get the mask
    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    # Assuming the model returns masks, and we want the first one
    masks = outputs["masks"].squeeze().cpu().numpy()

    # Find the largest mask (this would likely correspond to the main object in the image)
    largest_mask = np.argmax(masks, axis=0)

    # Apply the mask to the image (segment the object)
    mask_applied = np.where(largest_mask[..., None], image, 0)

    return mask_applied

def main(args=None):
    rclpy.init(args=args)
    camera_receiver = CameraReceiver()
    rclpy.spin(camera_receiver)
    camera_receiver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
