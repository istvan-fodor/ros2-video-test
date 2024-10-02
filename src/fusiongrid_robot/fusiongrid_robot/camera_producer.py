import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, 'camera/image', 10)
        self.timer = self.create_timer(0.1, self.publish_image)  # Publish every 0.1 seconds
        self.bridge = CvBridge()

        # Open the laptop camera (0 is usually the default camera)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')

    def publish_image(self):
        ret, frame = self.cap.read()  # Capture frame from camera
        if not ret:
            self.get_logger().error('Failed to capture image')
            return
        
        resized_frame = self.resize_image(frame, 640)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # You can adjust the quality here
        success, encoded_image = cv2.imencode('.jpg', frame, encode_param)

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'  # Specify the format
        msg.data = encoded_image.tobytes()  # Convert the encoded image to bytes

        # Convert OpenCV image (BGR format) to ROS2 Image message
        self.publisher_.publish(msg)
        self.get_logger().info('Image published')

    def resize_image(self, image, max_width):
        h, w = image.shape[:2]
        scaling_factor = max_width / float(w)
        new_width = int(w * scaling_factor)
        new_height = int(h * scaling_factor)
        # Resize the image to the specified dimensions
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    def destroy_node(self):
        # Release the camera when the node is destroyed
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
