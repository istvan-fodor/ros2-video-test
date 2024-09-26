import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image', 10)
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

        # Convert OpenCV image (BGR format) to ROS2 Image message
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('Image published')

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
