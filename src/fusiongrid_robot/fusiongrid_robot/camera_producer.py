import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # Declare parameters
        self.declare_parameter('compressed', True)
        self.declare_parameter('ResizeX', 0)
        
        self.compressed = self.get_parameter('compressed').value
        self.resize_x = self.get_parameter('ResizeX').value

        # Create publisher based on 'compressed' parameter
        topic = 'camera/image/compressed' if self.compressed else 'camera/image'
        msg_type = CompressedImage if self.compressed else Image
        self.publisher_ = self.create_publisher(msg_type, topic, 10)

        self.timer = self.create_timer(1.0/30.0, self.publish_image)  
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
        
        # Resize the image if ResizeX is set
        if self.resize_x > 0:
            frame = self.resize_image(frame, self.resize_x)

        if self.compressed:
            # Publish as CompressedImage
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
            if not success:
                self.get_logger().error('Failed to encode image')
                return
            
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = 'jpeg'
            msg.data = encoded_image.tobytes()
        else:
            # Publish as normal Image
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()

        self.publisher_.publish(msg)
        self.get_logger().info(f'Image published on {self.publisher_.topic}')

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
