import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        # Declare and get the video path parameter
        self.declare_parameter('video_path', '')
        self.declare_parameter('topic_name', '/image_raw')
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('repeat', True)


        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.repeat = self.get_parameter('repeat').get_parameter_value().bool_value

        # Verify that the video path is provided
        if not self.video_path:
            self.get_logger().error("No video file path specified. Set the 'video_path' parameter.")
            rclpy.shutdown()
            return

        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        self.init_video()
        self.timer = self.create_timer(1.0 / frame_rate, self.publish_frame)
        self.get_logger().info(f"Publishing video frames from {self.video_path} to {topic_name} at {frame_rate} FPS")

    def init_video(self):
         self.cap = cv2.VideoCapture(self.video_path)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            if self.repeat:
                self.get_logger().info("End of video file reached, restarting video.")
                self.init_video()
            else:
                self.get_logger().info("End of video file reached, stopping publisher.")
                self.destroy_node()
            return

        # Convert the frame to a ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info("Published a video frame")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
