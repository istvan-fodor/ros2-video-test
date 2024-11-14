#!/usr/bin/env python3
# coding: utf8
# Copyright (c) 2023 jk-ethz.

import rclpy
from rclpy.node import Node
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import tempfile
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
from analog_gauge_reader_msgs.msg import GaugeReading, GaugeReadings
from analog_gauge_reader_msgs.srv import GaugeReader as GaugeReaderSrv
from .pipeline import ImageProcessor

class AnalogGaugeReaderRos(Node):
    def __init__(self, debug=False):
        super().__init__("analog_gauge_reader")
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detection_model_path', ""),
                ('key_point_model_path', ""),
                ('segmentation_model_path', ""),
                ('image_topic', '/camera/image_raw'),
                ('round_decimals', -1),
                ('latch', False),
                ('continuous', True)
            ]
        )
        self.detection_model_path = self.get_parameter("detection_model_path").get_parameter_value().string_value
        self.key_point_model_path = self.get_parameter("key_point_model_path").get_parameter_value().string_value
        self.segmentation_model_path = self.get_parameter("segmentation_model_path").get_parameter_value().string_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.round_decimals = self.get_parameter("round_decimals").get_parameter_value().integer_value
        self.latch = self.get_parameter("latch").get_parameter_value().bool_value
        self.continuous = self.get_parameter("continuous").get_parameter_value().bool_value
        self.debug = debug

        self.readings_pub = self.create_publisher(GaugeReadings, "readings", 10)
        self.image_pub = self.create_publisher(Image, "analog_gauge_reader/visualization", 10)
        self.bridge = CvBridge()
        self.trigger_srv = self.create_service(GaugeReaderSrv, "read", self.read)
        self.image = None
        self.image_processor = ImageProcessor(self.detection_model_path, self.key_point_model_path, self.segmentation_model_path)
        self._init_subscribers()

    def _init_subscribers(self):
        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, 1)

    def _image_callback(self, msg: Image):
        self.image = msg
        self.get_logger().info("Got new image")
        if self.continuous:
            self.read(GaugeReaderSrv.Request(), GaugeReaderSrv.Response())

    def read(self, req: GaugeReaderSrv.Request, res: GaugeReaderSrv.Response):
        self.get_logger().info("Processing read request...")
        original_image = self.image
        if original_image is None:
            self.get_logger().warn("No image received yet, ignoring read request")
            return res

        try:
            image = self.bridge.imgmsg_to_cv2(original_image)
            with tempfile.TemporaryDirectory() as out_path:
                os.removedirs(out_path)
                gauge_readings = [self.image_processor.process_image(
                    image=image,
                    image_is_raw=True,
                    run_path=out_path,
                    debug=self.debug,
                    eval_mode=True
                )]

            for gauge_reading in gauge_readings:
                if gauge_reading["value"] is None:
                    raise Exception("Value reading failed")
                reading = GaugeReading()
                value = Float64()
                value.data = gauge_reading["value"] if self.round_decimals < 0 else round(gauge_reading["value"], self.round_decimals)
                unit = String()
                unit.data = gauge_reading["unit"] if gauge_reading["unit"] is not None else ''
                reading.value = value
                reading.unit = unit
                res.result.readings.append(reading)
            self.get_logger().info("Successfully processed read request.")
            self.readings_pub.publish(res.result)
            return res
        finally:
            self._init_subscribers()

def main(args=None):
    rclpy.init(args=args)
    analog_gauge_reader = AnalogGaugeReaderRos(debug=True)
    rclpy.spin(analog_gauge_reader)
    analog_gauge_reader.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
