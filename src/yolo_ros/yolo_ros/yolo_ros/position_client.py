#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from yolo_msgs.srv import GetTargetPosition

class PositionClient(Node):

    def __init__(self):
        super().__init__("position_client")
        self.cli = self.create_client(GetTargetPosition, "/yolo/get_target_position")

    def call(self, name):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for service...")

        req = GetTargetPosition.Request()
        req.class_name = name

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    if len(sys.argv) < 2:
        print("Usage: ros2 run yolo_ros position <class_name>")
        return

    target = sys.argv[1]

    rclpy.init()
    node = PositionClient()
    res = node.call(target)
    node.destroy_node()
    rclpy.shutdown()

    if res is None or not res.success:
        print("No detection found.")
        return

    print(f"class: {target}")
    print(f"x: {res.x:.2f}")
    print(f"y: {res.y:.2f}")
    print(f"z: {res.z:.2f}")
    print(f"frame: {res.frame_id}")

if __name__ == "__main__":
    main()
