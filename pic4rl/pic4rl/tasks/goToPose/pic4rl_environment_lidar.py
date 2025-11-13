#!/usr/bin/env python3

import os
import numpy as np
from numpy import savetxt
import math
import subprocess
import json
import random
import sys
import time
import datetime
import yaml
import logging
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.sensors import Sensors
from pic4rl.utils.env_utils import *
from pic4rl.testing.nav_metrics import Navigation_Metrics


class Pic4rlEnvironmentLidar(Node):
    def __init__(self):
        """ """
        super().__init__("pic4rl_training_lidar")
        self.declare_parameter("package_name", "pic4rl")
        self.declare_parameter("training_params_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("main_params_path", rclpy.Parameter.Type.STRING)
        self.package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )
        goals_path = os.path.join(
            get_package_share_directory(self.package_name), "goals_and_poses"
        )
        self.main_params_path = self.get_parameter("main_params_path").get_parameter_value().string_value
        train_params_path = self.get_parameter("training_params_path").get_parameter_value().string_value
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), "models/goal_box/model.sdf"
        )

        with open(train_params_path, "r") as train_param_file:
            train_params = yaml.safe_load(train_param_file)["training_params"]

        self.declare_parameters(
            namespace="",
            parameters=[
                ("mode", rclpy.Parameter.Type.STRING),
                ("data_path", rclpy.Parameter.Type.STRING),
                ("robot_name", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
                ("update_frequency", rclpy.Parameter.Type.DOUBLE),
                ("sensor", rclpy.Parameter.Type.STRING),
            ],
        )

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        goals_path = os.path.join(goals_path, self.mode)
        self.data_path = (
            self.get_parameter("data_path").get_parameter_value().string_value
        )
        self.data_path = os.path.join(goals_path, self.data_path)
        print(train_params["--change_goal_and_pose"])
        self.change_episode = int(train_params["--change_goal_and_pose"])
        self.starting_episodes = int(train_params["--starting_episodes"])
        self.timeout_steps = int(train_params["--episode-max-steps"])
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        self.goal_tolerance = (
            self.get_parameter("goal_tolerance").get_parameter_value().double_value
        )
        self.lidar_distance = (
            self.get_parameter("laser_param.max_distance")
            .get_parameter_value()
            .double_value
        )
        self.lidar_points = (
            self.get_parameter("laser_param.num_points")
            .get_parameter_value()
            .integer_value
        )
        self.update_freq = (
            self.get_parameter("update_frequency").get_parameter_value().double_value
        )
        self.sensor_type = (
            self.get_parameter("sensor").get_parameter_value().string_value
        )

        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)
        log_path = os.path.join(get_package_share_directory(self.package_name),'../../../../', train_params["--logdir"])

        self.logdir = create_logdir(
            train_params["--policy"], self.sensor_type, log_path
        )
        self.get_logger().info(f"Logdir: {self.logdir}")
        
        if "--model-dir" in train_params:
            self.model_path = os.path.join(get_package_share_directory(self.package_name),'../../../../', train_params["--model-dir"])
        if "--rb-path-load" in train_params:
            self.rb_path_load = os.path.join(get_package_share_directory(self.package_name),'../../../../', train_params["--rb-path-load"])

        self.spin_sensors_callbacks()

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", qos)

        self.reset_world_client = self.create_client(Empty, "reset_world")
        self.pause_physics_client = self.create_client(Empty, "pause_physics")
        self.unpause_physics_client = self.create_client(Empty, "unpause_physics")

        self.episode_step = 0
        self.previous_twist = Twist()
        self.episode = 0
        self.collision_count = 0
        self.t0 = 0.0
        self.evaluate = False
        self.index = 0

        self.initial_pose, self.goals, self.poses = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]

        self.get_logger().info(f"Gym mode: {self.mode}")
        if self.mode == "testing":
            self.nav_metrics = Navigation_Metrics(self.logdir)
        self.get_logger().debug("PIC4RL_Environment: Starting process")

    def step(self, action, episode_step=0):
        """ """
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.episode_step = episode_step

        observation, reward, done = self._step(twist)

        # 🔍 关键调试：每步都打印action
        if episode_step <= 10 or episode_step % 20 == 0:
            goal_distance = observation[0]
            goal_angle = observation[1]
            lidar_readings = observation[2:]
            lidar_min = lidar_readings.min()
            lidar_max = lidar_readings.max()
            lidar_mean = lidar_readings.mean()

            logging.info("="*70)
            logging.info(f"[DEBUG Step {episode_step}]")
            logging.info(f"  RAW ACTION from model: v={action[0]:.6f}, w={action[1]:.6f}")
            logging.info(f"  Twist published: v={twist.linear.x:.6f}, w={twist.angular.z:.6f}")
            logging.info(f"  Goal: distance={goal_distance:.4f}, angle={goal_angle:.4f} rad ({np.rad2deg(goal_angle):.1f}°)")
            logging.info(f"  Lidar: min={lidar_min:.4f}, max={lidar_max:.4f}, mean={lidar_mean:.4f}")
            logging.info(f"  危险点(<0.5m): {np.sum(lidar_readings < 0.5)}/{len(lidar_readings)}")
            logging.info("="*70)           

        info = None

        return observation, reward, done, info

    def _step(self, twist=Twist(), reset_step=False):
        """ """
        self.get_logger().debug("sending action...")
        self.send_action(twist)

        self.get_logger().debug("getting sensor data...")
        self.spin_sensors_callbacks()
        (
            lidar_measurements,
            goal_info,
            robot_pose,
            collision,
        ) = self.get_sensor_data()

        if not reset_step:
            if self.mode == "testing":
                self.nav_metrics.get_metrics_data(lidar_measurements, self.episode_step)

            self.get_logger().debug("checking events...")
            done, event = self.check_events(
                lidar_measurements, goal_info, robot_pose, collision
            )
            self.get_logger().debug("getting reward...")
            reward = self.get_reward(
                twist, lidar_measurements, goal_info, robot_pose, done, event
            )

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(
                twist, lidar_measurements, goal_info, robot_pose
            )
        else:
            reward = None
            observation = None
            done = False
            event = None

        self.update_state(twist, lidar_measurements, goal_info, robot_pose, done, event)

        return observation, reward, done

    def get_goals_and_poses(self):
        """ """
        data = json.load(open(self.data_path, "r"))

        return data["initial_pose"], data["goals"], data["poses"]

    def spin_sensors_callbacks(self):
        """ """
        self.get_logger().debug("spinning node...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            empty_measurements = [ k for k, v in self.sensors.sensor_msg.items() if v is None]
            self.get_logger().debug(f"empty_measurements: {empty_measurements}")
            rclpy.spin_once(self)
            self.get_logger().debug("spin once ...")
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def send_action(self, twist):
        """ """
        # print(f"[DEBUG send_action] Publishing twist: v={twist.linear.x:.6f}, w={twist.angular.z:.6f}")
        
        self.cmd_vel_pub.publish(twist)

        # print(f"[DEBUG send_action] Twist published successfully")

        # Regulate frequency of send action if needed
        freq, t1 = compute_frequency(self.t0)
        self.get_logger().debug(f"frequency : {freq}")
        self.t0 = t1
        if freq > self.update_freq:
            frequency_control(self.update_freq)

        # self.get_logger().debug("pausing...")
        # self.pause()

    def get_sensor_data(self):
        """ """
        sensor_data = {}
        sensor_data["scan"], collision = self.sensors.get_laser()
        sensor_data["odom"] = self.sensors.get_odom(vel=False)

        if sensor_data["scan"] is None:
            sensor_data["scan"] = (
                np.ones(self.lidar_points) * self.lidar_distance
            ).tolist()
        if sensor_data["odom"] is None:
            sensor_data["odom"] = [0.0, 0.0, 0.0]

        goal_info, robot_pose = process_odom(self.goal_pose, sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]

        return lidar_measurements, goal_info, robot_pose, collision

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
        """ """
        if collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info(
                    f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision"
                )
                logging.info(
                    f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision"
                )
                return True, "collision"
            else:
                return False, "None"

        if goal_info[0] < self.goal_tolerance:
            self.get_logger().info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal"
            )
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            return True, "goal"

        if self.episode_step + 1 == self.timeout_steps:
            self.get_logger().info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            logging.info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            return True, "timeout"

        return False, "None"

    def get_reward(self, twist, lidar_measurements, goal_info, robot_pose, done, event):
        """ 
        设计原则：
        1. 避免reward符号陷阱（远离目标不应被过度惩罚）
        2. 所有项都应该连续可微
        3. 终止状态奖励与步奖励在同一量级
        """
        
        # 1. 距离奖励（只奖励接近）
        distance_change = self.previous_goal_info[0] - goal_info[0]
        distance_reward = max(0, distance_change * 5.0)
        
        # 2. 距离惩罚（距离越远越焦虑）
        distance_penalty = -0.02 * goal_info[0]  # 每米-0.02
        
        # 3. 朝向奖励（归一化到[0,1]）
        heading_reward = (1 - abs(goal_info[1]) / math.pi) * 2.0
        
        # 4. 障碍物惩罚（连续指数增长）
        min_obstacle_dist = min(lidar_measurements)
        safe_distance = 0.6
        
        if min_obstacle_dist < safe_distance:
            ratio = (safe_distance - min_obstacle_dist) / safe_distance
            obstacle_penalty = -10.0 * (ratio ** 2)
        else:
            obstacle_penalty = 0.0
        
        # 5. 速度奖励（鼓励前进）
        speed_reward = twist.linear.x * 0.3
        
        # 6. 平滑惩罚（已经对准时才惩罚过度转向）
        if abs(goal_info[1]) < 0.2:
            angular_penalty = -0.3 * abs(twist.angular.z)
        else:
            angular_penalty = 0.0
        
        # 7. 组合reward
        reward = (distance_reward + distance_penalty + heading_reward + 
                obstacle_penalty + speed_reward + angular_penalty)
        
        # 8. 终止状态（与步奖励在同一量级）
        if event == "goal":
            reward += 100  # 相当于前进20米
        elif event == "collision":
            reward += -50  # 相当于10次危险接近
        elif event == "timeout":
            reward += -10  # 轻微惩罚
            
        return reward

    def get_observation(self, twist, lidar_measurements, goal_info, robot_pose):
        """ """
        state_list = goal_info
        
        # Add lidar measurements to the observation
        lidar_array = np.array(lidar_measurements, dtype=np.float32)
        state_list = np.concatenate((state_list, lidar_array))

        state = np.array(state_list, dtype=np.float32)

        return state

    def update_state(
        self, twist, lidar_measurements, goal_info, robot_pose, done, event
    ):
        """ """
        self.previous_twist = twist
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose

    def reset(self, n_episode, tot_steps, evaluate=False):
        """ """
        # 在testing模式下,只在非第一个episode时保存上一个episode的指标
        # 注意: 这里保存的是上一个episode的数据,因为reset在新episode开始时被调用
        if self.mode == "testing" and n_episode > 0:
            self.nav_metrics.calc_metrics(n_episode - 1, self.initial_pose, self.goal_pose)
            self.nav_metrics.log_metrics_results(n_episode - 1)
            self.nav_metrics.save_metrics_results(n_episode - 1)

        self.episode = n_episode
        self.evaluate = evaluate
        logging.info(
            f"Total_episodes: {n_episode}{' evaluation episode' if self.evaluate else ''}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n"
        )
        print()
        self.get_logger().info("Initializing new episode ...")
        logging.info("Initializing new episode ...")
        self.new_episode()
        self.get_logger().debug("Performing null step to reset variables")
        self.episode_step = 0

        _, _, _ = self._step(reset_step=True)
        observation, _, _ = self._step()

        return observation

    def new_episode(self):
        """ """
        self.get_logger().debug("Resetting simulation ...")
        req = Empty.Request()

        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")
        self.reset_world_client.call_async(req)

        if self.evaluate:
            # evaluate模式：顺序遍历所有配对，确保每个都被测试到
            self.index = self.episode % len(self.poses)
        elif self.episode % self.change_episode == 0.0:
            # 训练模式：随机选择配对
            self.index = np.random.randint(0, len(self.poses))

        self.get_logger().debug("Respawing robot ...")
        self.respawn_robot(self.index)

        self.get_logger().debug("Respawing goal ...")
        self.respawn_goal(self.index)

        self.get_logger().debug("Environment reset performed ...")

    def respawn_goal(self, index):
        """ """
        if self.evaluate:
            # evaluate模式直接使用goals[index]确保配对正确
            self.get_goal(index)
        elif self.episode <= self.starting_episodes:
            self.get_random_goal()
        else:
            self.get_goal(index)

        self.get_logger().info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}"
        )
        logging.info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}"
        )


    def get_goal(self, index):
        """ """
        self.goal_pose = self.goals[index]

    def get_random_goal(self):
        """ """
        if self.episode < 6 or self.episode % 25 == 0:
            x = 0.55
            y = 0.55
        else:
            x = random.randrange(-29, 29) / 10.0
            y = random.randrange(-29, 29) / 10.0

        x += self.initial_pose[0]
        y += self.initial_pose[1]

        self.goal_pose = [x, y]

    def respawn_robot(self, index):
        """ """
        if self.evaluate:
            # evaluate模式直接使用poses[index]确保配对正确
            x, y, yaw = tuple(self.poses[index])
        elif self.episode <= self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y, yaw = tuple(self.poses[index])

        qz = np.sin(yaw / 2)
        qw = np.cos(yaw / 2)

        self.get_logger().info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}"
        )
        logging.info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}"
        )

        position = (
            "position: {x: " + str(x) + ",y: " + str(y) + ",z: " + str(0.07) + "}"
        )
        orientation = "orientation: {z: " + str(qz) + ",w: " + str(qw) + "}"
        pose = position + ", " + orientation
        state = "'{state: {name: '" + self.robot_name + "',pose: {" + pose + "}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "
            + state,
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        time.sleep(0.25)

    def pause(self):
        """ """
        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")
        future = self.pause_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def unpause(self):
        """ """
        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")
        future = self.unpause_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
