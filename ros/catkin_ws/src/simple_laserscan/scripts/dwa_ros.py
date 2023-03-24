#! /usr/bin/env python
# -*- coding: utf-8 -*-

# import queue
import rospy


from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D,PoseArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from gazebo_msgs.srv import *
import matplotlib.pyplot as plt
from gazebo_msgs.msg import ModelStates, ModelState



import math
from enum import Enum
import numpy as np


import time

# import tf2_ros
# from tf.transformations import euler_from_quaternion

import dynamic_window_approach as dwa



class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 0.4  # [m/s]
        self.min_speed = 0.01  # [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s]        # 最高角速度   0.69 rad/s
        self.max_accel = 1.0  # [m/ss]                              # 最高线加速度
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss]      # 最高角加速度 0.69 rad/ss
        self.v_reso = 0.05  # [m/s]                                 
        self.yawrate_reso = 3 * math.pi / 180.0  # [rad/s]        # 0.00174 rad/s        
        self.dt = 0.3  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.8
        self.speed_cost_gain = 0.3
        self.obstacle_cost_gain = 0.96
        self.robot_type = dwa.RobotType.circle
        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.34  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check

    # @property
    # def robot_type(self):
    #     return self._robot_type

    # @robot_type.setter
    # def robot_type(self, value):
    #     if not isinstance(value, RobotType):
    #         raise TypeError("robot_type must be an instance of RobotType")
    #     self._robot_type = value

def PlotTrajecory(axl, trajectorys, best_trajectory):
    '''
    Plot the trajectorys
    '''
    filePath = '/home/yangbo/environments_python3/my_env3.8/picture'
    # print('len: ', len(trajectorys))
    for i in range(len(trajectorys)):
        axl.plot(trajectorys[i][:,0], trajectorys[i][:,1], 'r')
    # print('pp: ', best_trajectory)
    if best_trajectory.shape[0] > 0:
        axl.plot(best_trajectory[:,0], best_trajectory[:,1], 'b')
    #plt.show()
    #plt.close()
    plt.savefig(filePath)
    #print('p')


if __name__ == "__main__":
    rospy.init_node("dwa_ros")

    config = Config()
    config.robot_type = dwa.RobotType.circle
    fig = plt.figure(figsize=(5, 4))  # figsize是常用的参数.(宽，高)
    axl = fig.add_subplot(1, 1, 1)
    dwa_ros = dwa.DWA(config)
    r = rospy.Rate(2)
    r.sleep()

    while not rospy.is_shutdown():

        rospy.Subscriber("/gazebo/model_states", ModelStates, dwa_ros.goalCallback) 
        rospy.Subscriber("/odom", Odometry, dwa_ros.odomCallback, queue_size=1)
        PlotTrajecory(axl, dwa_ros.trajectorys, dwa_ros.best_trajectory)
        plt.cla()
        #print('Here2')
        #print('len: ', len(dwa_ros.trajectorys))
        # flag = True
        # if flag:
        #     if len(dwa_ros.trajectorys) > 0:
        #         tmp_data = np.array(dwa_ros.trajectorys)
        #         print('save data!')
        #         np.save('/home/yangbo/environments_python3/my_env3.8/trajectory_data.npy', tmp_data)
        #         flag = False
        r.sleep()

