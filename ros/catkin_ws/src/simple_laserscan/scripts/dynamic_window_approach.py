#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import rospy


from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D,PoseArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import *

import math
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

import time

# import tf2_ros
# from tf.transformations import euler_from_quaternion


def euler_from_quaternion(q):

    # quat = [msg.pose[-2].orientation.x,
    #         msg.pose[-2].orientation.y,
    #         msg.pose[-2].orientation.z,
    #         msg.pose[-2].orientation.w]
    # siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
    # cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
    # yaw = math.atan2(siny_cosp, cosy_cosp)

    # sinr_cosp = 2 * (q.w * q.x * q.y * q.z)
    # cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    # sinp = 2 * (q.w * q.y - q.z * q.x)
    # roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch = math.asin(sinp)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw

def toRadian(degree):                    # degree to radian
    if degree > 180:
        degree = degree - 180
    elif degree < -180:
        degree = degree + 180
    radian = degree / 180 * np.pi
    return radian

def toDegree(radian):
    if radian > 3.14:
        radian = radian - radian
    elif radian < -3.14:
        radian = radian + 3.14
    degree = radian / np.pi * 180
    return degree


def motion(x, u, dt):
    """
    motion model
    :param u[0] : linear velocity
    :param u[1] : angular velocity
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    :param x_init:   initial state -> x(m), y(m), yaw(s), linear.x(m/s), angular.z(m/s) 
    :param v     :   linear velocity
    :param y     :   angular velocity
    :param config:   some parameters
    """
    x = np.array(x_init)
    traj = np.array(x)
    t = 0
    while t <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        
        t1 = time.time()
        traj = np.vstack((traj, x))
        # print(time.time() - t1)
        t += config.dt

    return traj


class RobotType(Enum):
    circle = 0
    rectangle = 1
    
class DWA():
    def __init__(self, config):

        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # goal position [x(m), y(m)]
        self.goal = np.array([0.0, 0.0])
        self.trajectorys = []
        self.best_trajectory = np.array([])
        self.config = config

        # obstacles [x(m) y(m), ....]
        self.object_point = np.array([[50.0, 50.0]])

        self.zero_object_point = np.array([[50.0, 50.0]])

        #self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pub = rospy.Publisher("/recommended_cmd_vel", Twist, queue_size=1)

        # rospy.Subscriber("/odom", Odometry, self.odomCallback)
        # rospy.Subscriber("/gazebo/model_states", ModelStates, self.goalCallback)
        # self.object_subscriber = rospy.Subscriber("object_points", Float32MultiArray,  self.objectCallback)

    
    #クオータニオンをオイラー角に変換してyaw[rad]を取得
    def getYaw(self, quat):
        # q = [quat.x, quat.y, quat.z, quat.w]
        yaw = euler_from_quaternion(quat)

        return yaw

    def PlotTrajecory(self, axl, trajectorys, best_trajectory):
        '''
        Plot the trajectorys
        '''

        for i in range(len(trajectorys)):
            axl.plot(trajectorys[i][:,0], trajectorys[i][:,1], 'r')
            plt.show()

        # axl.plot(best_trajectory[:,0], best_trajectory[:,1], 'r')
        # plt.show()


    # 得到小车当前的状态
    def odomCallback(self, data):
        # print('Odom')
        odom = data

        self.x[0] = odom.pose.pose.position.x
        self.x[1] = odom.pose.pose.position.y 
        
        self.x[2] = self.getYaw(odom.pose.pose.orientation)
        # print('yaw: ', self.x[2])

        self.x[3] = odom.twist.twist.linear.x
        self.x[4] = odom.twist.twist.angular.z

        # print(self.x)

    # 得到目标 target 的位置和周围人的位置
    def goalCallback(self, msg):
        # print('Here')                
        # self.goal[0] = msg.pose[-3].position.x
        # self.goal[1] = msg.pose[-3].position.y
        # print('goal', self.goal)

        get_model = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        model_people = GetModelStateRequest()
        people_name = 'target_robot'
        model_people.model_name = people_name
        people_msg = get_model(model_people)
        self.goal[0] = people_msg.pose.position.x
        self.goal[1] = people_msg.pose.position.y
        #print('goal', self.goal)

        tmp_ob = []
        for i in range(1, 21):
            people_name = 'people' + str(i) + '_robot'
            model_people.model_name = people_name
            people_msg = get_model(model_people)
            people_pos = np.array([people_msg.pose.position.x, people_msg.pose.position.y])
            tmp_ob.append(people_pos)
        #print('tmp_ob: ', np.array(tmp_ob).shape)

        # tmp_ob = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 
        #           np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 
        #           np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 
        #           np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]

        self.object_point = np.array(tmp_ob)

        # print('x: ', self.x)
        # print('goal: ', self.goal)
        u, trajectorys, best_trajectory = self.dwa_control(self.x, self.config, self.goal, self.object_point)
        self.trajectorys = trajectorys
        self.best_trajectory = best_trajectory
        
        # 目标距离判定
        dist_to_goal = math.hypot(self.x[0] - self.goal[0], self.x[1] - self.goal[1])
        #print('action: ', u)

        if dist_to_goal <= 0.3:
            u = [0.0, 0.0]
            self.publishTwist(u)
        else:
            self.publishTwist(u)

        #rospy.sleep(2)
        
        # self.PlotTrajecory(self.axl, trajectorys, best_trajectory)
        

    #制御量を出力
    def publishTwist(self, u):
        twist = Twist()
        twist.linear.x = u[0]
        twist.angular.z = u[1]

        self.pub.publish(twist)

    def dwa_control(self, x, config, goal, ob):
        """
        Dynamic Window Approach control
        """

        dw = self.calc_dynamic_window(x, config)   #  [vmin, vmax, yaw_rate min, yaw_rate max]

        u, trajectorys, best_trajectory = self.calc_control_and_trajectory(x, dw, config, goal, ob)

        return u, trajectorys, best_trajectory


    def calc_dynamic_window(self, x, config):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [config.min_speed, 
              config.max_speed,
             -config.max_yawrate, 
              config.max_yawrate]

        # Dynamic window from motion model
        Vd = [x[3] - config.max_accel * config.dt,      # vmin
              x[3] + config.max_accel * config.dt,      # vmax
              x[4] - config.max_dyawrate * config.dt,   # wmin
              x[4] + config.max_dyawrate * config.dt]   # wmax

        #  [vmin, vmax, yaw_rate min, yaw_rate max]
        dw = [max(Vs[0], Vd[0]), 
              min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), 
              min(Vs[3], Vd[3])]

        return dw

    def calc_control_and_trajectory(self, x, dw, config, goal, ob):
        """
        calculation final input with dynamic window
        :param x     :  current state ->  x(m), y(m), yaw(s), linear.x(m/s), angular.z(m/s) 
        :param dw    :  [vmin, vmax, wmin, wmax]
        :param config:  
        :param goal  :  [x, y]
        :ob          :  
        """
        t2 = time.time()

        x_init = x[:]
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        trajectorys = []
        v_ = np.array([])
        omega = np.array([])
        to_goal_costs = np.array([])
        speed_costs = np.array([])
        ob_costs = np.array([])

        for v in np.arange(dw[0], dw[1], config.v_reso):
            for y in np.arange(dw[2], dw[3], config.yawrate_reso):
                t5 = time.time()
                # path 计算
                path = predict_trajectory(x_init, v, y, config)     # 每次产生一条轨迹 path
                trajectorys.append(path[:,0:2])
                t5_ = time.time() - t5
                # print(t5_ * 100 , "t5")
                v_ = np.append(v_, v)
                omega = np.append(omega, y)

                t4 = time.time()
                # cost calculation
                # goal_cost = self.CalcHeadingEval(path, goal)
                goal_cost = self.CalcDistEval(path, goal) 
                speed_cost = path[-1, 3]                            # linear.x
                ob_cost = self.calc_obstacle_cost(path, ob, config) # 此条轨迹中每步离障碍物最近的距离  ob: np, 20 x 2
                # print(100 * (time.time() - t4), "t4")

                to_goal_costs = np.append(to_goal_costs, goal_cost)
                speed_costs = np.append(speed_costs, speed_cost)
                ob_costs = np.append(ob_costs, ob_cost)

        t1 = time.time()
        # normalize
        for scores in [to_goal_costs, speed_costs, ob_costs]:            
            scores = self.min_max_normalize(scores)

        #コストが最大になるPathを探索
        max_cost = 0.0
        for i in range(len(v_)):     # 找到那个使得final_cost最大的那个速度
            final_cost = 0
            final_cost = config.to_goal_cost_gain * to_goal_costs[i] + config.speed_cost_gain * speed_costs[i] + config.obstacle_cost_gain * ob_costs[i]
            # final_cost = config.to_goal_cost_gain * to_goal_costs[i]

            if final_cost > max_cost:
                max_cost = final_cost
                best_u = [v_[i],-omega[i]]
                best_trajectory = trajectorys[i]
        

        return best_u, trajectorys, best_trajectory


    def calc_obstacle_cost(self, trajectory, ob, config):
        """
            calc obstacle cost inf: collision
            :param trajectory:   one trajectory
            :param ob        :   obstacle list  np.array(20x2)
        """
        ox = ob[:, 0]
        # print('ox: ', ox.shape)
        oy = ob[:, 1]
        # print('op: ', trajectory[:, 0].shape)
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)
        # print('r: ', r)

        if (r <= config.robot_radius).any():
            return 0

        min_r = np.min(r)   # 找出最小的那个距离
        # print('min_r: ', min_r)
        return min_r  # OK

    def CalcHeadingEval(self, trajectory, goal):               # heading的评价函数计算
        #print('angle p:', trajectory[-1, 2])
        theta = toDegree(trajectory[-1, 2])  # 机器人朝向
        #print('theta is: ',theta)
        goalTheta = toDegree(math.atan2(goal[1] - trajectory[-1, 1], goal[0] - trajectory[-1, 0]))
        #print('goalTheta: ', goalTheta)
        if goalTheta > theta:
            targetTheta = goalTheta - theta  # degree
        else:
            targetTheta = theta - goalTheta

        # targetTheta = toRadian(targetTheta)
        # print('targetTheta: ', targetTheta)
        
        heading = 180 - targetTheta                   # when targetTheta is smaller, heading will be bigger 
        return heading  


    def CalcDistEval(self, trajectory, goal):
        dy = goal[1] - trajectory[-1, 1]
        dx = goal[0] - trajectory[-1, 0]
        dist = math.sqrt(dy*dy + dx*dx)
        return 15 - dist


    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        # print('dx', dx)
        # print('dy', dy)
        error_angle = math.atan2(dy, dx)
        # print('angle: ', error_angle)
        #print('tra angle: ', trajectory[-1, 2])
        cost_angle = error_angle - trajectory[-1, 2]
        #print('cost angle: ', cost_angle)
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return math.pi - cost

    # 正規化
    def min_max_normalize(self, data):

        if data.shape[0] > 0:
            max_data = np.max(data)
            min_data = np.min(data)

            if max_data - min_data == 0:
                data = [0.0 for i in range(len(data))]
            else:
                data = (data - min_data) / (max_data - min_data)
            return data
        else:
            return [0.0]


