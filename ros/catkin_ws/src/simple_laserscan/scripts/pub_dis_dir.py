#!/usr/bin/env python3.8

import rospy
import numpy as np
import math
from gazebo_msgs.srv import *
from simple_laserscan.msg import Spying
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates, ModelState

def compute_dis_dir_2_goal(target, robot):
     """
     compute the difference of distance and direction to goal position
     :param target: the position of target
     :param robot : the position of robot
     :return : distance, direction
     """
     delta_x = target[0] - robot[0]
     delta_y = target[1] - robot[1]
     distance = math.sqrt(delta_x**2 + delta_y**2)
     ego_direction = math.atan2(delta_y, delta_x)
     robot_direction = robot[2]   # yaw
     while robot_direction < 0:
          robot_direction += 2 * math.pi
     while robot_direction > 2 * math.pi:
          robot_direction -= 2 * math.pi
     while ego_direction < 0:
          ego_direction += 2 * math.pi
     while ego_direction > 2 * math.pi:
          ego_direction -= 2 * math.pi
     pos_dir = abs(ego_direction - robot_direction)
     neg_dir = 2 * math.pi - abs(ego_direction - robot_direction)
     if pos_dir <= neg_dir:
          direction = math.copysign(pos_dir, ego_direction - robot_direction)
     else:
          direction = math.copysign(neg_dir, -(ego_direction - robot_direction))
     # print(distance)
     # print(direction)
     return distance, direction  # direction 是有正有负的，根据右手定则规定正负


def process_pose(raw_pose):
     """
     :return : yaw
     """
     x, y, z, w = raw_pose
     siny_cosp = 2. * (x * y  +  z * w)
     cosy_cosp = 1. - 2. * (y**2 + z**2)
     yaw = math.atan2(siny_cosp, cosy_cosp)
     return yaw

# def state_cb(msg):
#      # print('msg callback:', msg.pose)
#      robot_quat = [msg.pose[-2].orientation.x,
#                    msg.pose[-2].orientation.y,
#                    msg.pose[-2].orientation.z,
#                    msg.pose[-2].orientation.w]
#      siny_cosp = 2. * (robot_quat[0] * robot_quat[1] + robot_quat[2] * robot_quat[3])
#      cosy_cosp = 1. - 2. * (robot_quat[1]**2 + robot_quat[2]**2)
#      robot_yaw = math.atan2(siny_cosp, cosy_cosp)
#      print('robot yaw', robot_yaw)
#      robot_position = [msg.pose[-2].position.x, msg.pose[-2].position.y, robot_yaw]
#      target_position = [msg.pose[-1].position.x, msg.pose[-1].position.y]
#      dis, dir = compute_dis_dir_2_goal(target_position, robot_position)
#      spying_data = Spying()
#      spying_data.distance = dis
#      spying_data.direction = dir
#      dis_dir_pub.publish(spying_data)
#      print('dis: ', dis)
#      print('dir: ', dir)
#      print('=======================')

# def robot_state_cb(msg):
#      # for i in range(-3, -24, -1):
#      #      robot_pose = [msg.pose[i].position.x, msg.pose[i].position.y]
#      #      print('people pose: {0}/{1}'.format(i, robot_pose))
#      # i = -7
#      # robot_pose = [msg.pose[i].position.x, msg.pose[i].position.y]
#      # print('people pose: {0}'.format(robot_pose))
#      get_model = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
#      model_people = GetModelStateRequest()
#      for i in range(1, 21):
#           people_name = 'people' + str(i) + '_robot'
#           model_people.model_name = people_name
#           people_msg = get_model(model_people)
#           people_pos = [people_msg.pose.position.x, people_msg.pose.position.y]
#           print(people_name + ': ', people_pos)


if __name__ == '__main__':
     
     rospy.init_node('pub_dis_dir')
     get_model = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
     model1 = GetModelStateRequest()
     model2 = GetModelStateRequest()
     model_people = GetModelStateRequest()
     # rospy.Subscriber('/gazebo/model_states', ModelStates, state_cb)
     dis_dir_pub = rospy.Publisher('/Spying_signal', Spying, queue_size=1)
     dis_around_people = rospy.Publisher('/around_people', Float32MultiArray, queue_size=1)
     # rospy.Subscriber('/gazebo/model_states', ModelStates, robot_state_cb)

     # rospy.spin()

     while not rospy.is_shutdown():

          spying_data = Spying()
          distance_around = Float32MultiArray()
          model1.model_name = 'target'
          target_pos_msg = get_model(model1)
          target_pos = [target_pos_msg.pose.position.x, target_pos_msg.pose.position.y]
          model2.model_name = 'robot'
          mrobot_pos_msg = get_model(model2)
          mrobot_pos = [mrobot_pos_msg.pose.position.x, mrobot_pos_msg.pose.position.y]
          mrobot_yaw = [mrobot_pos_msg.pose.orientation.x, mrobot_pos_msg.pose.orientation.y, \
                    mrobot_pos_msg.pose.orientation.z, mrobot_pos_msg.pose.orientation.w]
          yaw_rob = process_pose(mrobot_yaw)
          print('yaw_robot: ', yaw_rob)
          mrobot_pos.append(yaw_rob)

          dis, dir = compute_dis_dir_2_goal(target_pos, mrobot_pos)
          spying_data.distance = dis
          spying_data.direction = dir
          print('dis: ', dis)
          print('dir: ', dir)
          print('=======================')

          for i in range(1, 11):
               people_name = 'people' + str(i) + '_robot'
               model_people.model_name = people_name
               people_msg = get_model(model_people)
               people_pos = [people_msg.pose.position.x, people_msg.pose.position.y]
               dis, _ = compute_dis_dir_2_goal(people_pos, mrobot_pos)
               distance_around.data.append(dis-0.3)

          for i in range(11, 21):
               people_name = 'people' + str(i) + '_robot'
               model_people.model_name = people_name
               people_msg = get_model(model_people)
               people_pos = [people_msg.pose.position.x, people_msg.pose.position.y]
               dis, _ = compute_dis_dir_2_goal(people_pos, mrobot_pos)
               distance_around.data.append(dis)

          dis_dir_pub.publish(spying_data)
          dis_around_people.publish(distance_around)

     
