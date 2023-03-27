import rospy
import math
import time
import copy
import random
import torch
import numpy as np
from shapely.geometry import Point
from sensor_msgs.msg import LaserScan
from simple_laserscan.msg import Spying
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from std_msgs.msg import Float32MultiArray
import sys
import os
sys.path.append('../../')
from training.utility_original import ddpg_state_2_spike_value_state,wheeled_network_2_robot_action_decoder


class RandEvalGpu:
    """ Perform Random Evaluation on GPU """
    def __init__(self,
                 actor_net,
                 robot_init_pose_list,
                 goal_pos_list,
                 obstacle_poly_list,
                 ros_rate=10,
                 max_steps=2000, 
                 min_spd=0.05,
                 max_spd=0.5,
                 is_spike=False,
                 is_scale=False,
                 is_poisson=False,
                 batch_window=50,
                 action_rand=0.05,
                 scan_half_num=9,
                 scan_min_dis=0.22,
                 goal_dis_min_dis=0.3,
                 goal_th=0.5,
                 obs_near_th=0.18,
                 use_cuda=True,
                 is_record=False):
        """
        :param actor_net: Actor Network
        :param robot_init_pose_list: robot init pose list
        :param goal_pos_list: goal position list
        :param obstacle_poly_list: obstacle list
        :param ros_rate: ros rate
        :param max_steps: max step for single goal
        :param min_spd: min wheel speed
        :param max_spd: max wheel speed
        :param is_spike: is using SNN
        :param is_scale: is scale DDPG state input
        :param is_poisson: is use rand DDPG state input
        :param batch_window: batch window of SNN
        :param action_rand: random of action
        :param scan_half_num: half number of scan points
        :param scan_min_dis: min distance of scan
        :param goal_dis_min_dis: min distance of goal distance
        :param goal_th: distance for reach goal
        :param obs_near_th: distance for obstacle collision
        :param use_cuda: if true use cuda
        :param is_record: if true record running data
        """
        self.actor_net = actor_net
        self.robot_init_pose_list = robot_init_pose_list
        self.goal_pos_list = goal_pos_list
        self.obstacle_poly_list = obstacle_poly_list
        self.ros_rate = ros_rate
        self.max_steps = max_steps
        self.min_spd = min_spd
        self.max_spd = max_spd
        self.is_spike = is_spike
        self.is_scale = is_scale
        self.is_poisson = is_poisson
        self.batch_window = batch_window
        self.action_rand = action_rand
        self.scan_half_num = scan_half_num
        self.scan_min_dis = scan_min_dis
        self.goal_dis_min_dis = goal_dis_min_dis
        self.goal_th = goal_th
        self.obs_near_th = obs_near_th
        self.use_cuda = use_cuda
        self.is_record = is_record
        self.record_data = []
        # Put network to device
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.actor_net.to(self.device)
        # Robot State
        self.robot_state_init = False
        self.robot_scan_init  = False
        self.people_distance_init = False
        self.robot_pose = [0., 0.]
        self.robot_speed  = [0., 0.]
        self.robot_scan = np.zeros(360)
        self.goal_dis_dir_pre = [0., 0.]
        self.goal_dis_dir_cur = [0., 0.]
        # Subscriber
        rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        rospy.Subscriber('/scan', LaserScan, self._robot_scan_cb)
        rospy.Subscriber('/Spying_signal', Spying, self._robot_spying_cb)
        rospy.Subscriber('/around_people', Float32MultiArray, self._people_distance_cb)
        # Publisher
        self.pub_action = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        self.pub_action_people1 = rospy.Publisher('/people1_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people2 = rospy.Publisher('/people2_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people3 = rospy.Publisher('/people3_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people4 = rospy.Publisher('/people4_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people5 = rospy.Publisher('/people5_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people6 = rospy.Publisher('/people6_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people7 = rospy.Publisher('/people7_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people8 = rospy.Publisher('/people8_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people9 = rospy.Publisher('/people9_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people10 = rospy.Publisher('/people10_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people11 = rospy.Publisher('/people11_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people12 = rospy.Publisher('/people12_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people13 = rospy.Publisher('/people13_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people14 = rospy.Publisher('/people14_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people15 = rospy.Publisher('/people15_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people16 = rospy.Publisher('/people16_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people17 = rospy.Publisher('/people17_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people18 = rospy.Publisher('/people18_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people19 = rospy.Publisher('/people19_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people20 = rospy.Publisher('/people20_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people21 = rospy.Publisher('/people21_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people22 = rospy.Publisher('/people22_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people23 = rospy.Publisher('/people23_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people24 = rospy.Publisher('/people24_robot/cmd_vel', Twist, queue_size=5)
        self.pub_action_people25 = rospy.Publisher('/people25_robot/cmd_vel', Twist, queue_size=5)
        # Service
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        # Init Subscriber
        while not self.robot_state_init:
            print('loop state init!')
            continue
        while not self.robot_scan_init:
            print('loop scan init!')
            continue
        while not self.people_distance_init:
            print('loop people init!')
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def run_ros(self):
        """
        ROS ROS Node
        :return: run_data
        """
        rate = rospy.Rate(self.ros_rate)

        robot_pos_all_dict = {'robot_pos10':0,'robot_pos11':0,'robot_pos12':0,'robot_pos13':0,'robot_pos14':0,'robot_pos15':0,'robot_pos16':0,'robot_pos17':0,\
            'robot_pos18':0,'robot_pos19':0,'robot_pos20':0,'robot_pos21':0,'robot_pos22':0,'robot_pos23':0,'robot_pos24':0,'robot_pos25':0,'robot_pos26':0,\
                'robot_pos27':0,'robot_pos28':0,'robot_pos29':0}

        target_all_dict = {'target10':0,'target11':0,'target12':0,'target13':0,'target14':0,'target15':0,'target16':0,'target17':0,'target18':0,'target19':0,'target20':0,\
            'target21':0,'target22':0,'target23':0,'target24':0,'target25':0,'target26':0,'target27':0,'target28':0,'target29':0}

        people_dict = {'people1':0, 'people2':0, 'people3':0, 'people4':0, 'people5':0, 'people6':0, 'people7':0, 'people8':0, \
            'people9':0, 'people10':0, 'people11':0, 'people12':0, 'people13':0, 'people14':0, 'people15':0, 'people16':0, \
                'people17':0, 'people18':0, 'people19':0, 'people20':0, 'people21':0, 'people22':0, 'people23':0, 'people24':0, \
                    'people25':0}

        people_dict['people1'] = [5, 7, 0]
        people_dict['people2'] = [3, 9, -1.5]
        people_dict['people3'] = [7, 9, -2.2]
        people_dict['people4'] = [6, 9, 2.1]
        people_dict['people5'] = [9, 7, -1.22]
        people_dict['people6'] = [9, 7, 0.53]
        people_dict['people7'] = [13, 5, -2.1]
        people_dict['people8'] = [12, 4, 2.4]
        people_dict['people9'] = [10, 10, 2.6]
        people_dict['people10'] = [4, 13, 1.03]
        people_dict['people11'] = [11, 6, 3.14]
        people_dict['people12'] = [12, 9, -0.32]
        people_dict['people13'] = [7, 7, 1.563]
        people_dict['people14'] = [12, 9, -1.54]
        people_dict['people15'] = [4, 3, 0.67]
        people_dict['people16'] = [13, 5, 2.45]
        people_dict['people17'] = [11, 6, -0.89]
        people_dict['people18'] = [5, 5, 1.89]
        people_dict['people19'] = [3, 3, -2.34]
        people_dict['people20'] = [6, 3, 2.46]
        people_dict['people21'] = [2, 8, 0.12]
        people_dict['people22'] = [6, 1, 3.02]
        people_dict['people23'] = [3, 3, -0.78]
        people_dict['people24'] = [1, 6, -3.05]
        people_dict['people25'] = [8, 6, -1.35]

        robot_pos_all_dict['robot_pos10'] = [[10, 4, -1.25]]
        target_all_dict['target10'] = [[3, 7]]

        batch = 1
        hidden_size = 256
        device = torch.device('cuda')

        rates_num = 0
        rates = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        for k in [10]:
            robot_name = 'robot_pos' + str(k)
            target_name = 'target' + str(k)
            for n in range(1):
                robot_pos = robot_pos_all_dict[robot_name]
                target_pos = target_all_dict[target_name]
                data_robot_pos = robot_pos[n]
                data_target_pos = target_pos[n]
                for i in range(10):   # 对一组位置点做10次实验
                    
                    h0 = torch.zeros(batch, hidden_size).to(device)
                    self._set_new_target(robot_position=data_robot_pos, target_position=data_target_pos, \
                        people_dict=people_dict)
                    robot_path = []
                    flag_fail = False
                    flag1_time = time.time()
                    start_time = time.time()
                    while not rospy.is_shutdown():
                        rates_num += 1
                        flag2_time = time.time()
                        if abs(flag2_time - flag1_time) > 3:
                            flag1_time = flag2_time
                            if abs(self.goal_dis_dir_cur[0] - self.goal_dis_dir_pre[0]) < 0.01 \
                                and abs(self.goal_dis_dir_cur[1] - self.goal_dis_dir_pre[1]) < 0.01:
                                self.goal_dis_dir_pre = self.goal_dis_dir_cur
                                flag_fail = True
                            else:
                                self.goal_dis_dir_pre = self.goal_dis_dir_cur
                        tmp_robot_pose = copy.deepcopy(self.robot_pose)
                        tmp_robot_spd = copy.deepcopy(self.robot_speed)
                        tmp_robot_scan = copy.deepcopy(self.robot_scan)
                        tmp_robot_scan = tmp_robot_scan - 1
                        tmp_robot_scan[tmp_robot_scan <= 0] = 0.001
                        tmp_robot_scan = self.scan_min_dis / tmp_robot_scan
                        tmp_robot_scan = np.clip(tmp_robot_scan, 0, 1)
                        end_time = time.time()
                        tmp_robot_pose.append(end_time - start_time)
                        robot_path.append(tmp_robot_pose)
                        '''
                        Perform Action
                        '''
                        # tmp_goal_dis = goal_dis
                        tmp_goal_dis = self.goal_dis_dir_cur[0]
                        if tmp_goal_dis == 0:
                            tmp_goal_dis = 1
                        else:
                            tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis
                            if tmp_goal_dis > 1:
                                tmp_goal_dis = 1
                        
                        ddpg_state = []
                        ddpg_state.append(np.array([self.goal_dis_dir_cur[1], tmp_goal_dis, tmp_robot_spd[0], tmp_robot_spd[1]]))
                        ddpg_state.append(tmp_robot_scan)   # np: (364)
                        pre_rates = rates * (rates_num - 1)
                        action, h0, rates = self._network_2_robot_action(ddpg_state, h0, linear_spd_max=0.8, linear_spd_min=0.0)
                        # print('action: ', action)
                        avg_rates = (pre_rates + rates) / rates_num
                        print('rates: ', avg_rates)
                        print('=======================')
                        rates = avg_rates
                        move_cmd = Twist()
                        move_cmd.linear.x = action[0]
                        move_cmd.angular.z = action[1]
                        move_cmd_stop = Twist()
                        move_cmd_stop.linear.x = 0
                        move_cmd_stop.angular.z = 0
                        dis_people = min(self.people_distance)
                        # print('goal_dis_dir: ', self.goal_dis_dir_cur)

                        if self.goal_dis_dir_cur[0] < 1.0 or flag_fail or dis_people < 0.55:
                            self.pub_action.publish(move_cmd_stop)
                            self._set_new_target(robot_position=data_robot_pos, target_position=data_target_pos, \
                                people_dict=people_dict)
                            robot_path = np.array(robot_path)
                            dirName = './trajectories_obs/trajectory'+str(k)+'_'+str(n)
                            try:
                                os.mkdir(dirName)
                                print("Directory ", dirName, " Created ")
                            except FileExistsError:
                                print("Directory ", dirName, " already exists")
                            np.save(dirName+'/trj'+str(i)+'.npy', robot_path)
                            print('===== trj'+str(i)+'saved =====')
                            break
                        else:
                            self.pub_action.publish(move_cmd)
                        
                        self.pub_people_action()
                        rate.sleep()

    def pub_people_action(self):
        """
        Publish the people's action
        """
        action_people1 = Twist()
        action_people2 = Twist()
        action_people3 = Twist()
        action_people4 = Twist()
        action_people5 = Twist()
        action_people6 = Twist()
        action_people7 = Twist()
        action_people8 = Twist()
        action_people9 = Twist()
        action_people10 = Twist()
        action_people11 = Twist()
        action_people12 = Twist()
        action_people13 = Twist()
        action_people14 = Twist()
        action_people15 = Twist()
        action_people16 = Twist()
        action_people17 = Twist()
        action_people18 = Twist()
        action_people19 = Twist()
        action_people20 = Twist()
        action_people21 = Twist()
        action_people22 = Twist()
        action_people23 = Twist()
        action_people24 = Twist()
        action_people25 = Twist()

        people_linear_spd = 0.3
        people_angular_spd = 0
        action_people1.linear.x = people_linear_spd
        action_people1.angular.x = people_angular_spd
        action_people2.linear.x = people_linear_spd
        action_people2.angular.x = people_angular_spd
        action_people3.linear.x = people_linear_spd
        action_people3.angular.x = people_angular_spd
        action_people4.linear.x = people_linear_spd
        action_people4.angular.x = people_angular_spd
        action_people5.linear.x = people_linear_spd
        action_people5.angular.x = people_angular_spd
        action_people6.linear.x = people_linear_spd
        action_people6.angular.x = people_angular_spd        
        action_people7.linear.x = people_linear_spd
        action_people7.angular.x = people_angular_spd
        action_people8.linear.x = people_linear_spd
        action_people8.angular.x = people_angular_spd
        action_people9.linear.x = people_linear_spd
        action_people9.angular.x = people_angular_spd
        action_people10.linear.x = people_linear_spd
        action_people10.angular.x = people_angular_spd
        action_people11.linear.x = people_linear_spd
        action_people11.angular.x = people_angular_spd
        action_people12.linear.x = people_linear_spd
        action_people12.angular.x = people_angular_spd
        action_people13.linear.x = people_linear_spd
        action_people13.angular.x = people_angular_spd
        action_people14.linear.x = people_linear_spd
        action_people14.angular.x = people_angular_spd
        action_people15.linear.x = people_linear_spd
        action_people15.angular.x = people_angular_spd
        action_people16.linear.x = people_linear_spd
        action_people16.angular.x = people_angular_spd
        action_people17.linear.x = people_linear_spd
        action_people17.angular.x = people_angular_spd
        action_people18.linear.x = people_linear_spd
        action_people18.angular.x = people_angular_spd
        action_people19.linear.x = people_linear_spd
        action_people19.angular.x = people_angular_spd
        action_people20.linear.x = people_linear_spd
        action_people20.angular.x = people_angular_spd
        action_people21.linear.x = people_linear_spd
        action_people21.angular.x = people_angular_spd    
        action_people22.linear.x = people_linear_spd
        action_people22.angular.x = people_angular_spd
        action_people23.linear.x = people_linear_spd
        action_people23.angular.x = people_angular_spd
        action_people24.linear.x = people_linear_spd
        action_people24.angular.x = people_angular_spd   
        action_people25.linear.x = people_linear_spd
        action_people25.angular.x = people_angular_spd    
        self.pub_action_people1.publish(action_people1)
        self.pub_action_people2.publish(action_people2)
        self.pub_action_people3.publish(action_people3)
        self.pub_action_people4.publish(action_people4)
        self.pub_action_people5.publish(action_people5)
        self.pub_action_people6.publish(action_people6)
        self.pub_action_people7.publish(action_people7)
        self.pub_action_people8.publish(action_people8)
        self.pub_action_people9.publish(action_people9)
        self.pub_action_people10.publish(action_people10)
        self.pub_action_people11.publish(action_people11)
        self.pub_action_people12.publish(action_people12)
        self.pub_action_people13.publish(action_people13)
        self.pub_action_people14.publish(action_people14)
        self.pub_action_people15.publish(action_people15)
        self.pub_action_people16.publish(action_people16)
        self.pub_action_people17.publish(action_people17)
        self.pub_action_people18.publish(action_people18)
        self.pub_action_people19.publish(action_people19)
        self.pub_action_people20.publish(action_people20)
        self.pub_action_people21.publish(action_people21)
        self.pub_action_people22.publish(action_people22)
        self.pub_action_people23.publish(action_people23)
        self.pub_action_people24.publish(action_people24)
        self.pub_action_people25.publish(action_people25)

    def _network_2_robot_action(self, state, h0, linear_spd_max, linear_spd_min):
        """
        Generate robot action based on network output
        :param state: ddpg state
        :return: [linear spd, angular spd]
        """
        with torch.no_grad():
            batch_size = 1
            run_time = 10
            # normal_state_spikes, scan_state_spikes = self._state_2_state_spikes(state, batch_size)
            # normal_state_spikes = torch.Tensor(normal_state_spikes).to(self.device)
            # scan_state_spikes = torch.Tensor(scan_state_spikes).to(self.device)
            # state = ddpg_state_2_spike_value_state(state)
            # print(state)
            # state = self.actor_net.pre_process_input(state)
            # self.actor_net.input(state)
            # self.actor_net.run(run_time)
            # print(self.actor_net.output.predict)
            # # action = (self.actor_net.output.predict/(run_time*10)).to('cpu')
            # action = self.actor_net.output.predict.to('cpu')
            # action = action.numpy().squeeze().tolist()
            # state = np.array(state).reshape((1, -1))
            normal_state_spikes, scan_state_spikes = self._state_2_state_spikes(state, 1)  # {0}, batch x 6 x batch_window {1}, batch x channels x width x batch_window
            #print('shape normal: ', normal_state_spikes.shape)
            #print('shape scan: ', scan_state_spikes.shape)
            normal_state_spikes = torch.Tensor(normal_state_spikes).to(self.device)
            scan_state_spikes = torch.Tensor(scan_state_spikes).to(self.device)
            action = self.actor_net([normal_state_spikes, scan_state_spikes], 1).to(self.device)
            action = action.cpu().numpy().squeeze()
            raw_snn_action = copy.deepcopy(action)
            # action, h0, rates = self.actor_net([normal_state_spikes, scan_state_spikes], h0, 1)

            # action = action.cpu().numpy().squeeze().tolist()
            decode_action = wheeled_network_2_robot_action_decoder(
                action, linear_spd_max, linear_spd_min
            )

        return decode_action, 0, 0

    def _state_2_state_spikes(self, state, batch_size):
        """
        Transform state to spikes of input neurons
        :param state: robot state
        :return: state_spikes
        """
        spike_state_num = 366
        spike_state_value = ddpg_state_2_spike_value_state(state, spike_state_num)
        spike_normal, spike_scan = spike_state_value[0], spike_state_value[1]

        spike_normal = spike_normal.reshape((-1, 6, 1))    # batch_size x normal_state_num x 1
        normal_state_spikes = np.random.rand(batch_size, 6, self.batch_window) < spike_normal
        normal_state_spikes = normal_state_spikes.astype(float)

        # print('shape: ', spike_scan.shape)
        spike_scan = spike_scan.reshape((-1, 1, 360, 1))   # batch_size x channels x width x batch_window
        scan_state_spikes = np.random.rand(batch_size, 1, 360, self.batch_window) < spike_scan
        scan_state_spikes = scan_state_spikes.astype(float)

        return normal_state_spikes, scan_state_spikes

    def _state_2_scale_state(self, state):
        """
        Transform state to scale state with or without Poisson random
        :param state: robot state
        :return: scale_state
        """
        if self.is_poisson:
            scale_state_num = self.scan_half_num * 2 + 6
            state = ddpg_state_2_spike_value_state(state, scale_state_num)
            state = np.array(state)
            spike_state_value = state.reshape((1, scale_state_num, 1))
            state_spikes = np.random.rand(1, scale_state_num, self.batch_window) < spike_state_value
            poisson_state = np.sum(state_spikes, axis=2).reshape((1, -1))
            poisson_state = poisson_state / self.batch_window
            scale_state = poisson_state.astype(float)
        else:
            scale_state_num = self.scan_half_num * 2 + 4
            scale_state = ddpg_state_rescale(state, scale_state_num)
            scale_state = np.array(scale_state).reshape((1, scale_state_num))
            scale_state = scale_state.astype(float)
        return scale_state

    def _near_obstacle(self, pos):
        """
        Test if robot is near obstacle
        :param pos: robot position
        :return: done
        """
        done = False
        robot_point = Point(pos[0], pos[1])
        for poly in self.obstacle_poly_list:
            tmp_dis = robot_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                done = True
                break
        return done

    def _set_new_target(self, robot_position, target_position, people_dict):
        """
        Set new robot pose and goal position
        :param robot_position : the initial position of robot           --> [x, y, yaw]
        :param target_position: the initial position of target robot    --> [x, y]
        """
        target_msg = ModelState()
        target_msg.model_name = 'target'
        target_msg.pose.position.x = target_position[0]
        target_msg.pose.position.y = target_position[1]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(target_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        # self.pub_action.publish(Twist())

        robot_init_pose = robot_position
        robot_init_quat = self._euler_2_quat(yaw=robot_init_pose[2])
        robot_msg = ModelState()
        robot_msg.model_name = 'robot'
        robot_msg.pose.position.x = robot_init_pose[0]
        robot_msg.pose.position.y = robot_init_pose[1]
        robot_msg.pose.orientation.x = robot_init_quat[1]
        robot_msg.pose.orientation.y = robot_init_quat[2]
        robot_msg.pose.orientation.z = robot_init_quat[3]
        robot_msg.pose.orientation.w = robot_init_quat[0]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(robot_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)

        # set the position and orientation of the people
        for k in range(1, 25):
            people_msg = ModelState()
            name = 'people' + str(k)
            people_pos = people_dict[name]
            people_msg.model_name = name + '_' + 'robot'
            people_msg.pose.position.x = people_pos[0]
            people_msg.pose.position.y = people_pos[1]
            people_init_quat = self._euler_2_quat(yaw=people_pos[2])
            people_msg.pose.orientation.x = people_init_quat[1]
            people_msg.pose.orientation.y = people_init_quat[2]
            people_msg.pose.orientation.z = people_init_quat[3]
            people_msg.pose.orientation.w = people_init_quat[0]
            rospy.wait_for_service('gazebo/set_model_state')
            try:
                resp = self.set_model_target(people_msg)
            except rospy.ServiceException as e:
                print("Set Target Service Failed: %s" % e)

        rospy.sleep(0.5)

    def _euler_2_quat(self, yaw=0, pitch=0, roll=0):
        """
        Transform euler angule to quaternion
        :param yaw: z
        :param pitch: y
        :param roll: x
        :return: quaternion
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return [w, x, y, z]


    def _robot_state_cb(self, msg):
        """
        Callback function for robot state
        :param msg: message
        """
        # print('------ robot_state_cb ------')
        if self.robot_state_init is False:
            self.robot_state_init = True
        quat = [msg.pose[-2].orientation.x,
                msg.pose[-2].orientation.y,
                msg.pose[-2].orientation.z,
                msg.pose[-2].orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        linear_spd = math.sqrt(msg.twist[-2].linear.x**2 + msg.twist[-2].linear.y**2)
        # self.robot_pose = [msg.pose[-2].position.x, msg.pose[-2].position.y, yaw]      
        # self.robot_speed = [linear_spd, msg.twist[-2].angular.z]                 


    def _robot_scan_cb(self, msg):
        """
        Callback function for robot scan
        :param msg: message
        """
        if self.robot_scan_init is False:
            self.robot_scan_init = True
        range_size = len(msg.ranges)
        print('range_size:', range_size)
        tmp_robot_scan = np.zeros((1, 360))
        
        for i in range(range_size):
            if msg.ranges[i] == float('inf'):
                tmp_robot_scan[0][i] = 25
            else:
                # self.robot_scan[i] = msg.ranges[i]
                tmp_robot_scan[0][i] = msg.ranges[i]

        self.robot_scan = tmp_robot_scan


    def _robot_spying_cb(self, msg):
        """
        Callback function for spying the state of robot and target
        :param msg: spying signal
        """
        self.goal_dis_dir_cur = [msg.distance, msg.direction]
        # print('goal_dis_dir_cur: ', self.goal_dis_dir_cur)
        self.robot_pose = [msg.x, msg.y]
        self.robot_speed = [msg.linearspd, msg.angularz]  

    def _people_distance_cb(self, msg):
        """
        Callback function for people distance around
        """
        if self.people_distance_init is False:
            self.people_distance_init = True
        self.people_distance = msg.data