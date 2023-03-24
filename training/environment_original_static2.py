import rospy
import math
import copy
import random
import numpy as np
from shapely.geometry import Point
from simple_laserscan.msg import Spying
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge, CvBridgeError
# from dvs_msgs.msg import EventArray, Event
import cv2
from std_msgs.msg import Float32MultiArray


class GazeboEnvironment:
    """
    Class for Gazebo Environment

    Main Function:
        1. Reset: Rest environment at the end of each episode
        and generate new goal position for next episode

        2. Step: Execute new action and return state
     """
    def __init__(self,
                 dvs_dim=(6, 480, 640),
                 obs_near_th=0.35,
                 goal_near_th=0.5,
                 goal_reward=20,
                 obs_reward=-5,
                 goal_dis_min_dis=0.5,
                 goal_dis_scale=1.0,
                 laser_scan_min_dis=0.55,
                 laser_scan_scale=1.0,
                 goal_dis_amp=5,
                 goal_dir_amp=5,
                 step_time=0.1):
        """

        :param laser_scan_half_num: half number of scan points
        :param laser_scan_min_dis: Min laser scan distance
        :param laser_scan_scale: laser scan scale
        :param scan_dir_num: number of directions in laser scan
        :param goal_dis_min_dis: minimal distance of goal distance
        :param goal_dis_scale: goal distance scale
        :param obs_near_th: Threshold for near an obstacle
        :param goal_near_th: Threshold for near an goal
        :param goal_reward: reward for reaching goal
        :param obs_reward: reward for reaching obstacle
        :param goal_dis_amp: amplifier for goal distance change
        :param step_time: time for a single step (DEFAULT: 0.1 seconds)
        """
        self.goal_pos_list = None
        self.goal_dis_amp = 30
        self.goal_dir_amp = 15
        self.obstacle_poly_list = None
        self.robot_init_pose_list = None
        self.people_list = None
        self.obs_near_th = obs_near_th
        self.goal_near_th = goal_near_th
        self.goal_reward = goal_reward
        self.obs_reward = obs_reward
        self.step_time = step_time
        self.goal_far_th = 8.5
        self.people_near_reward = -5
        self.dir_th = 1.7
        self.goal_dis_min_dis = goal_dis_min_dis  
        self.goal_dis_scale = goal_dis_scale
        self.laser_scan_min_dis = laser_scan_min_dis
        self.laser_scan_scale = laser_scan_scale
        # cv_bridge class
        self.cv_bridge = CvBridge()
        self.cv_norm = np.empty((480, 640), dtype=np.float32)
        # Robot State
        self.robot_pose = [0., 0., 0.]
        self.robot_speed = [0., 0.]
        self.robot_scan = np.zeros(((1,360)))
        self.people_distance = None
        ##self.robot_scan = np.zeros(self.scan_dir_num)
        self.events_cubic = np.zeros(dvs_dim)
        self.robot_state_init = False
        self.robot_scan_init = False
        self.robot_spying_init = False
        self.robot_depth_init = False
        self.people_distance_init = False
        # Goal Position
        self.goal_position = [0., 0.]
        self.goal_dis_dir_pre = [0., 0.]
        self.goal_dis_dir_cur = [0., 0.]
        # speed range
        self.linear_spd_range = 0.5
        self.angular_spd_range = 2.0
        # recommended vel from DWA
        self.recommended_vel_lin = 0
        self.recommended_vel_ang = 0
        # bound of the map
        self.bound_low_x = 2
        self.bound_up_x = 14
        self.bound_low_y = 2
        self.bound_up_y = 14

        # Subscriber
        #print('pp')
        # rospy.Subscriber('/gazebo/model_states', ModelStates, self._robot_state_cb)
        # rospy.Subscriber('/mybot/camera1/events', EventArray, self._robot_dvs_cb)
        rospy.Subscriber('/scan', LaserScan, self._robot_scan_cb)
        rospy.Subscriber('/Spying_signal', Spying, self._robot_spying_cb)
        rospy.Subscriber('/around_people', Float32MultiArray, self._people_distance_cb)
        rospy.Subscriber('/recommended_cmd_vel', Twist, self._recommend_vel_cb)
        #print('hh')
        # Publisher
        self.pub_action = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        # self.pub_action_target = rospy.Publisher('/target_robot/cmd_vel', Twist, queue_size=5)
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
        # Service
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # Init Subscriber
        while not self.robot_spying_init:
            print('loop_spying')
            continue
        while not self.robot_scan_init:
            print('loop_scan')
            continue
        # while not self.robot_depth_init:
        #     print('loop2')
        #     continue
        rospy.loginfo("Finish Subscriber Init...")

    def step(self, action):
        """
        Step Function for the Environment

        Take a action for the robot and return the updated state
        :param action: action taken
        :return: state, reward, done
        """
        assert self.goal_pos_list is not None
        assert self.obstacle_poly_list is not None
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)

        '''
        First move the target robot
        '''

        # move_target = Twist()
        # move_target.linear.x = np.random.uniform(0, 0.7)
        # move_target.angular.z = np.random.uniform(-2, 2)
        # move_target.linear.x = 0
        # move_target.angular.z = 0

        '''
        Then move the around people
        '''
        move_people1 = Twist()
        people_linear_spd = 0
        people_angular_spd = 0
        people_linear_spd1 = 0
        people_angular_spd1 = 0
        move_people1.linear.x = people_linear_spd
        move_people1.angular.z = people_angular_spd

        move_people2 = Twist()
        move_people2.linear.x = people_linear_spd
        move_people2.angular.z = people_angular_spd

        move_people3 = Twist()
        move_people3.linear.x = people_linear_spd
        move_people3.angular.z = people_angular_spd

        move_people4 = Twist()
        move_people4.linear.x = people_linear_spd
        move_people4.angular.z = people_angular_spd

        move_people5 = Twist()
        move_people5.linear.x = people_linear_spd
        move_people5.angular.z = people_angular_spd

        move_people6 = Twist()
        move_people6.linear.x = people_linear_spd
        move_people6.angular.z = people_angular_spd

        move_people7 = Twist()
        move_people7.linear.x = people_linear_spd
        move_people7.angular.z = people_angular_spd

        move_people8 = Twist()
        move_people8.linear.x = people_linear_spd
        move_people8.angular.z = people_angular_spd

        move_people9 = Twist()
        move_people9.linear.x = people_linear_spd
        move_people9.angular.z = people_angular_spd

        move_people10 = Twist()
        move_people10.linear.x = people_linear_spd
        move_people10.angular.z = people_angular_spd

        move_people11 = Twist()
        move_people11.linear.x = people_linear_spd1
        move_people11.angular.z = people_angular_spd

        move_people12 = Twist()
        move_people12.linear.x = people_linear_spd1
        move_people12.angular.z = people_angular_spd

        move_people13 = Twist()
        move_people13.linear.x = people_linear_spd1
        move_people13.angular.z = people_angular_spd

        move_people14 = Twist()
        move_people14.linear.x = people_linear_spd1
        move_people14.angular.z = people_angular_spd

        move_people15 = Twist()
        move_people15.linear.x = people_linear_spd1
        move_people15.angular.z = people_angular_spd

        move_people16 = Twist()
        move_people16.linear.x = people_linear_spd1
        move_people16.angular.z = people_angular_spd

        move_people17 = Twist()
        move_people17.linear.x = people_linear_spd1
        move_people17.angular.z = people_angular_spd

        move_people18 = Twist()
        move_people18.linear.x = people_linear_spd1
        move_people18.angular.z = people_angular_spd

        move_people19 = Twist()
        move_people19.linear.x = people_linear_spd1
        move_people19.angular.z = people_angular_spd

        move_people20 = Twist()
        move_people20.linear.x = people_linear_spd1
        move_people20.angular.z = people_angular_spd

        '''
        Second give action to robot and let robot execute and get next state
        '''
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.angular.z = action[1]

        # self.pub_action_target.publish(move_target)
        self.pub_action.publish(move_cmd)
        self.pub_action_people1.publish(move_people1)
        self.pub_action_people2.publish(move_people2)
        self.pub_action_people3.publish(move_people3)
        self.pub_action_people4.publish(move_people4)
        self.pub_action_people5.publish(move_people5)
        self.pub_action_people6.publish(move_people6)
        self.pub_action_people7.publish(move_people7)
        self.pub_action_people8.publish(move_people8)
        self.pub_action_people9.publish(move_people9)
        self.pub_action_people10.publish(move_people10)
        self.pub_action_people11.publish(move_people11)
        self.pub_action_people12.publish(move_people12)
        self.pub_action_people13.publish(move_people13)
        self.pub_action_people14.publish(move_people14)
        self.pub_action_people15.publish(move_people15)
        self.pub_action_people16.publish(move_people16)
        self.pub_action_people17.publish(move_people17)
        self.pub_action_people18.publish(move_people18)
        self.pub_action_people19.publish(move_people19)
        self.pub_action_people20.publish(move_people20)

        rospy.sleep(self.step_time)
        next_rob_state = self._get_next_robot_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        '''
        Then stop the simulation
        1. Transform Robot State to DDPG State
        2. Compute Reward of the action
        3. Compute if the episode is ended
        '''
        state = self._robot_state_2_ddpg_state(next_rob_state)  # [np.array(1x4), np.array(1x480x640)]
        reward, done = self._compute_reward(next_rob_state)
        self.goal_dis_dir_pre = [self.goal_dis_dir_cur[0], self.goal_dis_dir_cur[1]]
        return state, reward, done

    def reset(self, ita):
        """
        Reset Function to reset simulation at start of each episode

        Return the initial state after reset
        :param ita: number of route to reset to
        :return: state
        """
        assert self.goal_pos_list is not None
        assert self.obstacle_poly_list is not None
        assert self.robot_init_pose_list is not None
        assert self.people_list is not None
        ita = ita % 20000
        assert ita < len(self.goal_pos_list)
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        First choose new goal position and set target model to goal
        '''
        self.goal_position = self.goal_pos_list[ita]
        target_init_quat = self._euler_2_quat(yaw=self.goal_position[2])
        target_msg = ModelState()
        target_msg.model_name = 'target'
        target_msg.pose.position.x = self.goal_position[0]
        target_msg.pose.position.y = self.goal_position[1]
        # target_msg.pose.orientation.x = target_init_quat[1]
        # target_msg.pose.orientation.y = target_init_quat[2]
        # target_msg.pose.orientation.z = target_init_quat[3]
        # target_msg.pose.orientation.w = target_init_quat[0]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(target_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        '''
        Then reset robot state and get initial state
        '''
        self.pub_action.publish(Twist())
        robot_init_pose = self.robot_init_pose_list[ita]
        robot_init_quat = self._euler_2_quat(yaw=robot_init_pose[2])
        robot_msg = ModelState()
        # robot_msg.model_name = 'mobile_base'
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

        people_init_pose = self.people_list[ita]   # np: 20 x 3
        people_num = people_init_pose.shape[0]     # 20
        # print('shape: ', people_num)

        people_position_x = [5, 8, 11, 5, 8, 11]
        people_position_y = [6, 6, 6, 10.5, 10.5, 10.5]

        for i in range(1, 7):
            people_name = 'people' + str(i) + '_robot'
            people_yaw = people_init_pose[i-1][2]
            people_init_quat = self._euler_2_quat(people_yaw)
            people_msg = ModelState()

            people_msg.model_name = people_name
            people_msg.pose.position.x = people_position_x[i-1]
            people_msg.pose.position.y = people_position_y[i-1]
            people_msg.pose.orientation.x = people_init_quat[1]
            people_msg.pose.orientation.y = people_init_quat[2]
            people_msg.pose.orientation.z = people_init_quat[3]
            people_msg.pose.orientation.w = people_init_quat[0]
            rospy.wait_for_service('gazebo/set_model_state')
            try:
                resp = self.set_model_target(people_msg)
            except rospy.ServiceException as e:
                print("Set People Service Failed: %s" % e)


        rospy.sleep(0.5)
        '''
        Transfer the initial robot state to the state for the DDPG Agent
        '''
        rob_state = self._get_next_robot_state()       # [tmp_robot_pose, tmp_robot_spd, tmp_robot_img]
                                                       # :param tmp_robot_pose: [x, y, yaw]
                                                       # :param tmp_robot_spd : [sqrt(linear.x**2 + linear.y**2), angular.z]
                                                       # :param tmp_robot_scan : np.array(5x360)
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        state = self._robot_state_2_ddpg_state(rob_state)
        return state  # [list(4), np.array(360)] 

    def set_new_environment(self, init_pose_list, goal_list, obstacle_list, people_list):
        """
        Set New Environment for training
        :param init_pose_list: init pose list of robot
        :param goal_list: goal position list
        :param obstacle_list: obstacle list
        """

        self.robot_init_pose_list = init_pose_list  # 30000 x 3
        self.goal_pos_list = goal_list              # 30000 x 3
        self.obstacle_poly_list = obstacle_list
        self.people_list = people_list              # 30000 x 3

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

    def _get_next_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time

        State will be: [robot_pose, robot_spd, scan]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_spd = copy.deepcopy(self.robot_speed)
        tmp_robot_scan = copy.deepcopy(self.robot_scan)   # np: 1x360
        state = [tmp_robot_pose, tmp_robot_spd, tmp_robot_scan]
        """
        tmp_robot_pose : [x, y, yaw]
        tmp_robot_spd  : [sqrt(linear.x**2 + linear.y**2), angular.z]
        tmp_robot_scan : np.array(): (5x360)
        """
        return state
    
    def _scaled_position(self, x, y, low_x, up_x, low_y, up_y):
        """
        Scale the position x, y in [0, 1]
        """
        scaled_x = (x - low_x) / (up_x - low_x)
        scaled_y = (y - low_y) / (up_y - low_y)

        return scaled_x, scaled_y

    def _robot_state_2_ddpg_state(self, state):
        """
        Transform robot state to DDPG state
        Robot State: [robot_pose, robot_spd, robot_scan]
        DDPG state:  [Distance to goal, Direction to goal, Linear Spd, Angular Spd, depth_img]
        :param state: robot state
        :return: ddpg_state
        """

        tmp_goal_dis = self.goal_dis_dir_cur[0]
        if tmp_goal_dis == 0:
            tmp_goal_dis = self.goal_dis_scale
        else:
            tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis
            if tmp_goal_dis > 1:
                tmp_goal_dis = 1
            tmp_goal_dis = tmp_goal_dis * self.goal_dis_scale
        ddpg_state = []
        # scaled_position_x, scaled_position_y = self._scaled_position(state[0][0], state[0][1], \
        #                                                              self.bound_low_x, self.bound_up_x, self.bound_low_y, self.bound_up_y)  
        ddpg_state.append(np.array([self.goal_dis_dir_cur[1], tmp_goal_dis, state[1][0], state[1][1]]))
        '''
        Transform distance in laser scan to [0, scale]
        '''
        # print('laser scan {0}, {1}', np.min(state[2]), np.max(state[2]))
        tmp_laser_scan = self.laser_scan_scale * (self.laser_scan_min_dis / state[2])
        tmp_laser_scan = np.clip(tmp_laser_scan, 0, self.laser_scan_scale)
        #print('---scan min: ', np.min(tmp_laser_scan))
        #print('---scan max: ', np.max(tmp_laser_scan))
        ddpg_state.append(tmp_laser_scan)
        #ddpg_state.append(tmp_laser_scan)

        return ddpg_state   # list(np.array(4), np.array(5x360))


    def _compute_reward(self, state):   # state: robot state
        """
        Compute Reward of the action base on current robot state and last step goal distance and direction

        Reward:
            1. R_Arrive If Distance to Goal is smaller than D_goal
            2. R_Collision If Distance to Obstacle is smaller than D_obs
            3. a * (Last step distance to goal - current step distance to goal)

        If robot near obstacle then done
        :param state: DDPG state
        :return: reward, done
        """
        done = False
        '''
        First compute distance to all obstacles
        '''
        near_obstacle = False
        robot_point = Point(state[0][0], state[0][1])
        for poly in self.obstacle_poly_list:
            tmp_dis = robot_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                near_obstacle = True
                break
        '''
        Assign Rewards
        '''

        if self.goal_dis_dir_cur[0] < self.goal_near_th:
            reward = self.goal_reward
            done = True
        elif near_obstacle:
            reward = self.obs_reward
            done = True
        elif min(self.people_distance) < 0.54:
            reward = self.people_near_reward
            # reward = 0
            done = True
        # elif abs(self.goal_dis_dir_cur[1]) > self.dir_th:
        #     reward = self.obs_reward
        #     done = True
        # elif self.goal_dis_dir_cur[0] > self.goal_far_th:
        #     # reward = self.obs_reward
        #     reward = 0
        #     done = True
        else:
            reward = self.goal_dis_amp * max((self.goal_dis_dir_pre[0] - self.goal_dis_dir_cur[0]), -(1/50))
            # reward = reward + self.goal_dir_amp * (abs(self.goal_dis_dir_pre[1]) - abs(self.goal_dis_dir_cur[1]))

        return reward, done

    def _recommend_vel_cb(self, msg):
        """
        Callback function for recommended vel from DWA
        """
        self.recommended_vel_lin = msg.linear.x
        self.recommended_vel_ang = msg.angular.z


    def _robot_spying_cb(self, msg):
        """
        Callback function for spying the state of robot and target
        :param msg: spying signal
        """
        if self.robot_spying_init is False:
            self.robot_spying_init = True
        self.goal_dis_dir_cur = [msg.distance, msg.direction]
        self.robot_pose = [msg.x, msg.y]
        self.robot_speed = [msg.linearspd, msg.angularz]

        # print('goal_dis_dir: ',self.goal_dis_dir_cur)

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
        self.robot_pose = [msg.pose[-2].position.x, msg.pose[-2].position.y, yaw]      # 分别是 X, Y, rotation by Z
        self.robot_speed = [linear_spd, msg.twist[-2].angular.z]                       # [sqrt(linear.x**2, linear.y**2), angular.z]


    def _robot_scan_cb(self, msg):
        """
        Callback function for robot scan
        :param msg: message
        """
        if self.robot_scan_init is False:
            self.robot_scan_init = True
        range_size = len(msg.ranges)
        tmp_robot_scan = np.zeros((1, 360))
        
        for i in range(range_size):
            if msg.ranges[i] == float('inf'):
                tmp_robot_scan[0][i] = 25
            else:
                tmp_robot_scan[0][i] = msg.ranges[i]
        # self.robot_scan[0:4] = self.robot_scan[1:5]
        # self.robot_scan[4] = tmp_robot_scan
        self.robot_scan = tmp_robot_scan


    def _people_distance_cb(self, msg):
        """
        Callback function for people distance around
        """
        if self.people_distance_init is False:
            self.people_distance_init = True
        self.people_distance = msg.data
        # print('distance: ', self.people_distance)


def robot_state_cb(msg):
    """
    Callback function for robot state
    :param msg: message
    """
    # print('------ robot_state_cb ------')
    i = -21
    quat = [msg.pose[i].orientation.x,
            msg.pose[i].orientation.y,
            msg.pose[i].orientation.z,
            msg.pose[i].orientation.w]
    siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
    cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    linear_spd = math.sqrt(msg.twist[i].linear.x**2 + msg.twist[i].linear.y**2)
    robot_pose = [msg.pose[i].position.x, msg.pose[i].position.y, yaw]      # 分别是 X, Y, rotation by Z
    robot_speed = [linear_spd, msg.twist[i].angular.z] 
    print('robot_pose: ', robot_pose)

if __name__ == "__main__":

    rospy.init_node('test_node')
    rospy.Subscriber('/gazebo/model_states', ModelStates, robot_state_cb)
    rospy.spin()