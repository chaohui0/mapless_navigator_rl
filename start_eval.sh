cd ros/catkin_ws
source devel/setup.bash

nohup roslaunch mbot_gazebo view_test1.launch > bot_eval.log &
sleep 25
nohup  rosrun simple_laserscan pub_dis_dir_original.py > /dev/null &



#tail -f train.log


 
