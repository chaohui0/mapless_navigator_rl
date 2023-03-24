cd ros/catkin_ws
source devel/setup.bash

nohup roslaunch mbot_gazebo view_test1.launch > bot_train.log &
sleep 25
nohup  rosrun simple_laserscan pub_dis_dir_original.py > /dev/null &

cd ../../training/train_spaic
nohup python -u train_sddpg.py > train_train.log &

#tail -f train.log


 
