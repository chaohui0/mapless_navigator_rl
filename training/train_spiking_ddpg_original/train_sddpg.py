from typing import OrderedDict
import rospy
import time
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import sys
import torch
import numpy as np

sys.path.append('../../')
from training.train_spiking_ddpg_original.sddpg_agent import AgentSpiking
from training.environment_original import GazeboEnvironment
from training.utility_original import *
#from training.train_spiking_ddpg.utility import Trained_CNN


def train_sddpg(run_name="SNN_R1", exp_name="Rand_R1", episode_num=(100000, 1600, 1700, 1800),
                iteration_num_start=(100000, 1600, 1700, 1800), iteration_num_step=(1, 2, 3, 4),
                iteration_num_max=(100000, 1600, 1700, 1800),
                linear_spd_max=1.6, linear_spd_min=0.01, save_steps=4000,
                env_epsilon=(0.9, 0.6, 0.6, 0.6), env_epsilon_decay=(0.999, 0.9999, 0.9999, 0.9999),
                goal_dis_min_dis=0.3,
                obs_reward=-20, goal_reward=30, goal_dis_amp=15, goal_th=0.5, obs_th=0.35,
                state_num=364, action_num=2, spike_state_num=366, batch_window=50, actor_lr=1e-5,
                memory_size=100000, batch_size=300, epsilon_end=0.1, rand_start=10000, rand_decay=0.999,
                rand_step=2, target_tau=0.01, target_step=1, use_cuda=True, use_trained=False):
    """
    Training Spiking DDPG for Mapless Navigation

    :param run_name: Name for training run
    :param exp_name: Name for experiment run to get random positions
    :param episode_num: number of episodes for each of of the 4 environments
    :param iteration_num_start: start number of maximum steps for 4 environments
    :param iteration_num_step: increase step of maximum steps after each episode
    :param iteration_num_max: max number of maximum steps for 4 environments
    :param linear_spd_max: max wheel speed
    :param linear_spd_min: min wheel speed
    :param save_steps: number of steps to save model
    :param env_epsilon: start epsilon of random action for 4 environments
    :param env_epsilon_decay: decay of epsilon for 4 environments
    :param laser_half_num: half number of scan points
    :param laser_min_dis: min laser scan distance
    :param scan_overall_num: overall number of scan points
    :param goal_dis_min_dis: minimal distance of goal distance
    :param obs_reward: reward for reaching obstacle
    :param goal_reward: reward for reaching goal
    :param goal_dis_amp: amplifier for goal distance change
    :param goal_th: threshold for near a goal
    :param obs_th: threshold for near an obstacle
    :param state_num: number of state
    :param action_num: number of action
    :param spike_state_num: number of state for spike action
    :param batch_window: inference timesteps
    :param actor_lr: learning rate for actor network
    :param memory_size: size of memory
    :param batch_size: batch size
    :param epsilon_end: min value for random action
    :param rand_start: max value for random action
    :param rand_decay: steps from max to min random action
    :param rand_step: steps to change
    :param target_tau: update rate for target network
    :param target_step: number of steps between each target update
    :param use_cuda: if true use gpu
    """
    # Create Folder to save weights
    torch.manual_seed(6)
    np.random.seed(6)
    dirName = 'save_sddpg_weights'
    try:
        os.mkdir('../' + dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    # Define 4 training environments
    env1_poly_list, env1_raw_poly_list, env1_goal_list, env1_init_list = gen_rand_list_env1(episode_num[0])
    env2_poly_list, env2_raw_poly_list, env2_goal_list, env2_init_list = gen_rand_list_env2(episode_num[1])
    env3_poly_list, env3_raw_poly_list, env3_goal_list, env3_init_list = gen_rand_list_env3(episode_num[2])
    env4_poly_list, env4_raw_poly_list, env4_goal_list, env4_init_list = gen_rand_list_env4(episode_num[3])
    overall_poly_list = [env1_poly_list, env2_poly_list, env3_poly_list, env4_poly_list]

    # Read Random Start Pose and Goal Position based on experiment name
    # overall_list = pickle.load(open("../random_positions/" + exp_name + ".p", "rb"))
    # overall_init_list = overall_list[0]
    # overall_goal_list = overall_list[1]

    overall_init_list = np.load("../random_positions/robot_init_position_original1.6.npy")
    overall_goal_list = np.load("../random_positions/target_init_position_original1.6.npy")
    people_list = np.load("../random_positions/people_position20.npy")              # np: 30000 x 10 x 3
    print('init shape :', overall_init_list.shape)
    print('goal shape :', overall_goal_list.shape)

    print("Use Training Rand Start and Goal Positions: ", exp_name)

    # Define Environment and Agent Object for training
    rospy.init_node("train_sddpg")
    print('Here1')
    env = GazeboEnvironment(obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp,
                            goal_near_th=goal_th, obs_near_th=obs_th)
    print('Here2')
    agent = AgentSpiking(state_num, action_num, spike_state_num,
                         batch_window=batch_window, actor_lr=actor_lr,
                         memory_size=memory_size, batch_size=batch_size, epsilon_end=epsilon_end,
                         epsilon_rand_decay_start=rand_start, epsilon_decay=rand_decay,
                         epsilon_rand_decay_step=rand_step,
                         target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda, use_trained=use_trained)
    print('Here3')
    # Define Tensorboard Writer
    tb_writer = SummaryWriter()

    # Define maximum steps per episode and reset maximum random action
    overall_steps = 0
    overall_episode = 0
    env_episode = 0
    env_ita = 0
    #
    dwa_decay_step = 100
    dwa_episilon = 0.6
    dwa_episilon_decay = 0.99
    random_data = 0.6
    random_decay = 0.99
    random_num_decay = 20
    dwa_flag = 1
    #

    ita_per_episode = iteration_num_start[env_ita]            # iteration_num_start=(5000, 1600, 1700, 1800)
    env.set_new_environment(overall_init_list[env_ita],
                            overall_goal_list[env_ita],
                            overall_poly_list[env_ita],
                            people_list)
    agent.reset_epsilon(env_epsilon[env_ita],
                        env_epsilon_decay[env_ita])

    # load the trained CNN
    device = torch.device('cuda')

    # Start Training
    start_time = time.time()
    flag = 0
    pre_real_vel = env.robot_speed
    cur_real_vel = env.robot_speed

    print('start to train!')
    while True:
        state = env.reset(env_episode)  # list(np.array(4), np.array(5x360))
        # print('state size: ', state[0].shape)
        cur_real_vel = env.robot_speed  # update current speed
        spike_state_value = ddpg_state_2_spike_value_state(state, spike_state_num)  # list(np.array(6), np.array(1x360))
        
        random_num = np.random.binomial(1, random_data, 1)
        if overall_episode % random_num_decay == 0:
            random_data = random_data * random_decay
            dwa_episilon = dwa_episilon * dwa_episilon_decay

        random_num[0] = 0
        if random_num[0] == 1:
            dwa_flag = 1
            for dwa_ita in range(200):
                overall_steps += 1
                dwa_vel = [env.recommended_vel_lin, env.recommended_vel_ang]
                dwa_vel = np.array(dwa_vel)
                raw_action = wheeled_dwa_encoder(dwa_vel, linear_spd_max, linear_spd_min)

                # Add some noise to DWA
                raw_action = np.array(raw_action)
                noise = np.random.rand(2) * dwa_episilon
                raw_action = noise + (1 - dwa_episilon) * raw_action
                raw_action = np.clip(raw_action, [0., 0.], [1., 1.])
                decode_action = wheeled_network_2_robot_action_decoder(
                    raw_action.tolist(), linear_spd_max, linear_spd_min
                )

                # next_state, reward, done = env.step(dwa_vel)
                next_state, reward, done = env.step(decode_action)
                spike_next_value = ddpg_state_2_spike_value_state(next_state)
                agent.remember(state, spike_state_value, raw_action, reward, next_state, spike_next_value, done)
                state = next_state
                spike_state_value = spike_next_value
                if len(agent.memory) > batch_size:
                    actor_loss_value, critic_loss_value = agent.replay()
                if done:
                    break
        
        else:
            dwa_flag = 0
            episode_reward = 0
            for ita in range(ita_per_episode):  # 5000
                flag = flag + 1
                ita_time_start = time.time()
                overall_steps += 1

                # dwa_vel = [env.recommended_vel_lin, env.recommended_vel_ang]
                raw_action, raw_snn_action = agent.act(spike_state_value)
                decode_action = wheeled_network_2_robot_action_decoder(
                    raw_action, linear_spd_max, linear_spd_min
                )
                lin_min, lin_max, ang_min, ang_max = compute_current_achivable_vel(pre_real_vel, 1, 0.69, 0.3)
                next_state, reward, done = env.step(decode_action)
                spike_nstate_value = ddpg_state_2_spike_value_state(next_state)
                episode_reward += reward
                print("raw_action:", raw_action, "raw_snn_action:",raw_snn_action, "decode_action:", decode_action, " reward:",reward)
                # Train network with replay
                # print('agent memory: ',len(agent.memory))
                agent.remember(state, spike_state_value, raw_action, reward, next_state, spike_nstate_value, done)
                state = next_state
                spike_state_value = spike_nstate_value
                if len(agent.memory) > batch_size:
                    actor_loss_value, critic_loss_value = agent.replay()
                    tb_writer.add_scalar('Spike-DDPG/actor_loss', actor_loss_value, overall_steps)
                    tb_writer.add_scalar('Spike-DDPG/critic_loss', critic_loss_value, overall_steps)
                    # print('actor_loss: {0}, critic_loss: {1}', actor_loss_value, critic_loss_value)
                ita_time_end = time.time()
                tb_writer.add_scalar('Spike-DDPG/ita_time', ita_time_end - ita_time_start, overall_steps)
                tb_writer.add_scalar('Spike-DDPG/action_epsilon', agent.epsilon, overall_steps)
                tb_writer.add_scalar('Spike-DDPG/raw_left_wheel_output', raw_snn_action[0], overall_steps)
                tb_writer.add_scalar('Spike-DDPG/raw_right_wheel_output', raw_snn_action[1], overall_steps)
                # tb_writer.add_scalars('Achivable_linear_vel', {'lin_min':lin_min, 'lin_max':lin_max, 'real_lin':cur_real_vel[0]}, overall_steps)
                # tb_writer.add_scalars('Achivable_angular_vel', {'ang_min':ang_min, 'ang_max':ang_max, 'real_ang':cur_real_vel[1]}, overall_steps)

                # Save Model
                if overall_steps % save_steps == 0:
                    # max_w, min_w, max_b, min_b, shape_w, shape_b = agent.save("../save_sddpg_weights",
                    #                                                           overall_steps // save_steps, run_name)
                    agent.save("../save_sddpg_weights_original_2", overall_steps // save_steps, run_name)
                    print('======== save the weights ========')
                    # print("Max weights of SNN each layer: ", max_w)
                    # print("Min weights of SNN each layer: ", min_w)
                    # print("Shape of weights: ", shape_w)
                    # print("Max bias of SNN each layer: ", max_b)
                    # print("Min bias of SNN each layer: ", min_b)
                    # print("Shape of bias: ", shape_b)

                # If Done then break
                if done or ita == 400 - 1:
                    print("Episode: {}/{}, Avg Reward: {}, Steps: {}"
                        .format(overall_episode, episode_num, episode_reward / (ita + 1), ita + 1))
                    tb_writer.add_scalar('Spike-DDPG/avg_reward', episode_reward / (ita + 1), overall_steps)
                    break
        # cur_real_vel = pre_real_vel
        if ita_per_episode < iteration_num_max[env_ita]:    # episode_num=(5000, 1600, 1700, 1800)
            ita_per_episode += iteration_num_step[env_ita]
        if overall_episode == 99999:
            # max_w, min_w, max_b, min_b, shape_w, shape_b = agent.save("../save_sddpg1_weights",
            #                                                           0, run_name)
            agent.save("../save_sddpg_weights_original_1", 0, run_name)
            print('======== save the weights ========')
            # print("Max weights of SNN each layer: ", max_w)
            # print("Min weights of SNN each layer: ", min_w)
            # print("Shape of weights: ", shape_w)
            # print("Max bias of SNN each layer: ", max_b)
            # print("Min bias of SNN each layer: ", min_b)
            # print("Shape of bias: ", shape_b)
            break
        overall_episode += 1
        if dwa_flag == 0:
            tb_writer.add_scalar('follow_steps', ita, overall_episode)
            
        env_episode += 1
        if env_episode == episode_num[env_ita]:  # episode_num=(5000, 1600, 1700, 1800)
            print("Environment ", env_ita, " Training Finished ...")
            if env_ita == 3:
                break
            env_ita += 1
            env.set_new_environment(overall_init_list[env_ita],
                                    overall_goal_list[env_ita],
                                    overall_poly_list[env_ita])
            agent.reset_epsilon(env_epsilon[env_ita],
                                env_epsilon_decay[env_ita])
            ita_per_episode = iteration_num_start[env_ita]
            env_episode = 0
    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--step', type=int, default=50)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    train_sddpg(use_cuda=USE_CUDA, batch_window=args.step, use_trained=False)
