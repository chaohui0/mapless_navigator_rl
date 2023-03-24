from collections import deque
import pickle
import copy
import os
from pickletools import optimize
import random
import math
from typing import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from training.train_spaic.spaic_networks import ActorNetSpiking, CriticNetSpiking


class AgentSpiking:
    """
    Class for DDPG Agent for Spiking Actor Network

    Main Function:
        1. Remember: Insert new memory into the memory list

        2. Act: Generate New Action base on actor network

        3. Replay: Train networks base on mini-batch replay

        4. Save: Save model
    """
    def __init__(self,
                 state_num=4,
                 action_num=2,
                 spike_normal_state_num=6,
                 spike_scan_state_num=360,
                 actor_net_dim=(256, 256, 256),
                 critic_net_dim=(512, 512, 512),
                 batch_window=50,
                 memory_size=1000,
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 actor_lr=0.0001,
                 critic_lr=0.0001,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_rand_decay_start=60000,
                 epsilon_rand_decay_step=1,
                 use_cuda=True,
                 use_trained=False,
                 run_time=10):
        """

        :param state_num: number of state
        :param action_num: number of action
        :param spike_state_num: number of state for spike actor
        :param actor_net_dim: dimension of actor network
        :param critic_net_dim: dimension of critic network
        :param batch_window: window steps for one sample
        :param memory_size: size of memory
        :param batch_size: size of mini-batch
        :param target_tau: update rate for target network
        :param target_update_steps: update steps for target network
        :param reward_gamma: decay of future reward
        :param actor_lr: learning rate for actor network
        :param critic_lr: learning rate for critic network
        :param epsilon_start: max value for random action
        :param epsilon_end: min value for random action
        :param epsilon_decay: steps from max to min random action
        :param epsilon_rand_decay_start: start step for epsilon start to decay
        :param epsilon_rand_decay_step: steps between epsilon decay
        :param use_cuda: if or not use gpu
        """
        self.pcnt=0
        self.spike_state_num = 366
        self.state_num = state_num
        self.action_num = action_num
        self.spike_normal_state_num = spike_normal_state_num
        self.spike_scan_state_num = spike_scan_state_num
        self.batch_window = batch_window
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.reward_gamma = reward_gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_rand_decay_start = epsilon_rand_decay_start
        self.epsilon_rand_decay_step = epsilon_rand_decay_step
        self.use_cuda = use_cuda
        self.use_trained = use_trained
        self.run_time = 10

        self.save_name_critic = '../save_sddpg_weights/SNN_R1SNN_critic_weights_500.pth'
        '''
        Random Action
        '''
        self.epsilon = epsilon_start
        '''
        Device
        '''
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        """
        Memory
        """
        self.memory = deque(maxlen=self.memory_size)
        """
        Networks and Target Networks
        """
        if self.use_trained:
            self.actor_net = ActorNetSpiking()
            self.actor_net.build()
            self.actor_net.state_from_dict(filename="actor_net", direct="../save_sddpg_weights", device="cuda")

            self.target_actor_net = ActorNetSpiking()
            self.target_actor_net.build()
            self.target_actor_net.state_from_dict(filename="actor_net", direct="../save_sddpg_weights", device="cuda")


            self.critic_net = CriticNetSpiking(self.state_num, self.action_num,
                                                hidden1=critic_net_dim[0],
                                                hidden2=critic_net_dim[1],
                                                hidden3=critic_net_dim[2])
            
            self.target_critic_net = CriticNetSpiking(self.state_num, self.action_num,
                                                        hidden1=critic_net_dim[0],
                                                        hidden2=critic_net_dim[1],
                                                        hidden3=critic_net_dim[2])

            self.critic_net = self.critic_net.cuda()
            self.target_critic_net = self.target_critic_net.cuda()

            self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)

            model_data_critic = torch.load(self.save_name_critic)

            self.critic_net.load_state_dict(model_data_critic['model_dict'])
            self.critic_optimizer.load_state_dict(model_data_critic['optimizer_dict'])

            self.target_critic_net.load_state_dict(model_data_critic['model_dict'])
            
            
        else:
            self.actor_net = ActorNetSpiking()
            self.actor_net.build()
            self.target_actor_net = ActorNetSpiking()

            self.critic_net = CriticNetSpiking(self.state_num, self.action_num,
                                                hidden1=critic_net_dim[0],
                                                hidden2=critic_net_dim[1],
                                                hidden3=critic_net_dim[2])
            
            self.target_critic_net = CriticNetSpiking(self.state_num, self.action_num,
                                                        hidden1=critic_net_dim[0],
                                                        hidden2=critic_net_dim[1],
                                                        hidden3=critic_net_dim[2])


            # self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)

        self._hard_update_snnflow(self.target_actor_net, self.actor_net)
        self._hard_update(self.target_critic_net, self.critic_net)
        self.critic_net.to(self.device)
        self.target_critic_net.to(self.device)
        """
        Criterion and optimizers
        """
        self.criterion = nn.MSELoss()

        """
        Step Counter
        """
        self.step_ita = 0


    def remember(self, state, spike_state, action, reward, next_state, next_spike_state, done):
        """
        Add New Memory Entry into memory deque
        :param state: current state
        :param spike_state: current state with separate neg and pos values
        :param action: current action
        :param reward: reward after action
        :param next_state: next state
        :param next_spike_state: next with separate neg and pos values
        :param done: if is done
        """
        self.memory.append((state, spike_state, action, reward, next_state, next_spike_state, done))


    def act(self, state, h_0, explore=True, train=True, ita=0):  # state:  list(np.array(6), np.array(1x360))
        """
        Generate Action based on state
        :param state: current state
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: action
        """
        # normal_state = state[0]   # np.array(6)
        # scan_state   = state[1]   # np.array(5x360)
        with torch.no_grad():
            state = self.actor_net.pre_process_input(state)
            self.actor_net.input(state)
            self.actor_net.run(self.run_time)
            action = self.actor_net.output.predict.to('cpu')
            action = action.numpy().squeeze()
            raw_snn_action = copy.deepcopy(action)
            # self.actor_net.show_monitor()

        if  train:
            # if self.step_ita < 1000:
            #     seed = 0.1
            #     action = np.clip(np.random.normal(action, seed), 0, 1)
            #     print(f"action use random seed {seed}  with step_ita:{self.step_ita}")
            # else:
            if self.step_ita > self.epsilon_rand_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_rand_decay_step == 0:
                    self.epsilon = self.epsilon * self.epsilon_decay
            noise = np.random.randn(self.action_num) * self.epsilon
            action = noise + (1 - self.epsilon) * action
            print(f"random noise:{noise}, epsilon:{self.epsilon}, step_ita:{self.step_ita}")
            action = np.clip(action, [0., 0.], [1., 1.])
        # elif explore:
        #     noise = np.random.randn(self.action_num) * self.epsilon_end
        #     action = noise + (1 - self.epsilon_end) * action
        #     action = np.clip(action, [0., 0.], [1., 1.])
        # # print("act result:", action.tolist(), raw_snn_action.tolist())
        return action.tolist(), raw_snn_action.tolist(), h_0


    def _from_minibatch_to_replay(self, state):  # from np: batch_size x 6 x 480 x 640 --> np: batch_window x batch_size x channels x height x width
        """
        We need to transform the data from minibatch to replay
        """
        tmp_state = state.transpose(1,0,2,3)   # 6 x batch_size x 480 x 640
        tmp_state = tmp_state[:,:,np.newaxis,:,:]
        
        return tmp_state


    def replay(self):  # state_batch, action_batch, reward_batch, nstate_batch, done_batch, state_spikes_batch, nstate_spikes_batch
        """
        Experience Replay Training
        :return: actor_loss_item, critic_loss_item
        """
        state_batch, action_batch, reward_batch, nstate_batch, done_batch, state_spikes_batch, nstate_spikes_batch = self._random_minibatch()
        '''
        Compuate Target Q Value
        '''

        '''
        Check their device
        '''
        hidden_size = 256
        h_0 = torch.zeros(self.batch_size, hidden_size).to(self.device)
        # print('device:', state_batch.device, action_batch.device, reward_batch.device, nstate_batch.device, done_batch.device, state_spikes_batch.device, nstate_spikes_batch.device)
        with torch.no_grad():
            #naction_batch, _ = self.target_actor_net(nstate_spikes_batch, h_0, self.batch_size)
            nstate_spikes_batch = self.target_actor_net.pre_process_input(nstate_spikes_batch)
            self.target_actor_net.input(nstate_spikes_batch)
            self.target_actor_net.run(self.run_time)
            naction_batch = self.target_actor_net.output.predict

            next_q = self.target_critic_net([nstate_batch, naction_batch])
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)
        '''
        Update Critic Network
        '''
        self.critic_optimizer.zero_grad()

        current_q = self.critic_net([state_batch, action_batch])
        # print('q value:', current_q.device, target_q.device)
        critic_loss = self.criterion(current_q, target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        self.critic_optimizer.step()
        '''
        Update Actor Network
        '''
        # self.actor_optimizer.zero_grad()
        # current_action, _ = self.actor_net(state_spikes_batch, h_0, self.batch_size)

        self.actor_net.learner.optim_zero_grad()
        state_spikes_batch = self.actor_net.pre_process_input(state_spikes_batch)
        self.actor_net.input(state_spikes_batch)
        self.actor_net.run(self.run_time)
        current_action = self.actor_net.output.predict
        
        actor_loss = -self.critic_net([state_batch, current_action])
        print("action loss mean:",actor_loss.mean())
        actor_loss = actor_loss.mean()
        actor_loss_item = actor_loss.item()
        actor_loss.backward(retain_graph=True)
        self.actor_net.learner.optim_step()
        '''
        Update Target Networks
        '''
        self.step_ita += 1
        if self.step_ita % self.target_update_steps == 0:
            # self._soft_update(self.target_actor_net, self.actor_net)
            self._soft_update_snnflow(self.target_actor_net, self.actor_net)
            self._soft_update(self.target_critic_net, self.critic_net)
        return actor_loss_item, critic_loss_item

    def reset_epsilon(self, new_epsilon, new_decay):
        """
        Set Epsilon to a new value
        :param new_epsilon: new epsilon value
        :param new_decay: new epsilon decay
        """
        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay

    def save(self, save_dir, episode, run_name):
        """
        Save SNN Actor Net weights
        :param save_dir: directory for saving weights
        :param episode: number of episode
        :return: max_w, min_w, max_bias, min_bias, shape_w, shape_bias
        """
        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory", save_dir, " already exists")
        # self.actor_net.to('cpu')
        self.critic_net.to('cpu')

        # save_path_actor  = save_dir + '/' + run_name + 'SNN_actor_weights_' + str(episode) + '.pth'
        save_path_critic = save_dir + '/' + run_name + 'SNN_critic_weights_' + str(episode) + '.pth'

        # torch.save({'optimizer_dict': self.actor_optimizer.state_dict(),
        #             'model_dict': self.actor_net.state_dict()},
        #             save_path_actor)
        print("save actor net dir:", save_dir)
        self.actor_net.save_state(filename=f"actor_net_{episode}", direct=save_dir)
        self.actor_net.save_state(filename=f"actor_net", direct=save_dir)
        torch.save({'optimizer_dict': self.critic_optimizer.state_dict(),
                    'model_dict': self.critic_net.state_dict()},
                    save_path_critic)        

        # model_dict_actor = self.actor_net.state_dict()
        # model_dict_critic = self.critic_net.state_dict()
        # torch.save(model_dict_actor, save_dir + '/' + run_name + 'SNN_actor_weights_' + str(episode) + '.pth')
        # torch.save(model_dict_critic, save_dir + '/' + run_name + 'SNN_critic_weights_' + str(episode) + '.pth')
        # self.actor_net.to(self.device)
        self.critic_net.to(self.device)

    def _state_2_state_spikes(self, spike_state_value, batch_size):
        """
        Transform state to spikes of input neurons
        :param spike_state_value: state from environment transfer to firing rates of neurons
        :param batch_size: batch size
        :return: state_spikes
        """
        spike_normal, spike_scan = spike_state_value[0], spike_state_value[1]

        spike_normal = spike_normal.reshape((-1, 6, 1))    # batch_size x normal_state_num x 1
        normal_state_spikes = np.random.rand(batch_size, 6, self.batch_window) < spike_normal
        normal_state_spikes = normal_state_spikes.astype(float)

        spike_scan = spike_scan.reshape((-1, 1, 360, 1))   # batch_size x channels x width x batch_window
        scan_state_spikes = np.random.rand(batch_size, 1, 360, self.batch_window) < spike_scan
        scan_state_spikes = scan_state_spikes.astype(float)

        return normal_state_spikes, scan_state_spikes

    def _random_minibatch(self):   # state, spike_state, action, reward1, reward2, reward3, reward4, next_state4, next_spike_state4, done4
        """
        Random select mini-batch from memory
        :return: state_batch, action_batch, reward_batch, nstate_batch, done_batch
        """
        minibatch = random.sample(self.memory, self.batch_size)
        normal_state_batch               = np.zeros((self.batch_size, 4))
        scan_state_batch                 = np.zeros((self.batch_size, 1, 360))
        spike_normal_state_value_batch   = np.zeros((self.batch_size, 6))
        spike_scan_state_value_batch     = np.zeros((self.batch_size, 1, 360))
        action_batch                     = np.zeros((self.batch_size, 2))
        reward_batch                    = np.zeros((self.batch_size, 1))
        normal_nstate_batch             = np.zeros((self.batch_size, 4))
        scan_nstate_batch               = np.zeros((self.batch_size, 1, 360))
        spike_normal_nstate_value_batch = np.zeros((self.batch_size, 6))
        spike_scan_nstate_value_batch   = np.zeros((self.batch_size, 1, 360))
        done_batch                      = np.zeros((self.batch_size, 1))

        for num in range(self.batch_size):
            # print('size: ',np.array(minibatch[num][0][0]).shape)
            normal_state_batch[num, :]              = np.array(minibatch[num][0][0])
            scan_state_batch[num, :]                = np.array(minibatch[num][0][1]) 
            spike_normal_state_value_batch[num, :]  = np.array(minibatch[num][1][0])
            spike_scan_state_value_batch[num, :]    = np.array(minibatch[num][1][1]) 
            action_batch[num, :]                    = np.array(minibatch[num][2])
            reward_batch[num, 0]                    = minibatch[num][3]
            normal_nstate_batch[num, :]             = np.array(minibatch[num][4][0])
            scan_nstate_batch[num, :]               = np.array(minibatch[num][4][1])
            spike_normal_nstate_value_batch[num, :] = np.array(minibatch[num][5][0])
            spike_scan_nstate_value_batch[num, :]   = np.array(minibatch[num][5][1])
            done_batch[num, 0]                      = minibatch[num][6]

        # normal_state_spikes_batch, scan_state_spikes_batch = \
        #     self._state_2_state_spikes([spike_normal_state_value_batch, spike_scan_state_value_batch], self.batch_size)

        # normal_nstate_spikes_batch, scan_nstate_spikes_batch = \
        #     self._state_2_state_spikes([spike_normal_nstate_value_batch, spike_scan_nstate_value_batch], self.batch_size)
        normal_state_spikes_batch, scan_state_spikes_batch = spike_normal_state_value_batch, spike_scan_state_value_batch
        normal_nstate_spikes_batch, scan_nstate_spikes_batch = spike_normal_nstate_value_batch, spike_scan_nstate_value_batch

        normal_state_batch  = torch.Tensor(normal_state_batch).to(self.device)
        scan_state_batch    = torch.Tensor(scan_state_batch).to(self.device)
        action_batch        = torch.Tensor(action_batch).to(self.device)
        reward_batch        = torch.Tensor(reward_batch).to(self.device)
        normal_nstate_batch = torch.Tensor(normal_nstate_batch).to(self.device)
        scan_nstate_batch   = torch.Tensor(scan_nstate_batch).to(self.device)
        done_batch          = torch.Tensor(done_batch).to(self.device)
        normal_state_spikes_batch   = torch.Tensor(normal_state_spikes_batch).to(self.device)  # batch_size x 6 x batch_window
        scan_state_spikes_batch     = torch.Tensor(scan_state_spikes_batch).to(self.device)    # batch_size x 1 x 5 x 360 x batch_window
        normal_nstate_spikes_batch  = torch.Tensor(normal_nstate_spikes_batch).to(self.device) # batch_size x 6 x batch_window
        scan_nstate_spikes_batch    = torch.Tensor(scan_nstate_spikes_batch).to(self.device)   # batch_size x 1 x 5 x 360 x batch_window 

        state_batch         = [normal_state_batch, scan_state_batch]
        nstate_batch        = [normal_nstate_batch, scan_nstate_batch]
        state_spikes_batch  = [normal_state_spikes_batch, scan_state_spikes_batch]   
        nstate_spikes_batch = [normal_nstate_spikes_batch, scan_nstate_spikes_batch] 

        return state_batch, action_batch, reward_batch, nstate_batch, done_batch, state_spikes_batch, nstate_spikes_batch

    def _hard_update(self, target, source):
        """
        Hard Update Weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        """
        Soft Update weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau
                )

    def _hard_update_snnflow(self, target, source, env_name='actornet'):
        target.update_parameters(source)

    def _soft_update_snnflow(self, target, source,env_name='actornet'):
        target.update_parameters(source, ratio=self.target_tau)