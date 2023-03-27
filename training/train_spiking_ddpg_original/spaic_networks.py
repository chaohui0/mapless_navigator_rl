from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np
import spaic
from training.train_spiking_ddpg.spaic.Learning.Learner import Learner
from spaic.Learning.STCA_Learner import STCA
from spaic.Learning.Learner import Learner
from spaic.Network import Network


class ActorNetSpiking(spaic.Network):

    def __init__(self, node_num=480*640, action_num=2):
        super(ActorNetSpiking, self).__init__()

        self.input_pos = spaic.Encoder(shape=[1, 480, 640], num=node_num, coding_method='null')
        self.layer1    = spaic.NeuronGroup(10 * 240 * 320, neuron_shape=None, neuron_model='lif')
        self.layer2    = spaic.NeuronGroup(4 * 120 * 160, neuron_shape=None, neuron_model='lif')
        self.layer3    = spaic.NeuronGroup(4 * 30 * 40, neuron_shape=None, neuron_model='lif')
        self.layer4    = spaic.NeuronGroup(action_num, neuron_model='lif')
        self.output    = spaic.Decoder(num=action_num, dec_target=self.layer4, coding_method='spike_counts')

        self.input_neg = spaic.Encoder(shape=[1, 480, 640], num=node_num, coding_method='null')
        self.layer1    = spaic.NeuronGroup(10 * 240 * 320, neuron_shape=None, neuron_model='lif')
        self.layer2    = spaic.NeuronGroup(4 * 120 * 160, neuron_shape=None, neuron_model='lif')
        self.layer3    = spaic.NeuronGroup(4 * 30 * 40, neuron_shape=None, neuron_model='lif')
        self.layer4    = spaic.NeuronGroup(action_num, neuron_model='lif')
        self.output    = spaic.Decoder(num=action_num, dec_target=self.layer4, coding_method='spike_counts')


        # connection
        self.connection1 = spaic.Connection(self.input_pos, self.layer1, link_type='conv', in_channels=1, \
                                            out_chnnels=10, kernel_size=(4, 4), stride=2, padding=1, w_std=0.05, w_mean=0.05)
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='conv', in_channels=10, \
                                            out_channels=4, kernel_size=(4, 4), stride=2, padding=1, w_std=0.05, w_mean=0.05)
        self.connection3 = spaic.Connection(self.layer2, self.layer3, link_type='conv', in_channels=4, \
                                            out_channels=4, kernel_size=(4, 4), stride=2, padding=1, w_std=0.05, w_mean=0.05)
        self.connection4 = spaic.Connection(self.layer3, self.layer4, link_type='full', syn_type=['flatten', 'basic_synapse'], w_std=0.05, w_mean=0.05)
        self._learners = Learner(algorithm='STCA', a=0.01, trainable=[self.connection1, self.connection2, self.connection3, self.connection4, \
                                                                      self.layer1, self.layer2, self.layer3, self.layer4])
        self._learners.set_optimizer('Adam', 0.001)