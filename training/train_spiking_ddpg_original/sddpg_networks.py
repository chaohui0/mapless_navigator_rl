from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np


NEURON_VTH = 0.5
NEURON_CDECAY = 0.5
NEURON_VDECAY = 0.75
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()           # 大于 NEURON_VTH 就发放脉冲
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class ActorNetSpiking(nn.Module):
    """ Spiking Actor Network """
    def __init__(self, state_num, action_num, device, batch_window=50, hidden1=256, hidden2=256, hidden3=128):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param device: device used
        :param batch_window: window steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNetSpiking, self).__init__()
        self.state_num = 216  # 210 scan state & 6 normal state
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.pseudo_spike = PseudoSpikeRect.apply
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=(5, ), stride=2, padding=0, bias=True)  # batch x 5 x 178
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=(5, ), stride=2, padding=0, bias=True)  # batch x 5 x 87
        self.conv3 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=(5, ), stride=2, padding=0, bias=True)  # batch x 5 x 42
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.state_num, self.hidden1, bias=True)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=True)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3, bias=True)
        self.fc4 = nn.Linear(self.hidden3, self.action_num, bias=True)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        # print('size: ', syn_func(pre_layer_output).shape)
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, spikes_state, batch_size):
        """

        :param x: state batch
        :param batch_size: size of batch
        :return: out
        """
        scan_spikes = spikes_state[1]
        normal_spikes = spikes_state[0]

        cv1_u = torch.zeros(batch_size, 5, 178, device=self.device)
        cv1_v = torch.zeros(batch_size, 5, 178, device=self.device)
        cv1_s = torch.zeros(batch_size, 5, 178, device=self.device)

        cv2_u = torch.zeros(batch_size, 5, 87, device=self.device)
        cv2_v = torch.zeros(batch_size, 5, 87, device=self.device)
        cv2_s = torch.zeros(batch_size, 5, 87, device=self.device)

        cv3_u = torch.zeros(batch_size, 5, 42, device=self.device)
        cv3_v = torch.zeros(batch_size, 5, 42, device=self.device)
        cv3_s = torch.zeros(batch_size, 5, 42, device=self.device)

        fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_u = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_v = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_s = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc3_u = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_v = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_s = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc4_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_s = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)

        for step in range(self.batch_window):

            input_normal_spike = normal_spikes[:, :, step]    # input_normal_spike = batch_size x 6 x batch_window 
            input_scan_spike = scan_spikes[:, :, :, step]  # input_scan_spike   = batch_size x 1 x 360 x batch_window
            #print('input_scan_spike shape: ', input_scan_spike.shape)
            cv1_u, cv1_v, cv1_s = self.neuron_model(self.conv1, input_scan_spike, cv1_u, cv1_v, cv1_s)
            cv2_u, cv2_v, cv2_s = self.neuron_model(self.conv2, cv1_s, cv2_u, cv2_v, cv2_s)
            cv3_u, cv3_v, cv3_s = self.neuron_model(self.conv3, cv2_s, cv3_u, cv3_v, cv3_s)
            output_scan = self.flatten1(cv3_s)
            # print('output_scan shape: ', output_scan.shape)
            combined_data = torch.cat([output_scan, input_normal_spike], axis=1)
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, combined_data, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc3_u, fc3_v, fc3_s = self.neuron_model(self.fc3, fc2_s, fc3_u, fc3_v, fc3_s)
            fc4_u, fc4_v, fc4_s = self.neuron_model(self.fc4, fc3_s, fc4_u, fc4_v, fc4_s)
            fc4_sumspike += fc4_s
        out = fc4_sumspike / self.batch_window
        return out


class CriticNetSpiking(nn.Module): 
    """ Critic Network"""
    def __init__(self, state_num, action_num, hidden1=256, hidden2=256, hidden3=128):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(CriticNetSpiking, self).__init__()
        self.state_num = 216
        self.action_num = action_num
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3

        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=(5, ), stride=2, padding=0, bias=True),   # batch x 5 x 178
            nn.ReLU(),
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=(5, ), stride=2, padding=0, bias=True),   # batch x 5 x 87
            nn.ReLU(),
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=(5, ), stride=2, padding=0, bias=True),   # batch x 5 x 42
            nn.Flatten(),
            nn.ReLU()
        )

        self.net2 = nn.Sequential(
            nn.Linear(self.state_num, self.hidden1, bias=True),  # self.state_num = 356 + 4
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden2, self.hidden3, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden3, 1, bias=True),
        )


    def forward(self, xa):  # in: depth, state(from /gazebo/get_model_state) and action

        normal_state, scan_state, action = xa[0][0], xa[0][1], xa[1]
        # normal_state, scan_state = state[0], state[1]
        scan_state = self.net1(scan_state)
        out = self.net2(torch.cat([scan_state, normal_state, action], 1))

        return out


if __name__ == "__main__":

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')
    actor_net = ActorNetSpiking(combined_num=4804, action_num=2, device=device).to(device)
    critic_net = CriticNetSpiking(normal_state_num=4, action_num=2).to(device)

    # evaluation for critic network
    with torch.no_grad():
        spike_value_img = np.random.rand(1, 1, 480, 640)
        state_spike_img = spike_value_img.astype(float)
        state_spike_img = torch.Tensor(state_spike_img).to(device)
        normal_state = np.random.rand(1, 4)  # batch_size x normal_num
        normal_state = torch.Tensor(normal_state).to(device)
        action_state = np.random.rand(1, 2)  # batch_size x action_num
        action_state = torch.Tensor(action_state).to(device)
        xa = [state_spike_img, normal_state, action_state]
        output = critic_net(xa)
        print(output)
