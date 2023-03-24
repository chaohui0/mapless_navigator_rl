from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np
import sys
import copy
sys.path.append("/data/workspace/SPAIC_COMPLETE")
import spaic
class ActorNetSpiking(spaic.Network):
    def __init__(self, state_num=366,action_num=2,device="cuda"):
        super(ActorNetSpiking, self).__init__()
        # coding
        self.input = spaic.Encoder(num=state_num, coding_method='poisson')
        # 20230116目前编译器只支持IF模型
        self.layer1 = spaic.NeuronGroup(128, model='if')
        self.layer2 = spaic.NeuronGroup(128, model='if')
        self.layer3 = spaic.NeuronGroup(128, model='if')
        self.layer4 = spaic.NeuronGroup(action_num, model='if')
        # decoding
        self.output = spaic.Decoder(num=action_num, dec_target=self.layer4, coding_method='spike_counts')

        w_mean = 0.03
        w_std = 0.005
        # Connection
        self.connection1 = spaic.Connection(pre=self.input, post=self.layer1, link_type='full',w_mean=w_mean, w_std=w_std)
        self.connection2 = spaic.Connection(pre=self.layer1, post=self.layer2, link_type='full',w_mean=w_mean, w_std=w_std)
        self.connection3 = spaic.Connection(pre=self.layer2, post=self.layer3, link_type='full',w_mean=w_mean, w_std=w_std)
        self.connection4 = spaic.Connection(pre=self.layer3, post=self.layer4, link_type='full',w_mean=w_mean, w_std=w_std)
        # Learner
        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        self.learner.set_optimizer('Adam', 0.001)
        if torch.cuda.is_available() and device == "cuda":
            device = 'cuda'
        else:
            device = 'cpu'
        backend = spaic.Torch_Backend(device)
        backend.dt = 0.1
        self.set_backend(backend)

        self.mon_V1 = spaic.StateMonitor(self.layer1, 'V')
        self.mon_V2 = spaic.StateMonitor(self.layer2, 'V')
        self.mon_V3 = spaic.StateMonitor(self.layer3, 'V')
        self.mon_V4 = spaic.StateMonitor(self.layer4, 'V')
        self.mon_O1 = spaic.StateMonitor(self.layer1, 'O')
        self.mon_O2 = spaic.StateMonitor(self.layer2, 'O')
        self.mon_O3 = spaic.StateMonitor(self.layer3, 'O')
        self.mon_O4 = spaic.StateMonitor(self.layer4, 'O')
        self.mon_I = spaic.StateMonitor(self.input, 'O')
        self.spk_O = spaic.SpikeMonitor(self.layer4)

    def show_monitor(self):
        # print("mon_V1.value:", self.mon_V1.values)
        # print("mon_V1.times:", self.mon_V1.times)

        # print("mon_V2.value:", self.mon_V2.values)
        # print("mon_V2.times:", self.mon_V2.times)

        # print("mon_V3.value:", self.mon_V3.values)
        # print("mon_V3.times:", self.mon_V3.times)

        # print("mon_V4.value:", self.mon_V4.values)
        # print("mon_V4.times:", self.mon_V4.times)


        # print("mon_O1.value:", self.mon_O1.values)
        # print("mon_O1.times:", self.mon_O1.times)

        # print("mon_O2.value:", self.mon_O2.values)
        # print("mon_O2.times:", self.mon_O2.times)

        # print("mon_O3.value:", self.mon_O3.values)
        # print("mon_O3.times:", self.mon_O3.times)

        # print("mon_O4.value:", self.mon_O4.values)
        # print("mon_O4.times:", self.mon_O4.times)

        spikes = self.spk_O.spk_index[0]
        sptime = self.spk_O.spk_times[0]
        if isinstance(spikes, np.ndarray):
            spikes = spikes.tolist()
            sptime = sptime.tolist()
            print("spikes:", spikes)
            print("sptime:", sptime)

    def pre_process_input(self, state):
        if len(state[0].shape) == 1:
            batch_size=1
            normal_size = state[0].shape[0]
        else:
            batch_size, normal_size = state[0].shape

        if len(state[1].shape) == 1:
            batch_size=1
            scan_size = state[1].shape[0]
        elif len(state[1].shape) == 2:
            batch_size, scan_size =  state[1].shape
        elif len(state[1].shape) == 3:
            batch_size, _, scan_size =  state[1].shape

        p_state = np.zeros((batch_size,normal_size+scan_size))
        for num in range(batch_size):
            if torch.is_tensor(state[0][num]):
                p_state[num, 0:normal_size] = state[0][num].to("cpu")
                p_state[num, normal_size:] = state[1][num].to("cpu")
            else:
                p_state[num, 0:normal_size] = state[0][:]
                p_state[num, normal_size:] = state[1][:]
        return p_state

    def update_parameters(self, source_net, ratio=1):
        #根据指定网络的参数迭代
        for key, par in self._backend._parameters_dict.items():
            backend_key = source_net._backend.check_key(key, source_net._backend._parameters_dict)
            if backend_key in source_net._backend._parameters_dict:
                src_param = source_net._backend._parameters_dict[backend_key]
                # print(backend_key)
                # print(src_param)
                self._backend._parameters_dict[key] = src_param * ratio + par * (1 - ratio)

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
            nn.ReLU(),
            nn.Flatten()
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

if __name__ == '__main__':
    actor_net=ActorNetSpiking()
    actor_net.build()
    # actor_net.state_from_dict(filename="actor_net", direct="../save_weights_real_GRU_original", device="cuda")

    target_actor_net=ActorNetSpiking()
    target_actor_net.build()

    batch_size = 1

    normal_nstate_spikes_batch = np.ones((batch_size, 6),dtype=float)*0.1
    scan_nstate_spikes_batch   = np.ones((batch_size,1, 360),dtype=float)*-0.2

    # normal_nstate_spikes_batch = torch.Tensor(normal_nstate_spikes_batch).to("cuda")
    # scan_nstate_spikes_batch = torch.Tensor(scan_nstate_spikes_batch).to("cuda")

    # normal_nstate_spikes_batch = np.array([0.5 for _ in range(6)])
    state_spikes = [normal_nstate_spikes_batch, scan_nstate_spikes_batch]
    state_spikes = [np.array([0.40444791, 0.        , 0.08173225, 2.45610285, 0.        ,
       1.95795918]), np.array([[0.24192656, 0.24711158, 0.24880843, 0.25094916, 0.25218363,
        0.25794731, 0.25817121, 0.26221906, 0.26531317, 0.26802203,
        0.26880289, 0.27163137, 0.27388059, 0.27741647, 0.28031036,
        0.27956212, 0.28586   , 0.2852933 , 0.28900072, 0.2916852 ,
        0.29056934, 0.2897451 , 0.29629857, 0.29536403, 0.29742237,
        0.29895439, 0.30078313, 0.29951914, 0.30362503, 0.30416518,
        0.30705943, 0.30388481, 0.3046301 , 0.30800623, 0.30980539,
        0.30650882, 0.30595271, 0.30886695, 0.30823326, 0.30737582,
        0.30900497, 0.30909647, 0.30793913, 0.30907527, 0.30683852,
        0.30961697, 0.30868308, 0.30506256, 0.30464949, 0.30223689,
        0.30434576, 0.30101568, 0.30125893, 0.30226843, 0.29963538,
        0.29887996, 0.29676646, 0.29649281, 0.29406414, 0.29259859,
        0.29180088, 0.29112573, 0.2880424 , 0.28779431, 0.28290469,
        0.28149889, 0.27827631, 0.27794443, 0.27539297, 0.272417  ,
        0.27093561, 0.26694942, 0.26566512, 0.26147217, 0.25809958,
        0.25636675, 0.25308977, 0.24991828, 0.24847355, 0.24366279,
        0.23892802, 0.23932844, 0.23417021, 0.23040947, 0.22727147,
        0.22387392, 0.22144094, 0.2163881 , 0.21406458, 0.20855902,
        0.20576008, 0.20172186, 0.21709115, 0.22455802, 0.22731614,
        0.23153857, 0.23291424, 0.23662096, 0.23414267, 0.2347653 ,
        0.23320859, 0.2324555 , 0.23037316, 0.22725788, 0.2224657 ,
        0.2134105 , 0.13872661, 0.13416591, 0.12914851, 0.12466528,
        0.11932471, 0.11497888, 0.10977741, 0.10570064, 0.1007408 ,
        0.09574556, 0.022     , 0.022     , 0.022     , 0.022     ,
        0.10829798, 0.10996737, 0.11048209, 0.11019585, 0.10949762,
        0.10844572, 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.09269433, 0.09405379, 0.09516994, 0.09580118,
        0.09574809, 0.09602097, 0.0959257 , 0.09479531, 0.0942441 ,
        0.09234374, 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.022     , 0.022     , 0.022     , 0.14353406,
        0.14614015, 0.14807597, 0.14856359, 0.14960305, 0.14945526,
        0.14786683, 0.14643354, 0.14382556, 0.13345264, 0.13394164,
        0.13417696, 0.13443207, 0.13436481, 0.1344563 , 0.13310099,
        0.13216337, 0.13036256, 0.1284691 , 0.12439564, 0.022     ,
        0.10845075, 0.11148355, 0.11390796, 0.11515855, 0.11569839,
        0.11684132, 0.1170146 , 0.11641603, 0.11595041, 0.1157956 ,
        0.12447774, 0.13021862, 0.13321096, 0.13511612, 0.13658133,
        0.13740387, 0.13816751, 0.13867771, 0.13894413, 0.13786423,
        0.1376996 , 0.13645083, 0.13496982, 0.13371422, 0.13224752,
        0.12811657, 0.022     , 0.022     , 0.022     , 0.022     ,
        0.022     , 0.022     , 0.022     , 0.022     , 0.2310028 ,
        0.2383469 , 0.24445776, 0.2510454 , 0.25611721, 0.26076173,
        0.2638768 , 0.26341233, 0.26729932, 0.26868667, 0.27300394,
        0.27218153, 0.27183597, 0.27212634, 0.27206141, 0.27193922,
        0.27113684, 0.27073389, 0.26727749, 0.26678663, 0.26363865,
        0.26151876, 0.39339955, 0.40971021, 0.41094308, 0.42518477,
        0.42889821, 0.4333629 , 0.44003256, 0.44304774, 0.44496139,
        0.45164607, 0.45032112, 0.45604704, 0.44653733, 0.44926284,
        0.43915651, 0.44254238, 0.44046261, 0.43369629, 0.43014789,
        0.42127855, 0.41288205, 0.40415914, 0.39028191, 0.3151504 ,
        0.31710097, 0.31213339, 0.31204208, 0.30799651, 0.30668088,
        0.29375862, 0.2903756 , 0.20666851, 0.20967681, 0.21307916,
        0.21494052, 0.21645593, 0.219919  , 0.22284968, 0.22685891,
        0.22884252, 0.23007201, 0.23110648, 0.23367356, 0.23619834,
        0.23789867, 0.23844947, 0.24279723, 0.24280001, 0.24392156,
        0.24478127, 0.24976824, 0.24934959, 0.25222152, 0.25287739,
        0.25119137, 0.25672023, 0.25500654, 0.25497215, 0.25753356,
        0.25788834, 0.25735721, 0.25919513, 0.26178402, 0.26146997,
        0.26533086, 0.26412237, 0.26328285, 0.26430212, 0.26028703,
        0.26363489, 0.26475181, 0.264005  , 0.26255571, 0.26535647,
        0.26543522, 0.26275661, 0.26151405, 0.26103927, 0.26315214,
        0.26219438, 0.26298495, 0.26067951, 0.25794163, 0.25490969,
        0.25675312, 0.25659253, 0.2568976 , 0.25230353, 0.25173427,
        0.25048422, 0.25109305, 0.24662332, 0.24712241, 0.24624757,
        0.24483076, 0.24363633, 0.24116697, 0.23997707, 0.2385591 ,
        0.23513093, 0.23251688, 0.23248726, 0.22980583, 0.22491807,
        0.22490462, 0.22087177, 0.22050971, 0.21696695, 0.21584613]])]




    # state_spikes=np.ones((10,366),dtype=float)*0.5
    run_time = 50
    with torch.no_grad():
        state_spikes = actor_net.pre_process_input(state_spikes)
        print(state_spikes)
        actor_net.input(state_spikes)
        actor_net.run(run_time)
        print(actor_net.output.predict)
        # action = actor_net.output.predict
        action = (actor_net.output.predict).to('cpu')
        action = action.numpy().squeeze()
        raw_snn_action = copy.deepcopy(action)

    # epsilon=0.1
    # noise = np.random.randn(2) * epsilon
    # action = noise + (1 - epsilon) * action
    # action = np.clip(action, [0., 0.], [1., 1.])
    target_actor_net.update_parameters(actor_net, ratio=0.1)
    print(action.tolist(), raw_snn_action.tolist())
    # actor_net.save_state(filename="actor_net_20", direct="../save_weights_real_GRU_original")
