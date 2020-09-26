import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
        Actor network (policy) for DC-DRL,
        works as a mapping from state/observation to target action
    """
    def __init__(self,
                 obs_space,
                 act_space,
                 net_dims,
                 net_archs,
                 activate,
                 extractor=None,
                 raw_obs_space=None):
        super(Actor, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_dim = self.act_space.shape[0]
        self.actor_net_dims = net_dims
        self.actor_net_archs = net_archs
        self.actor_activate = activate

        self.modules = self.create_modules()

        self.layer_module = nn.ModuleList(self.modules)

        if extractor is not None:
            self.pre_extractor = self.CNNExtractor(raw_obs_space, obs_space.shape[0])
        else:
            self.pre_extractor = None


    def _get_data(self):
        # save hyper-parametrs to reproduce
        data = dict(
            # the processed observation, not always equals to the raw
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            actor_net_dims=self.actor_net_dims,
            actor_net_archs=self.actor_net_archs,
            actor_activate=self.actor_activate,
            pre_extractor=self.pre_extractor,
        )
        return data

    def create_modules(self):
        # create modules with provided parameters
        if len(self.actor_net_dims) == 0:
            linear = nn.Linear(self.obs_dim, self.act_dim)
            linear.weight.data.mul_(0.1)
            linear.bias.data.mul_(0.1)
            modules = [linear]
        else:
            modules = []

            linear_0 = nn.Linear(self.obs_dim, self.actor_net_dims[0])
            ln_0 = nn.LayerNorm(self.actor_net_dims[0])
            modules.append(linear_0)
            modules.append(ln_0)

            for idx in range(len(self.actor_net_dims) - 1):
                linear = nn.Linear(self.actor_net_dims[idx], self.actor_net_dims[idx + 1])
                ln = nn.LayerNorm(self.actor_net_dims[idx + 1])
                modules.append(linear)
                modules.append(ln)

            mu = nn.Linear(self.actor_net_dims[-1], self.act_dim)
            mu.weight.data.mul_(0.1)
            mu.bias.data.mul_(0.1)
            modules.append(mu)

        return modules

    def forward(self, inputs):
        obs_features = self.extract_features(inputs)

        if len(self.layer_module) == 1:
            if self.actor_activate == 'tanh':
                mu = torch.tanh(self.layer_module[0](obs_features))
            else:
                raise NotImplementedError()
        else:
            for idx in range(0, 2 * len(self.actor_net_dims), 2):
                obs_features = self.actor_net_archs[int(idx/2)](self.layer_module[idx + 1](self.layer_module[idx](obs_features)))

            if self.actor_activate == 'tanh':
                mu = torch.tanh(self.layer_module[-1](obs_features))
            else:
                raise NotImplementedError()

        return mu

    class CNNExtractor(nn.Module):
        # develop
        def __init__(self,
                     raw_obs_space,
                     fea_dim):
            super(Actor.CNNExtractor, self).__init__()
            n_input_channels = raw_obs_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(raw_obs_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, fea_dim), nn.ReLU())

        def forward(self, obs):
            return self.linear(self.cnn(obs))


    def extract_features(self, inputs):
        if self.pre_extractor is not None:
            preprocessed_obs = self.pre_extractor(inputs)
            return self.features_extractor(preprocessed_obs)
        else:
            return inputs


class Critic(nn.Module):
    """
        Critic network (policy) for DC-DRL
        works as a mapping from (s,a) to long-term reward
    """
    def __init__(self,
                 obs_space,
                 act_space,
                 net_dims,
                 net_archs,
                 activate,
                 extractor=None,
                 raw_obs_space=None):
        super(Critic, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_dim = self.act_space.shape[0]
        self.critic_net_dims = net_dims
        self.critic_net_archs = net_archs
        self.critic_activate = activate

        self.modules = self.create_modules()
        self.layer_module = nn.ModuleList(self.modules)

        if extractor is not None:
            self.pre_extractor = self.CNNExtractor(raw_obs_space, obs_space.shape[0])
        else:
            self.pre_extractor = None

    def _get_data(self):
        # save hyper-parametrs to reproduce
        data = dict(
            # the processed observation, not always equals to the raw
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            critic_net_dims=self.critic_net_dims,
            critic_net_archs=self.critic_net_archs,
            critic_activate=self.critic_activate,
            pre_extractor=self.pre_extractor,
        )
        return data

    def create_modules(self):
        # create modules with provided parameters
        # cmp actor network
        if len(self.critic_net_dims) == 0:
            # at least to divide s & a
            raise ValueError()
        else:
            modules = []

            h1_size = self.critic_net_dims[0]
            linear1 = nn.Linear(self.obs_dim, h1_size)
            ln1 = nn.LayerNorm(h1_size)

            h2_size = self.critic_net_dims[1]
            linear2 = nn.Linear(h1_size + self.act_dim, h2_size)
            ln2 = nn.LayerNorm(h2_size)

            modules.append(linear1)
            modules.append(ln1)
            modules.append(linear2)
            modules.append(ln2)

            for idx in range(1, len(self.critic_net_dims)-1):
                linear = nn.Linear(self.critic_net_dims[idx], self.critic_net_dims[idx + 1])
                ln = nn.LayerNorm(self.critic_net_dims[idx + 1])
                modules.append(linear)
                modules.append(ln)

            V = nn.Linear(self.critic_net_dims[-1], 1)
            V.weight.data.mul_(0.1)
            V.bias.data.mul_(0.1)

            modules.append(V)

        return modules

    def forward(self, inputs, actions):
        obs_features = self.extract_features(inputs)

        if len(self.layer_module) == 2:
            x = self.critic_net_archs[0](self.layer_module[1](self.layer_module[0](obs_features)))
            x = torch.cat((x, actions), 1)
            x = self.critic_net_archs[1](self.layer_module[3](self.layer_module[2](x)))
            if self.critic_activate == 'mul':
                v = self.layer_module[-1](x)
            else:
                raise NotImplementedError()
        else:
            x = self.critic_net_archs[0](self.layer_module[1](self.layer_module[0](obs_features)))
            x = torch.cat((x, actions), 1)
            x = self.critic_net_archs[1](self.layer_module[3](self.layer_module[2](x)))

            for idx in range(4, 2 * len(self.critic_net_dims), 2):
                x = self.critic_net_archs[int(idx/2)](self.layer_module[idx + 1](self.layer_module[idx](x)))

            if self.critic_activate == 'mul':
                v = self.layer_module[-1](x)
            else:
                raise NotImplementedError()

        return v

    class CNNExtractor(nn.Module):
        # develop
        def __init__(self,
                     raw_obs_space,
                     fea_dim):
            super(Critic.CNNExtractor, self).__init__()
            n_input_channels = raw_obs_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(raw_obs_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, fea_dim), nn.ReLU())

        def forward(self, obs):
            return self.linear(self.cnn(obs))


    def extract_features(self, inputs):
        if self.pre_extractor is not None:
            preprocessed_obs = self.pre_extractor(inputs)
            return self.features_extractor(preprocessed_obs)
        else:
            return inputs

